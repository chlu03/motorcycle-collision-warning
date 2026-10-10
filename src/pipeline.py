"""
End-to-end pipeline: lane detection -> YOLOv5 detection -> lane filtering ->
DeepSORT tracking -> speed estimation -> TTC risk assessment -> overlay -> warning sound.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .audio import export_video, select_warning_frames
from .config import ROOT, load_config, resolve, ttc_threshold
from .lane_detection import detect_lanes, lane_polygon, load_lanes
from .lane_filter import filter_detections
from .overlay import draw_info_panel, draw_lane_points, draw_lanes, draw_target_box
from .risk import analyze_target
from .speed import SpeedEstimator, track_centers

# Third-party code is imported from third_party/. YOLOv5 must be importable as top-level
# `models` and `utils`, because the weights file stores those module paths.
for _path in (ROOT / 'third_party' / 'yolov5', ROOT / 'third_party'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

# The DeepSORT code uses aliases removed in NumPy 1.24
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float


def load_model(weights, device):
    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only argument
        checkpoint = torch.load(weights, map_location=device)
    model = checkpoint['model'].float().eval()
    if device.type != 'cpu':
        model.half()
    return model


def load_tracker(cfg, use_cuda):
    ds_cfg = get_config()
    ds_cfg.merge_from_file(str(resolve(cfg['paths']['deepsort_config'])))
    d = ds_cfg.DEEPSORT
    return DeepSort(str(resolve(cfg['paths']['reid_checkpoint'])),
                    max_dist=d.MAX_DIST, min_confidence=d.MIN_CONFIDENCE,
                    nms_max_overlap=d.NMS_MAX_OVERLAP, max_iou_distance=d.MAX_IOU_DISTANCE,
                    max_age=d.MAX_AGE, n_init=d.N_INIT, nn_budget=d.NN_BUDGET,
                    use_cuda=use_cuda)


def to_deepsort_input(detections):
    """Convert (x1, y1, x2, y2, conf, cls) rows to DeepSORT's center-based boxes."""
    xywh, confs, classes = [], [], []
    for x1, y1, x2, y2, conf, cls in detections:
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        w, h = abs(x2 - x1), abs(y2 - y1)
        xywh.append([min(x1, x2) + w / 2, min(y1, y2) + h / 2, w, h])
        confs.append([conf.item()])
        classes.append([cls.item()])
    return torch.Tensor(xywh), torch.Tensor(confs), torch.Tensor(classes)


def run(source, config_path=None, skip_lanes=False, no_audio=False, save_lane_video=True, device=None):
    """Run the full pipeline on one video. Results are saved to outputs/<video name>/.

    Returns a dict with the paths of the generated files.
    """
    t_start = time.time()
    cfg = load_config(config_path)
    source = resolve(source)
    if not source.exists():
        raise FileNotFoundError(f'Video not found: {source}')

    out_dir = resolve(cfg['paths']['output_dir']) / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'lanes_csv': out_dir / 'lanes.csv',
        'lanes_video': out_dir / 'lanes.mp4',
        'result': out_dir / 'result.mp4',
        'warnings': out_dir / 'warnings.csv',
        'tracks': out_dir / 'tracks.csv',
        'speeds': out_dir / 'speeds.csv',
    }
    silent_video = out_dir / '_silent.mp4'

    # 1. Lane detection
    if skip_lanes and paths['lanes_csv'].exists():
        print(f'Using existing lane data: {paths["lanes_csv"].relative_to(ROOT)}')
    else:
        detect_lanes(source, paths['lanes_csv'], cfg['lane'],
                     video_out=paths['lanes_video'] if save_lane_video else None)
    lanes = load_lanes(paths['lanes_csv'])

    # 2. Models
    det_cfg = cfg['detection']
    torch_device = select_device(det_cfg['device'] if device is None else device)
    half = torch_device.type != 'cpu'
    model = load_model(str(resolve(cfg['paths']['weights'])), torch_device)
    stride = int(model.stride.max())
    img_size = check_img_size(det_cfg['img_size'], s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    vehicle_classes = set(det_cfg['vehicle_classes'])
    tracker = load_tracker(cfg, use_cuda=torch_device.type != 'cpu')

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speed_estimator = SpeedEstimator(fps, cfg['speed'])
    threshold = ttc_threshold(cfg)
    warning_frames = []
    writer = None

    # 3. Detection, tracking, risk assessment, and drawing
    with torch.no_grad(), \
            open(paths['tracks'], 'w', newline='') as f_tracks, \
            open(paths['speeds'], 'w', newline='') as f_speeds:
        tracks_csv = csv.writer(f_tracks)
        tracks_csv.writerow(['frame', 'track_id', 'class', 'x_center', 'y_center', 'box_width'])
        speeds_csv = csv.writer(f_speeds)
        speeds_csv.writerow(['frame', 'track_id', 'speed_kmh'])

        frame_idx = -1
        with tqdm(total=total, desc='Vehicle detection', unit='frame') as bar:
            while True:
                ok, im0 = cap.read()
                if not ok:
                    break
                frame_idx += 1
                bar.update(1)

                # YOLOv5 inference
                img = letterbox(im0, img_size, stride=stride)[0]
                img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR to RGB, HWC to CHW
                img = torch.from_numpy(img).to(torch_device)
                img = (img.half() if half else img.float()) / 255.0
                img = img.unsqueeze(0)
                pred = model(img)[0]
                det = non_max_suppression(pred, det_cfg['conf_threshold'], det_cfg['iou_threshold'])[0]

                lane = lanes[min(frame_idx, len(lanes) - 1)]
                draw_lanes(im0, lane)

                # Keep only vehicles in the rider's lane
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    det = filter_detections(det, lane_polygon(lane))

                targets = []
                if len(det):
                    # Tracking and speed estimation
                    xywhs, confs, classes = to_deepsort_input(det)
                    outputs = tracker.update(xywhs, confs, im0, classes)
                    centers = track_centers(outputs)
                    for x, y, track_id, class_id, box_w in centers:
                        tracks_csv.writerow([frame_idx, track_id, names[class_id],
                                             round(float(x), 1), round(float(y), 1), int(box_w)])
                    if speed_estimator.update(centers):
                        for kmh, track_id in speed_estimator.speeds:
                            speeds_csv.writerow([frame_idx, track_id, kmh])

                    # Risk assessment and drawing for each vehicle
                    for *box, conf, cls in reversed(det):
                        name = names[int(cls)]
                        if float(f'{conf:.2f}') <= det_cfg['display_conf'] or name not in vehicle_classes:
                            continue
                        info = analyze_target(box, name, float(conf), outputs, speed_estimator, cfg, threshold)
                        if info['flag'] == 'High risk':
                            warning_frames.append(frame_idx)
                        draw_target_box(im0, box, info, line_thickness=3)
                        targets.append(info)

                draw_lane_points(im0, det)
                draw_info_panel(im0, targets, threshold)

                if writer is None:
                    h, w = im0.shape[:2]
                    writer = cv2.VideoWriter(str(silent_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                writer.write(im0)

    cap.release()
    if writer is not None:
        writer.release()

    # 4. Warning sounds and final video
    warning_frames = select_warning_frames(warning_frames, fps, cfg['warning']['min_gap_s'])
    warning_times = [f / fps for f in warning_frames]
    with open(paths['warnings'], 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'time_s'])
        for frame, t in zip(warning_frames, warning_times):
            w.writerow([frame, round(t, 3)])

    print('Exporting video...')
    sound = None if no_audio else resolve(cfg['paths']['warning_sound'])
    export_video(silent_video, paths['result'], sound, warning_times)
    silent_video.unlink(missing_ok=True)

    if not save_lane_video:
        paths.pop('lanes_video')
    print(f'Done in {time.time() - t_start:.1f}s: {frame_idx + 1} frames, {len(warning_times)} warnings')
    for key, path in paths.items():
        print(f'  {key:12s} {Path(path).relative_to(ROOT)}')
    return paths
