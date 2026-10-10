"""Time-to-collision (TTC) and risk assessment for one detected vehicle."""
from .distance import estimate_distance


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def match_track(box, outputs, min_iou=0.3):
    """Track ID of the DeepSORT output that best overlaps `box`, or None."""
    box = [float(v) for v in box[:4]]
    best_id, best_iou = None, min_iou
    for o in outputs:
        score = iou(box, [float(v) for v in o[:4]])
        if score > best_iou:
            best_iou, best_id = score, int(o[4])
    return best_id


def analyze_target(box, name, conf, outputs, speed_estimator, cfg, ttc_threshold):
    """Distance, relative speed, TTC, and risk level for one vehicle.

    A vehicle is high risk when TTC = distance / speed is at or below `ttc_threshold`.
    """
    distance = estimate_distance(name, int(box[3]) - int(box[1]), cfg['distance'])
    track_id = match_track(box, outputs)
    speed = speed_estimator.speed_of(track_id)

    ttc, flag = None, 'No risk'
    if speed:
        ttc = distance / (speed / 3.6)   # km/h to m/s
        if ttc <= ttc_threshold:
            flag = 'High risk'

    return {'name': name, 'conf': conf, 'distance': distance, 'speed': speed,
            'track_id': track_id, 'ttc': ttc, 'flag': flag}
