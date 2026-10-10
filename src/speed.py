"""
Relative speed estimation from tracked box positions.

Every `interval_frames` frames with detections, the displacement of each tracked
vehicle between the first and last of those frames is converted from pixels to
meters using the vehicle's real width, then to km/h.
"""
import math

# COCO class IDs of the vehicles we track
VEHICLE_CLASS_IDS = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


def track_centers(outputs):
    """Convert DeepSORT outputs (x1, y1, x2, y2, track_id, class_id) of vehicles to
    [x_center, y_center, track_id, class_id, box_width]."""
    centers = []
    for x1, y1, x2, y2, track_id, class_id in outputs:
        if int(class_id) in VEHICLE_CLASS_IDS:
            centers.append([(x1 + x2) / 2, (y1 + y2) / 2, int(track_id), int(class_id), x2 - x1])
    return centers


def estimate_speeds(prev, curr, fps, cfg):
    """Speed (km/h) of every track present in both `prev` and `curr`.

    Returns a list of [speed_kmh, track_id].
    """
    widths = cfg['vehicle_widths_m']
    prev_by_id = {c[2]: c for c in prev}
    speeds = []
    for x, y, track_id, class_id, box_w in curr:
        if track_id not in prev_by_id or box_w <= 0:
            continue
        px, py = prev_by_id[track_id][:2]
        pixel_disp = math.sqrt((x - px) ** 2 + (y - py) ** 2)
        meters_per_pixel = widths[VEHICLE_CLASS_IDS[class_id]] / box_w
        # Original empirical formula: fps / 3, converted to km/h, then calibrated
        kmh = pixel_disp * meters_per_pixel * fps / 3 * 3.6 * cfg['calibration']
        speeds.append([round(kmh, 1), track_id])
    return speeds


class SpeedEstimator:
    """Collects track positions and updates speeds every `interval_frames` frames."""

    def __init__(self, fps, cfg):
        self.fps = fps
        self.cfg = cfg
        self.buffer = []
        self.speeds = []   # latest list of [speed_kmh, track_id]

    def update(self, centers):
        """Add one frame of track centers. Returns True when speeds were recomputed."""
        self.buffer.append(centers)
        if len(self.buffer) < self.cfg['interval_frames']:
            return False
        first, last = self.buffer[0], self.buffer[-1]
        self.buffer = []
        if first and last:
            self.speeds = estimate_speeds(first, last, self.fps, self.cfg)
            return True
        return False

    def speed_of(self, track_id):
        """Latest speed (km/h) of a track, or None if unknown."""
        if track_id is None:
            return None
        for kmh, tid in self.speeds:
            if tid == track_id:
                return float(kmh)
        return None
