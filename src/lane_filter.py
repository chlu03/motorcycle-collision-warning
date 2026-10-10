"""Keep only detections whose bottom-center point lies inside the rider's lane."""
import cv2


def bottom_center(box):
    """Bottom-center point of an (x1, y1, x2, y2, ...) box."""
    return (float((box[0] + box[2]) / 2), float(box[3]))


def filter_detections(detections, polygon):
    """Return the detections (rows of x1, y1, x2, y2, conf, cls) inside the lane polygon."""
    return [det for det in detections
            if cv2.pointPolygonTest(polygon, bottom_center(det), False) >= 0]
