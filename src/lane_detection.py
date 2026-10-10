"""
Lane detection with Canny edges, a road-area mask, and the Hough transform.

Each frame produces a left and right lane line. The lines are saved to a CSV file
(one row per frame) so the vehicle detection stage can keep only vehicles in the
rider's lane.
"""
import csv

import cv2
import numpy as np
from tqdm import tqdm


# Grayscale conversion
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Gaussian blur
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# Canny edge detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


# Road area mask (HSV: gray road surface + yellow markings)
def get_lane_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_dict_HSV = {'yellow': [(25, 50, 70), (35, 255, 255)],
                      'gray': [(0, 0, 40), (180, 40, 230)]}
    mask_yellow = cv2.inRange(hsv, *color_dict_HSV['yellow'])
    mask_gray = cv2.inRange(hsv, *color_dict_HSV['gray'])
    mask_lane = cv2.bitwise_or(mask_gray, mask_yellow)
    mask_lane = cv2.morphologyEx(mask_lane, cv2.MORPH_OPEN, kernel=np.ones((5, 5), dtype=np.uint8), iterations=6)
    mask_lane = cv2.morphologyEx(mask_lane, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), dtype=np.uint8))
    contours, _ = cv2.findContours(mask_lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(mask_lane, [hull], 0, (255), -1)
    mask_lane = cv2.morphologyEx(mask_lane, cv2.MORPH_ERODE, kernel=np.ones((5, 5), dtype=np.uint8))
    return mask_lane


# Ratio of background pixels in the mask
def get_mask_ratio(mask):
    pixel_count = np.prod(mask.shape)
    zero_count = np.count_nonzero(mask == 0)
    return zero_count / pixel_count


# Region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        ignore_mask_color = (255,) * img.shape[2]
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)


# Compute slope
def get_slope(x1, y1, x2, y2):
    if x2 == x1:
        return float('inf')
    return (y2 - y1) / (x2 - x1)


# Filter lane line candidates by slope
def choose_lines(lines, min_slope_thr, max_slope_thr):
    abs_slopes = np.abs([get_slope(*line) for line in lines])
    chosen_indices = np.logical_and(min_slope_thr <= abs_slopes, abs_slopes <= max_slope_thr)
    return lines[chosen_indices]


# Average left or right lane line candidates, removing outliers
def get_average_line(lines, default_line):
    lines = np.array(lines)
    if not lines.any():
        return default_line, [default_line]

    n = 3
    normal_indices = []
    lines = np.concatenate([lines, [default_line] * n], axis=0) if np.any(default_line) else lines
    lines_mean = np.mean(lines, axis=0)
    lines_std = np.std(lines, axis=0)
    for m in range(1, 3):
        normal_indices = np.all(abs(lines - lines_mean) <= m * lines_std, axis=1)
        if np.any(normal_indices):
            break
    lines = lines[normal_indices]
    average_line = np.mean(lines, axis=0, dtype=np.int32) if np.any(lines) else default_line
    return average_line, lines


# Compute intersection point of two lines
def line_intersection(line1, line2):
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(line1[:2], line1[2:]), det(line2[:2], line2[2:]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [round(x), round(y)]


# Extend the averaged segments to full lane lines
def get_complete_lines(left_line_seg, right_line_seg, img_height, img_width):
    bottom_line = [0, img_height, img_width, img_height]
    left_line = [0, 0, 0, img_height]
    right_line = [img_width, 0, img_width, img_height]

    left_bottom_line = line_intersection(bottom_line, left_line_seg)
    if left_bottom_line[0] < 0:
        left_bottom_line = line_intersection(left_line, left_line_seg)

    right_bottom_line = line_intersection(bottom_line, right_line_seg)
    if right_bottom_line[0] >= img_width:
        right_bottom_line = line_intersection(right_line, right_line_seg)

    lanes_intersection = line_intersection(left_line_seg, right_line_seg)

    left_lane_line = left_bottom_line + lanes_intersection
    right_lane_line = lanes_intersection + right_bottom_line
    return left_lane_line, right_lane_line


# Overlay lanes on the original frame
def weighted_img(img, initial_img, alpha=0.8, beta=0.5, gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


class LaneDetector:
    """Detects the rider's lane frame by frame.

    The previous frame's lane lines are kept and blended into the next frame,
    which keeps the lanes stable when a frame has few line candidates.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_left_line = []
        self.last_right_line = []

    # Hough transform, slope filtering, and averaging into left/right lane lines
    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        if lines is None:
            lines = []
        else:
            lines = lines.reshape(-1, 4)
            lines = choose_lines(lines, self.cfg['min_slope'], self.cfg['max_slope'])

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        line_img, left_lane_line, right_lane_line = self.draw_lines(line_img, lines)
        return line_img, left_lane_line, right_lane_line

    # Split candidates into left/right by slope, average them, and draw the lanes
    def draw_lines(self, img, lines, thickness=3):
        img_height, img_width, _ = img.shape
        slopes = np.array(list(map(lambda x: get_slope(*x), lines)))

        left_lines = lines[slopes < 0] if np.any(slopes) else []
        left_average_line, _ = get_average_line(left_lines, self.last_left_line)
        self.last_left_line = left_average_line

        right_lines = lines[slopes > 0] if np.any(slopes) else []
        right_average_line, _ = get_average_line(right_lines, self.last_right_line)
        self.last_right_line = right_average_line

        left_lane_line, right_lane_line = get_complete_lines(left_average_line, right_average_line,
                                                             img_height, img_width)
        cv2.line(img, left_lane_line[:2], left_lane_line[2:], [70, 130, 180], thickness * 2)
        cv2.line(img, right_lane_line[:2], right_lane_line[2:], [70, 130, 180], thickness * 2)
        return img, left_lane_line, right_lane_line

    def process_image(self, img):
        """Detect lanes in one RGB frame.

        Returns the frame with lanes drawn and the lane corners as
        (left_top, left_bottom, right_top, right_bottom).
        """
        cfg = self.cfg
        gray_image = grayscale(img)
        gaus_blur = gaussian_blur(gray_image, cfg['blur_kernel'])
        edges = canny(gaus_blur, cfg['canny_low'], cfg['canny_high'])

        imshape = img.shape
        mask_lane = get_lane_mask(img.copy())
        mask_ratio = get_mask_ratio(mask_lane)
        vertices = np.array([[(0, imshape[0] * 5 / 6),
                              (imshape[1] / 2, imshape[0] * 3 / 7),
                              (imshape[1], imshape[0] * 5 / 6)]], dtype=np.int32)
        edge_lane = region_of_interest(edges, vertices)

        if mask_ratio < cfg['mask_ratio_threshold']:
            edge_lane = cv2.bitwise_and(edge_lane, edge_lane, mask=mask_lane)

        line_img, left_lane_line, right_lane_line = self.hough_lines(
            edge_lane, rho=1, theta=np.pi / 90, threshold=cfg['hough_threshold'],
            min_line_len=cfg['hough_min_line_len'], max_line_gap=imshape[0] / 3)
        result = weighted_img(line_img, img)

        corners = ((left_lane_line[2], left_lane_line[3]),    # left top
                   (left_lane_line[0], left_lane_line[1]),    # left bottom
                   (right_lane_line[0], right_lane_line[1]),  # right top
                   (right_lane_line[2], right_lane_line[3]))  # right bottom
        return result, corners


def detect_lanes(video_path, csv_path, cfg, video_out=None):
    """Run lane detection on every frame and save the lane corners to a CSV file.

    Frames are read with OpenCV, the same way the detection stage reads them,
    so row N of the CSV always matches frame N of the video.
    """
    detector = LaneDetector(cfg)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Frame', 'Left_Top', 'Left_Bottom', 'Right_Top', 'Right_Bottom'])
        frame_number = 0
        with tqdm(total=total, desc='Lane detection', unit='frame') as bar:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_number += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                result_rgb, corners = detector.process_image(frame_rgb)
                csv_writer.writerow([frame_number] + [f'({x}, {y})' for x, y in corners])

                if video_out is not None:
                    if writer is None:
                        h, w = frame_bgr.shape[:2]
                        writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    writer.write(cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
                bar.update(1)

    cap.release()
    if writer is not None:
        writer.release()
    return frame_number


def load_lanes(csv_path):
    """Read the lane CSV into a list of (left_top, left_bottom, right_top, right_bottom), one per frame."""
    def parse(text):
        return tuple(map(int, text.strip('()').split(',')))

    lanes = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            lanes.append((parse(row['Left_Top']), parse(row['Left_Bottom']),
                          parse(row['Right_Top']), parse(row['Right_Bottom'])))
    if not lanes:
        raise ValueError(f'No lane data in {csv_path}')
    return lanes


def lane_polygon(lane):
    """Lane area as a polygon: left top, left bottom, right bottom, right top."""
    left_top, left_bottom, right_top, right_bottom = lane
    return np.array([left_top, left_bottom, right_bottom, right_top], dtype=np.int32)
