# On-screen overlay: label above each vehicle and a fixed info panel in the top-left corner
import cv2

SAFE_COLOR = (0, 180, 0)      # BGR green
RISK_COLOR = (0, 0, 255)      # BGR red
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_target_box(img, xyxy, info, line_thickness=3):
    """Draw the box and a label above it: class, distance, speed."""
    color = RISK_COLOR if info['flag'] == 'High risk' else SAFE_COLOR
    c1 = (int(xyxy[0]), int(xyxy[1]))
    c2 = (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    label = f"{info['name']} {info['distance']:.1f}m"
    if info['speed'] is not None:
        label += f" {info['speed']:.1f}km/h"

    scale = line_thickness / 3
    thickness = max(line_thickness - 1, 1)
    (tw, th), baseline = cv2.getTextSize(label, FONT, scale, thickness)
    # If the box is near the top edge, put the label inside the box
    top = c1[1] - th - baseline - 4 if c1[1] - th - baseline - 4 >= 0 else c1[1]
    cv2.rectangle(img, (c1[0], top), (c1[0] + tw + 6, top + th + baseline + 4), color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (c1[0] + 3, top + th + 2), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def draw_info_panel(img, targets, time_person):
    """Draw a semi-transparent panel showing the most critical in-lane vehicle."""
    s = img.shape[0] / 1080                      # scale with video resolution
    x0, y0 = int(20 * s), int(20 * s)
    w, h = int(470 * s), int(270 * s)
    pad, line_h = int(20 * s), int(40 * s)
    scale, thick = 0.85 * s, max(int(2 * s), 1)

    # Prefer a high-risk vehicle; otherwise show the nearest one
    target = None
    if targets:
        risky = [t for t in targets if t['flag'] == 'High risk']
        target = min(risky or targets, key=lambda t: t['distance'])
    warning = target is not None and target['flag'] == 'High risk'

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    if warning:
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), RISK_COLOR, max(int(4 * s), 2))

    threshold = time_person * 0.5
    if target is None:
        rows = [('Target', 'none in lane'), ('Distance', '--'), ('Rel. speed', '--'), ('TTC', '--')]
    else:
        rows = [
            ('Target', target['name']),
            ('Distance', f"{target['distance']:.1f} m"),
            ('Rel. speed', f"{target['speed']:.1f} km/h" if target['speed'] is not None else '--'),
            ('TTC', f"{target['ttc']:.1f} s (< {threshold:.1f} s)" if target['ttc'] is not None else '--'),
        ]

    y = y0 + pad + int(22 * s)
    for key, value in rows:
        cv2.putText(img, key, (x0 + pad, y), FONT, scale, (200, 200, 200), thick, cv2.LINE_AA)
        cv2.putText(img, value, (x0 + pad + int(190 * s), y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h

    status, color = ('WARNING', RISK_COLOR) if warning else ('SAFE', SAFE_COLOR)
    cv2.putText(img, 'Status', (x0 + pad, y + int(8 * s)), FONT, scale, (200, 200, 200), thick, cv2.LINE_AA)
    cv2.putText(img, status, (x0 + pad + int(190 * s), y + int(8 * s)), FONT, 1.2 * s, color, max(int(3 * s), 2), cv2.LINE_AA)
    return img
