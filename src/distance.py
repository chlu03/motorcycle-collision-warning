"""
Monocular distance estimation from the bounding box height.

distance = real_height * focal_length / box_height_in_pixels
"""


def estimate_distance(name, box_height, cfg):
    """
    Distance in meters to a vehicle of class `name` whose box is `box_height` pixels tall.
    Unknown classes return 1.0 m, matching the original implementation.
    """
    heights = cfg['vehicle_heights_inch']
    if name not in heights or box_height <= 0:
        return 1.00
    dis_inch = heights[name] * cfg['focal_length'] / box_height
    dis_cm = int(dis_inch * 2.54)
    return dis_cm / 100
