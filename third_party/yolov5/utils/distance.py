foc = 800        # focal length (cm)
real_hight_bicycle = 26     # bicycle height (inches)
real_hight_car = 63      # car height (inches)
real_hight_motorcycle = 51      # motorcycle height (inches)
real_hight_bus = 128      # bus height (inches)
real_hight_truck = 127.5   # truck height (inches)

# Monocular distance estimation
def detect_distance_car(h):
    dis_inch = (real_hight_car * foc) / h
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def detect_distance_bicycle(h):
    dis_inch = (real_hight_bicycle * foc) / h
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm / 100
    return dis_m

def detect_distance_motorcycle(h):
    dis_inch = (real_hight_motorcycle * foc) / h
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def detect_distance_bus(h):
    dis_inch = (real_hight_bus * foc) / h
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def detect_distance_truck(h):
    dis_inch = (real_hight_truck * foc) / h
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m








