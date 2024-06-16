def angles(error_np, angles_num):
    angles_lst = []
    interval = len(error_np) / (angles_num-1)
    for i in range(0,angles_num-1):
        
        angles_lst.append(int(interval*i))
    angles_lst.append(len(error_np)-1)
    return angles_lst

def split_xy_xyv(xy_xyv):
    splitted = xy_xyv.split(",")
    if len(splitted) == 3:
        return float(splitted[0]), float(splitted[1]), splitted[2]
    return float(splitted[0]), float(splitted[1])