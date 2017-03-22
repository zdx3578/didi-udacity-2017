from net.common import *


##extension for 3d
def top_to_lidar_coords(xx,yy):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    y = Xn*TOP_Y_DIVISION-(xx+0.5)*TOP_Y_DIVISION + TOP_Y_MIN
    x = Yn*TOP_X_DIVISION-(yy+0.5)*TOP_X_DIVISION + TOP_X_MIN

    return x,y


def lidar_to_top_coords(x,y,z=None):
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1

    xx = Yn-int((y-TOP_Y_MIN)//TOP_Y_DIVISION)
    yy = Xn-int((x-TOP_X_MIN)//TOP_X_DIVISION)

    return xx,yy


def box_to_box3d(boxes):

    num=len(boxes)
    boxes3d = np.zeros((num,8,3),dtype=np.float32)
    for n in range(num):
        x1,y1,x2,y2 = boxes[n]

        points = [ (x1,y1), (x1,y2), (x2,y2), (x2,y1) ]
        for k in range(4):
            xx,yy = points[k]
            x,y  = top_to_lidar_coords(xx,yy)
            boxes3d[n,k,  :] = x,y,0.4
            boxes3d[n,4+k,:] = x,y,-2

    return boxes3d



def box3d_transform(et_boxes3d, gt_boxes3d):

    ##<todo> refine this normalisation later ... e.g. use log(scale)
    et_centers =   np.sum(et_boxes3d,axis=1, keepdims=True)/8
    et_scales  =   np.sum((et_boxes3d-et_centers)**2, axis=2, keepdims=True)**0.5
    deltas = (gt_boxes3d-et_centers)/et_scales

    return deltas



def box3d_transform_inv(et_boxes3d, deltas):

    et_centers =  np.sum(et_boxes3d,axis=1, keepdims=True)/8
    et_scales  =  np.sum((et_boxes3d-et_centers)**2, axis=2, keepdims=True)**0.5
    boxes3d = deltas*et_scales+et_centers

    return boxes3d


def regularise_box3d(box3d):

    b=box3d
    dis=0
    corners = np.zeros((4,3),dtype=np.float32)
    for k in range(0,4):
        #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i,j=k,k+4
        dis +=np.sum((b[i]-b[j])**2) **0.5
        corners[k] = (b[i]+b[j])/2
    dis = dis/4

    b = np.zeros((8,3),dtype=np.float32)
    for k in range(0,4):
        i,j=k,k+4
        b[i]=corners[k]-dis/2*np.array([0,0,1])
        b[j]=corners[k]+dis/2*np.array([0,0,1])

    return b