from kitti_data import pykitti
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from kitti_data.draw import *
from kitti_data.io import *

from net.utility.draw import *


# run functions --------------------------------------------------------------------------





## objs to gt boxes ##
def obj_to_gt(objs):

    num         = len(objs)
    gt_boxes   = np.zeros((num,4),  dtype=np.float32)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels   = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label = 1 #<todo>

        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_coords(x0,y0)
        u1,v1=lidar_to_top_coords(x1,y1)
        u2,v2=lidar_to_top_coords(x2,y2)
        u3,v3=lidar_to_top_coords(x3,y3)

        umin=min(u0,u1,u2,u3)
        umax=max(u0,u1,u2,u3)
        vmin=min(v0,v1,v2,v3)
        vmax=max(v0,v1,v2,v3)

        gt_labels[n]=label
        gt_boxes[n]=np.array([umin,vmin,umax,vmax])
        gt_boxes3d[n]=b

    return gt_labels, gt_boxes, gt_boxes3d


## lidar to top ##
def lidar_to_top(lidar):

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)

    ## start to make top  here !!!
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:
                    yy,xx,zz = -(x-X0),-(y-Y0),z-Z0


                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top[yy,xx,zz]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
                    top[yy,xx,Zn]=max_intensity

                    #density
                    count = len(idx)
                    top[yy,xx,Zn+1]+=count

                pass
            pass
        pass
    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 0: #unprocess
        top_image = np.zeros((height,width,3),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y = qxs[n],qys[n]
            if x>=0 and x <width and y>0 and y<height:
                top_image[y,x,:] += 1

        max_value=np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image=top_image.astype(dtype=np.uint8)


    return top, top_image


## drawing ####

def draw_lidar(lidar, is_grid=False, is_top_region=True, fig=None):

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    if fig is None: fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs, prs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50,50,1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 =  50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50,50,1):
            x1,y1,z1 = x,-50, 0
            x2,y2,z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if 1:
        mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20.,0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = TOP_X_MIN
        x2 = TOP_X_MAX
        y1 = TOP_Y_MIN
        y2 = TOP_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)



    mlab.orientation_axes()
    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
    print(mlab.view())

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1), color=color, figure=fig)
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[ 12.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991


def draw_gt_boxes(image, gt_boxes):

    img=image.copy() 
    num = len(gt_boxes)
    for n in range(num):
       x1,y1,x2,y2 = gt_boxes[n]
       cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,255), 1)

    imshow('gt_boxes',img)



    imshow('gt_boxes3d',img)




# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    basedir = '/root/share/project/didi/data/kitti/dummy'
    date  = '2011_09_26'
    drive = '0005'

    # The range argument is optional - default is None, which loads the whole dataset
    dataset = pykitti.raw(basedir, date, drive) #, range(0, 50, 5))

    # Load some data
    dataset.load_calib()         # Calibration data are accessible as named tuples
    dataset.load_timestamps()    # Timestamps are parsed into datetime objects
    dataset.load_oxts()          # OXTS packets are loaded as named tuples
    #dataset.load_gray()         # Left/right images are accessible as named tuples
    #dataset.load_rgb()          # Left/right images are accessible as named tuples
    dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]

    tracklet_file = '/root/share/project/didi/data/kitti/dummy/2011_09_26/tracklet_labels.xml'

    num_frames=len(dataset.velo)  #154
    objects = read_objects(tracklet_file, num_frames)


    #----------------------------------------------------------
    lidar = dataset.velo[0]

    objs = objects[0]
    gt_labels, gt_boxes, gt_boxes3d = obj_to_gt(objs)

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(lidar, fig=fig)
    draw_gt_boxes3d(gt_boxes3d, fig=fig)
    mlab.show(1)

    print ('** calling lidar_to_tops() **')
    if 0:
        top, top_image = lidar_to_top(lidar)
        rgb = dataset.rgb[0][0]
    else:
        top = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/top.npy')
        top_image = cv2.imread('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.png')
        rgb = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/rgb.npy')

    rgb =(rgb*255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # -----------


    #check
    num = len(gt_boxes)
    for n in range(num):
       x1,y1,x2,y2 = gt_boxes[n]
       cv2.rectangle(top_image,(x1,y1), (x2,y2), (0,255,255), 1)


    ## check
    boxes3d0 = box_to_box3d(gt_boxes)

    draw_gt_boxes3d(boxes3d0,  color=(1,1,0), line_width=1, fig=fig)
    mlab.show(1)

    for n in range(num):
        qs = make_projected_box3d(gt_boxes3d[n])
        draw_projected_box3d(rgb,qs)

    imshow('rgb',rgb)
    cv2.waitKey(0)




    #save
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/rgb.npy',rgb)
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/lidar.npy',lidar)
    #np.save('/root/share/project/didi/data/kitti/dummy/one_frame/top.npy',top)
    #cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.png',top_image)
    #cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.maked.png',top_image)

    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_labels.npy',gt_labels)
    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes.npy',gt_boxes)
    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes3d.npy',gt_boxes3d)

    imshow('top_image',top_image)
    cv2.waitKey(0)

    pause











    exit(0)







    #-------------------------------------------------------------------
    if 1:
        lidar = dataset.velo[0]
        objs = objects[0]

        num_gt       = len(objs)
        gt_boxes    = np.zeros((num_gt,4),  dtype=np.float32)
        gt_boxes_3d = np.zeros((num_gt,8,3),dtype=np.float32)
        gt_labels    = np.zeros((num_gt),    dtype=np.int32)

        for n in range(num_gt):
            obj = objs[n]
            b   = obj.box
            label=1 #<todo>

            x0 = b[0,0]
            y0 = b[0,1]
            x1 = b[1,0]
            y1 = b[1,1]
            x2 = b[2,0]
            y2 = b[2,1]
            x3 = b[3,0]
            y3 = b[3,1]
            u0,v0=lidar_to_top_coords(x0,y0)
            u1,v1=lidar_to_top_coords(x1,y1)
            u2,v2=lidar_to_top_coords(x2,y2)
            u3,v3=lidar_to_top_coords(x3,y3)

            umin=min(u0,u1,u2,u3)
            umax=max(u0,u1,u2,u3)
            vmin=min(v0,v1,v2,v3)
            vmax=max(v0,v1,v2,v3)

            gt_boxes[n]=np.array([umin,vmin,umax,vmax])
            gt_boxes_3d[n]=b
            gt_labels[n]=label

            pass


        np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_top_boxes.npy',gt_boxes)
        np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes3d.npy',gt_boxes_3d)
        np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_labels.npy',gt_labels)
        exit(0)






    #-------------------------------------------------------------------

    # raw data
    if 0:
        lidar = dataset.velo[0]
        dataset.load_rgb()          # Left/right images are accessible as named tuples
        rgb = dataset.rgb[0][0]
        #rgb.shape
        #(375, 1242, 3)

        np.save('/root/share/project/didi/data/kitti/dummy/one_frame/rgb.npy',rgb)
        np.save('/root/share/project/didi/data/kitti/dummy/one_frame/lidar.npy',lidar)

    if 1:
        rgb = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/rgb.npy')
        lidar = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/lidar.npy')



    ## ---------------------------------------------
    image =(rgb*255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imshow('rgb',rgb)
    cv2.waitKey(1)


    K = dataset.calib.K_cam2
    M = dataset.calib.T_cam2_velo
    #M = np.linalg.inv(M)
    Kt = np.transpose(K)
    Mt = np.transpose(M)


    #make ground truth boxes
    num_gt_boxes = len(objs)
    gt_boxes=np.zeros((num_gt_boxes,5),dtype=np.int32)
    for n in range(num_gt_boxes):
        obj= objs[n]
        b = obj.box
        label=1  #<todo>

        Ps = np.hstack(( b, np.ones((8,1))) )
        Qs = np.matmul(Ps,Mt)

        Qs = Qs[:,0:3]
        qs = np.matmul(Qs,Kt)
        zs = qs[:,2].reshape(8,1)
        qs = (qs/zs).astype(np.int32)

        color=np.array(COLOR[obj.type])*255
        for k in range(0,4):

            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, 2, cv2.LINE_AA)

            i,j=k+4,(k+1)%4 + 4
            cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, 2, cv2.LINE_AA)

            i,j=k,k+4
            cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, 2, cv2.LINE_AA)



    imshow('rgb',image)
    cv2.waitKey(0)


    pass

    pose0 = dataset.calib.T_velo_imu
    pose  = dataset.oxts[0].T_w_imu

    #pose = np.matmul(pose,np.linalg.inv(pose0))
    #pose = np.matmul(np.linalg.inv(pose),pose0)
    #pose = np.matmul(pose0,np.linalg.inv(pose))


    ## show lidar 3d points and marking ###############################
    if 0:
        mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(500, 500))

        mlt = mlab.points3d(pxs, pys, pzs, prs,
             mode='point',  # 'point'  'sphere'
             colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
             scale_factor=1)

        #draw grid
        if 0:
            mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

            for y in np.arange(-50,50,1):
                x1,y1,z1 = -50, y, 0
                x2,y2,z2 =  50, y, 0
                mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1)

            for x in np.arange(-50,50,1):
                x1,y1,z1 = x,-50, 0
                x2,y2,z2 = x, 50, 0
                mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1)

        #draw axis
        if 1:
            mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

            axes=np.array([
                [2.,0.,0.,0.],
                [0.,2.,0.,0.],
                [0.,0.,2.,0.],
            ],dtype=np.float64)
            fov=np.array([
                [ 20.,20., 0.,0.],
                [20.,-20.,0.,0.],
            ],dtype=np.float64)

            #poseT=pose.transpose()   ???
            #axes = np.matmul(axes,poseT)
            #fov  = np.matmul(fov,poseT)

            mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None)
            mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None)
            mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None)
            mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1)
            mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1)

        #draw top_image feature area
        if 1:
            x1 = TOP_X_MIN
            x2 = TOP_X_MAX
            y1 = TOP_Y_MIN
            y2 = TOP_Y_MAX
            mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1)
            mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1)
            mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1)
            mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1)



        #draw objects
        num_objs = len(objs)
        for n in range(num_objs):
            obj= objs[n]
            b = obj.box
            color=COLOR[objs[n].type]

            mlab.text3d(b[0,0], b[0,1], b[0,2], '%d'%n, scale=(1, 1, 1))
            for k in range(0,4):

                #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i,j=k,(k+1)%4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=2) #representation='wireframe') #tube_radius=0.05, tube_sides=3)

                i,j=k+4,(k+1)%4 + 4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=2)

                i,j=k,k+4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=2)

        mlab.orientation_axes()
        mlab.show()

    ## make one top_image ###############################

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top_image   = np.zeros(shape=(height,width,channel), dtype=np.float32)


    #make ground truth top view boxes
    num_gt_boxes = len(objs)
    gt_boxes=np.zeros((num_gt_boxes,5),dtype=np.int32)
    for n in range(num_gt_boxes):
        obj= objs[n]
        b = obj.box
        label=1 #<todo>

        x0 = b[0,0]
        y0 = b[0,1]
        x1 = b[1,0]
        y1 = b[1,1]
        x2 = b[2,0]
        y2 = b[2,1]
        x3 = b[3,0]
        y3 = b[3,1]
        u0,v0=lidar_to_top_image_coords(x0,y0)
        u1,v1=lidar_to_top_image_coords(x1,y1)
        u2,v2=lidar_to_top_image_coords(x2,y2)
        u3,v3=lidar_to_top_image_coords(x3,y3)

        umin=min(u0,u1,u2,u3)
        umax=max(u0,u1,u2,u3)
        vmin=min(v0,v1,v2,v3)
        vmax=max(v0,v1,v2,v3)
        gt_boxes[n]=np.array([umin,vmin,umax,vmax,label])
        pass

    ## show 2d image and marking ###############################
    #draw boxes
    img = np.zeros((height,width,3),dtype=np.float32)
    num = len(lidar)

    for n in range(num):
        x,y = qxs[n],qys[n]
        if x>=0 and x <width and y>0 and y<height:
            img[y,x,:] += 1
    max_value=np.max(np.log(img+0.001))
    img = img/max_value *255
    img=img.astype(dtype=np.int32)
    #imshow('img',img)


    for n in range(num_gt_boxes):
       x1,y1,x2,y2,l = gt_boxes[n]
       cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,255), 1)

    imshow('img',img)
    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes.npy',gt_boxes)
    cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image_0.png',img)
    img = cv2.flip(img,0)
    cv2.imwrite('/root/share/project/didi/data/kitti/dummy/one_frame/top_image_0.flip.png',img)
    #imshow('img',img)
    cv2.waitKey(1)



    ## start to make top_image here !!!  ###############################
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:

                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top_image[y-Y0,x-X0,z-Z0]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
                    top_image[y-Y0,x-X0,Zn]=max_intensity

                    #density
                    count = len(idx)
                    top_image[y-Y0,x-X0,Zn+1]+=count

                pass
            pass
        pass
    top_image[:,:,Zn+1] = np.log(top_image[:,:,Zn+1]+1)/math.log(64)


    #save
    np.save('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.npy',top_image)



    fig, axs = plt.subplots(1,channel)
    for c in range(channel):
        d = top_image[:,:,c]

        axs[c].cla()
        axs[c].imshow(d, cmap='gray')
        axs[c].set_aspect('equal')
        #axs[c].set_ylim(axs[c].get_ylim()[::-1])
        if c!=0: axs[c].axis('off')

    plt.waitforbuttonpress(1)
    plt.show()


