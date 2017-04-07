from net.common import *
from net.utility.file import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from dummynet import *
from data import *

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target
from net.rcnn_target_op import rcnn_target

from net.rpn_nms_op     import draw_rpn_nms
from net.rcnn_nms_op    import rcnn_nms
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels, draw_rpn
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels, draw_rcnn



#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017
path='/home/ubuntu/didi-udacity-2017/data'

def load_dummy_data():
    rgb   = np.load(path+'/one_frame/rgb.npy')
    lidar = np.load(path+'/one_frame/lidar.npy')
    top   = np.load(path+'/one_frame/top.npy')
    gt_labels    = np.load(path+'/one_frame/gt_labels.npy')
    gt_boxes3d   = np.load(path+'/one_frame/gt_boxes3d.npy')
    gt_top_boxes = np.load(path+'/one_frame/gt_top_boxes.npy')

    top_image = cv2.imread(path+'/one_frame/top_image.png')
    rgb =(rgb*255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    gt_boxes3d = gt_boxes3d.reshape(-1,8,3)


    return  rgb, top, top_image, lidar, gt_labels, gt_boxes3d,  gt_top_boxes



def load_dummy_data1():
    rgb = cv2.imread('/root/share/data/dummy/cls2/panda/Images/train/001.jpg')
    lidar = None
    top = cv2.imread('/root/share/data/dummy/cls2/panda/Images/train/001.jpg')
    gt_labels = np.array([
          1  ,#  panda
          1  ,#  panda
          1  ,#  panda
          1  ,#  panda
    ], dtype=np.int32)
    gt_top_boxes = np.array([ #x1,y1,x2,y2
        [132,	189,	208,	289],
        [211,	263,	314,	390],
        [77,	273,	171,	399],
        [382,	279,	484,	402],
    ], dtype=np.float32)

    gt_boxes3d = np.array([ #x1,y1,x2,y2
        [132,	189,	208,	289],
        [211,	263,	314,	390],
        [77,	273,	171,	399],
        [382,	279,	484,	402],
    ], dtype=np.float32)

    top_image = top
    return  rgb, top, top_image, lidar, gt_labels, gt_boxes3d,  gt_top_boxes



def run_train():

    # output dir, etc
    out_dir = '/tmp/root/share/out/didi/xxx'
    makedirs(out_dir +'/tf')
    log = Logger(out_dir+'/log.txt',mode='a')

    #one lidar data -----------------
    if 1:
        ratios=np.array([0.5,1,2], dtype=np.float32)
        scales=np.array([1,2,3],   dtype=np.float32)
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 8

        rgb, top, top_image, lidar, gt_labels, gt_boxes3d,  gt_top_boxes = load_dummy_data()
        top_shape = top.shape
        top_feature_shape = (top_shape[0]//stride, top_shape[1]//stride)

        rgb_shape = rgb.shape
        out_shape=(8,3)


        #-----------------------
        #check data
        if 0:
            fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            draw_lidar(lidar, fig=fig)
            draw_gt_boxes3d(gt_boxes3d, fig=fig)
            mlab.show(1)

            draw_gt_boxes(top_image, gt_top_boxes)
            draw_projected_gt_boxes3d(rgb, gt_boxes3d)

            #imshow('top_image',top_image)
            #imshow('rgb',rgb)
            cv2.waitKey(1)

    #one dummy data -----------------
    if 0:
        ratios=[0.5, 1, 2]
        scales=2**np.arange(3, 6 )
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 8

        rgb, top, top_image, lidar, gt_labels, gt_boxes3d,  gt_top_boxes = load_dummy_data1()
        top_shape = top.shape
        top_feature_shape = (54, 72)  #(top_shape[0]//stride, top_shape[1]//stride)

        rgb_shape = rgb.shape
        out_shape=(4,)

        # img_gt =draw_gt_boxes(top_image, gt_top_boxes)
        # imshow('img_gt',img_gt)
        # cv2.waitKey(1)

    # set anchor boxes
    dim = np.prod(out_shape)
    num_class = 2 #incude background
    anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
    inside_inds = np.arange(0,len(anchors),dtype=np.int32)  #use all
    print ('dim=%d'%dim)

    #load model ##############
    top_images      = tf.placeholder(shape=[None, *top_shape], dtype=tf.float32, name='top')
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

    top_features, top_scores, top_probs, top_deltas, top_rois1, top_roi_scores1 = \
        top_lidar_feature_net(top_images, top_anchors, top_inside_inds, num_bases)

    rgb_images  = tf.placeholder(shape=[None, *rgb_shape], dtype=tf.float32, name='rgb')
    rgb_features = rgb_feature_net(rgb_images)

    top_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois') #<todo> change to int32???
    rgb_rois = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois')
    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
            (top_features,   rgb_features,),
            (top_rois,       rgb_rois,),
            ([6,6,1./stride],[6,6,1./stride],),
            num_class, out_shape)


    #loss ####################
    top_inds     = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_ind'    )
    top_pos_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_pos_ind')
    top_labels   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_label'  )
    top_targets  = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target' )
    top_cls_loss, top_reg_loss   = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds,top_labels, top_targets)

    fuse_labels  = tf.placeholder(shape=[None            ], dtype=tf.int32,   name='fuse_label' )
    fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
    fuse_cls_loss, fuse_reg_loss = rcnn_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)


    #put your solver here
    l2 = l2_regulariser(decay=0.0005)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    #solver_step = solver.minimize(top_cls_loss+top_reg_loss+l2)
    solver_step = solver.minimize(top_cls_loss+top_reg_loss+fuse_cls_loss+fuse_reg_loss+l2)

    max_iter = 10000

    # start training here ------------------------------------------------
    log.write('epoch        iter      rate     |  train_mse   valid_mse  |\n')
    log.write('----------------------------------------------------------------------------\n')

    num_ratios=len(ratios)
    num_scales=len(scales)
    fig, axs = plt.subplots(num_ratios,num_scales)

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        rate=0.1

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0
        for iter in range(max_iter):

            #random sample train data
            batch_top_images    = top.reshape(1,*top_shape)
            batch_top_gt_labels = gt_labels
            batch_top_gt_boxes  = gt_top_boxes

            batch_rgb_images    = rgb.reshape(1,*rgb_shape)

            batch_fuse_gt_labels  = gt_labels
            batch_fuse_gt_boxes   = gt_top_boxes
            batch_fuse_gt_boxes3d = gt_boxes3d


            ##-------------------------------
            fd={
                top_images:  batch_top_images,
                top_anchors: anchors,
                top_inside_inds: inside_inds,
                learning_rate: rate,
                IS_TRAIN_PHASE: True
            }
            batch_top_rois1, batch_top_roi_scores1, batch_top_features = sess.run([top_rois1, top_roi_scores1, top_features],fd)

            ## generate ground truth
            batch_top_inds, batch_top_pos_inds, batch_top_labels, batch_top_targets  = \
                rpn_target ( anchors, inside_inds, batch_top_gt_labels,  batch_top_gt_boxes)

            batch_top_rois, batch_fuse_labels, batch_fuse_targets  = \
                 rcnn_target(  batch_top_rois1, batch_fuse_gt_labels, batch_fuse_gt_boxes, batch_fuse_gt_boxes3d )

            #project to rgb roi -------------------------------------------------
            batch_rgb_rois  = batch_top_rois.copy()
            num = len(batch_top_rois)
            for n in range(num):
                box3d = box_to_box3d(batch_top_rois[n,1:5].reshape(1,4)).reshape(8,3)
                qs = make_projected_box3d(box3d)

                minx = np.min(qs[:,0])
                maxx = np.max(qs[:,0])
                miny = np.min(qs[:,1])
                maxy = np.max(qs[:,1])
                batch_rgb_rois[n,1:5] = minx,miny,maxx,maxy


            darken=0.7
            img_rgb_roi = rgb.copy()*darken
            for n in range(num):
                b = batch_rgb_rois[n,1:5]
                cv2.rectangle(img_rgb_roi,(b[0],b[1]),(b[2],b[3]),(0,255,255),1)

            imshow('img_rgb_roi',img_rgb_roi)
            #--------------------------------------------------------------------

            ##debug
            if 1:
                img_gt = draw_rpn_gt(top_image, batch_top_gt_boxes, batch_top_gt_labels)
                img_label  = draw_rpn_labels (top_image, anchors, batch_top_inds, batch_top_labels )
                img_target = draw_rpn_targets(top_image, anchors, batch_top_pos_inds, batch_top_targets)
                imshow('img_rpn_gt',img_gt)
                imshow('img_rpn_label',img_label)
                imshow('img_rpn_target',img_target)

                img_label  = draw_rcnn_labels (top_image, batch_top_rois, batch_fuse_labels )
                img_target = draw_rcnn_targets(top_image, batch_top_rois, batch_fuse_labels, batch_fuse_targets)
                imshow('img_rcnn_label',img_label)
                imshow('img_rcnn_target',img_target)
                cv2.waitKey(1)

            #---------------------------------------------------
            fd={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds,

                top_inds:     batch_top_inds,
                top_pos_inds: batch_top_pos_inds,
                top_labels:   batch_top_labels,
                top_targets:  batch_top_targets,

                top_rois:   batch_top_rois,
                #front_rois1: batch_front_rois,

                rgb_images: batch_rgb_images,
                rgb_rois:   batch_rgb_rois,

                fuse_labels:  batch_fuse_labels,
                fuse_targets: batch_fuse_targets,

                learning_rate: rate,
                IS_TRAIN_PHASE: True
            }
            #_, batch_top_cls_loss, batch_top_reg_loss = sess.run([solver_step, top_cls_loss, top_reg_loss],fd)


            _, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss = \
               sess.run([solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],fd)


            #print('ok')
            # debug: ------------------------------------

            if iter%4==0:
                batch_top_probs, batch_top_scores, batch_top_deltas  = \
                    sess.run([ top_probs, top_scores, top_deltas ],fd)

                batch_fuse_probs, batch_fuse_deltas = \
                    sess.run([ fuse_probs, fuse_deltas ],fd)

                probs, boxes3d, priors, priors3d, deltas = rcnn_nms(batch_fuse_probs,  batch_fuse_deltas, batch_top_rois)


                ## show rpn score maps
                p = batch_top_probs.reshape( *(top_feature_shape[0:2]), 2*num_bases)
                for n in range(num_bases):
                    r=n%num_scales
                    s=n//num_scales
                    pn = p[:,:,2*n+1]*255
                    axs[s,r].cla()
                    axs[s,r].imshow(pn, cmap='gray', vmin=0, vmax=255)
                plt.pause(0.01)

                img_rpn = draw_rpn(top_image, batch_top_probs, batch_top_deltas, anchors, inside_inds)
                img_rpn_nms = draw_rpn_nms(top_image, batch_top_rois1, batch_top_roi_scores1) # estimat after non-max
                imshow('img_rpn',img_rpn)
                imshow('img_rpn_nms',img_rpn_nms)
                cv2.waitKey(1)

                #draw rcnn results --------------------------------
                img_rcnn = draw_rcnn (top_image, batch_fuse_probs, batch_fuse_deltas, batch_top_rois)
                draw_projected_gt_boxes3d(rgb, boxes3d, color=(255,255,255), thickness=1)

                imshow('img_rcnn',img_rcnn)
                cv2.waitKey(1)






            # debug: ------------------------------------


            log.write('%d   | %0.5f   %0.5f  %0.5f   %0.5f : \n'%(iter, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss))



## main function ##########################################################################

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_train()