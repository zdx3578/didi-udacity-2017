from net.common import *
from net.utility.file import *


from dummynet import *
from net.rpn_target_op import make_bases, make_anchors, tf_rpn_target, draw_op_results
from net.processing.boxes import *
from net.utility.draw import *


#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017


# faster rcnn loss
def modified_smooth_l1( bbox_pred, bbox_target, bbox_weight, sigma=3.0):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    '''
    sigma2 = sigma * sigma
    diff  =  tf.subtract(bbox_pred, bbox_target)
    smooth_l1_sign = tf.cast(tf.less(tf.abs(diff), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diff, diff) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diff) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_sign) + tf.multiply(smooth_l1_option2, 1-smooth_l1_sign)
    smooth_l1 = tf.multiply(bbox_weight, smooth_l1_add)  #

    return smooth_l1



def run_train():

    # output dir, etc
    out_dir = '/root/share/out/didi/xxx'
    log = Logger(out_dir+'/log.txt',mode='a')

    #a dummy data -----------------
    if 0:  #dummy image
        lidar_top = cv2.imread('/root/share/data/dummy/cls2/panda/Images/train/001.jpg')
        gt_boxes = np.array([ #x1,y1,x2,y2,label
            [132,	189,	208,	289,	  1],#  panda
            [211,	263,	314,	390,	  1],#  panda
            [77,	273,	171,	399,	  1],#  panda
            [382,	279,	484,	402,	  1],#  panda
        ])
        img_height, img_width, img_channel = lidar_top.shape  #570,427
        num_gt_boxes = len(gt_boxes)


        ratios=[0.5, 1, 2]
        scales=2**np.arange(3, 6 )
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 8
        top_lidar_shape   = (427, 570, 3)
        top_feature_shape = (54, 72,  2*num_bases) #( 143, 107, 3)



    if 0:  #dummy image
        lidar_top = cv2.imread('/root/share/project/didi/data/kitti/dummy/one_frame/top_image_0.png')
        gt_boxes  = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes.npy')
        img_height, img_width, img_channel = lidar_top.shape
        num_gt_boxes = len(gt_boxes)

        ratios=np.array([0.5,1,2], dtype=np.float32)
        scales=np.array([1,2,3],   dtype=np.int32)
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 8
        top_lidar_shape   = (400, 400, 3)
        top_feature_shape = (50, 50,  2*num_bases)




    #one lidar data -----------------
    if 1:  #dummy image
        lidar_top = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/top_image.npy')
        gt_boxes  = np.load('/root/share/project/didi/data/kitti/dummy/one_frame/gt_boxes.npy')
        img_height, img_width, img_channel = lidar_top.shape
        num_gt_boxes = len(gt_boxes)

        ratios=np.array([0.5,1,2], dtype=np.float32)
        scales=np.array([1,2,3],   dtype=np.int32)
        bases = make_bases(
            base_size = 16,
            ratios=ratios,
            scales=scales
        )
        num_bases = len(bases)
        stride = 8
        top_lidar_shape   = (400, 400, 8)
        top_feature_shape = (50, 50,  2*num_bases)





    #load model
    top_image  = tf.placeholder(shape=[None, *top_lidar_shape], dtype=tf.float32, name='input')
    top_feature, top_score, top_box = top_lidar_feature_net(top_image, num_bases)

    #rpn ground truth generator
    anchors, inds_inside =  make_anchors(bases, stride, top_lidar_shape, top_feature_shape)
    ##inds_inside=np.arange(0,len(anchors))  #use all

    rpn_gt_boxes    = tf.placeholder(shape=[None, 5], dtype=tf.float32, name ='gt_boxes'   )  #gt_boxes
    rpn_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )  #anchors
    rpn_inds_inside = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inds_inside')  #inds_inside
    rpn_target = tf_rpn_target (rpn_gt_boxes, rpn_anchors, rpn_inds_inside)


    #-----------------------------------------------------------------
    rpn_label       =  tf.placeholder(shape=[None   ], dtype=tf.int32,   name='rpn_label')
    rpn_bbox_target =  tf.placeholder(shape=[None, 4], dtype=tf.float32, name='rpn_bbox_target')
    rpn_bbox_weight =  tf.placeholder(shape=[None, 4], dtype=tf.float32, name='rpn_bbox_weight')

    rpn_cls_score   = tf.reshape(top_score,[-1,2])
    rpn_cls_prob    = tf.nn.softmax(rpn_cls_score)
    rpn_cls_score1  = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2]) # remove ignore label
    rpn_label1      = tf.reshape(tf.gather(rpn_label,    tf.where(tf.not_equal(rpn_label,-1))),[-1])
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score1, labels=rpn_label1))

    rpn_bbox_pred   = tf.reshape(top_box,[-1,4])
    rpn_smooth_l1 = modified_smooth_l1(rpn_bbox_pred, rpn_bbox_target,rpn_bbox_weight, sigma=3.0)
    rpn_loss_box  = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))


    #put your solver here
    l2 = l2_regulariser(decay=0.0005)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver      = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    solver_step = solver.minimize(rpn_cross_entropy+rpn_loss_box+l2)  #rpn_loss_box


    max_iter = 10000
    batch_label  = np.zeros((100),dtype=np.int32)
    batch_target = np.zeros((1,100,4),dtype=np.float32)
    batch_bbox_weight= np.zeros((1,100,4),dtype=np.float32)




    # start training here ------------------------------------------------
    log.write('epoch        iter      rate     |  train_mse   valid_mse  |\n')
    log.write('----------------------------------------------------------------------------\n')


    num_ratios=len(ratios)
    num_scales=len(scales)
    fig, axs = plt.subplots(num_ratios,num_scales)

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        rate=0.1

        batch_img         = lidar_top.reshape(1,img_height, img_width, img_channel)
        batch_gt_box      = gt_boxes
        batch_anchors     = anchors
        batch_inds_inside = inds_inside

        ##-------------------------------


        batch_box_loss=0
        batch_cls_loss=0
        for iter in range(max_iter):
            fd={
                rpn_gt_boxes:    batch_gt_box,
                rpn_anchors:     batch_anchors,
                rpn_inds_inside: batch_inds_inside,
                IS_TRAIN_PHASE: True
            }
            batch_label, batch_target, batch_bbox_weight = sess.run(rpn_target, fd)


            #make an image for drawing
            img = np.sum(lidar_top,axis=2)
            img = img-np.min(img)
            img = (img/np.max(img)*255)
            img = np.dstack((img, img, img)).astype(np.uint8)

            #http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
            #http://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
            #img = np.stack((img,)*3)

            draw_op_results(batch_label,batch_target, batch_bbox_weight, anchors, img, batch_gt_box)
            cv2.waitKey(1)

            ##-------------------------------
            fd={
                top_image: batch_img,
                rpn_label: batch_label,
                rpn_bbox_target: batch_target,
                rpn_bbox_weight: batch_bbox_weight,
                learning_rate: rate,
                IS_TRAIN_PHASE: True
            }
            #_, rpn_cls_loss = sess.run([solver_step, rpn_cross_entropy],fd)
            _, batch_cls_loss, batch_box_loss = sess.run([solver_step, rpn_cross_entropy, rpn_loss_box],fd)

            #print('ok')
            # debug: ------------------------------------
            if 1:
                ww, aa  = sess.run([rpn_bbox_weight, top_box],fd)

            ##def draw_proposals(im, proposals, fraction=1.0):
            if iter%4==0:
                batch_prob, batch_score, batch_box = sess.run([ rpn_cls_prob, top_score, top_box  ],fd)
                H, W, _  = top_feature_shape

                p = batch_prob.reshape( H, W, 2*num_bases)
                for n in range(num_bases):
                    r=n%num_scales
                    s=n//num_scales
                    pn = p[:,:,2*n+1]*255
                    axs[s,r].cla()
                    axs[s,r].imshow(pn, cmap='gray', vmin=0, vmax=255)
                plt.pause(0.01)


                scores = batch_prob.reshape(-1,2)
                scores = scores[:,1]
                ##scores = batch_label.reshape(-1)

                boxes = batch_box.reshape(-1,4)
                inds = np.argsort(scores)[::-1]       #sort ascend #[::-1]
                inds = inds[0:100]
                #img  = img.copy()

                num_anchors = len(anchors)
                states=np.zeros((num_anchors),dtype=np.int32)
                states[inds_inside]=1
                for j in range(len(inds)):
                    i=inds[j]

                    if states[i]==0:
                        continue

                    a = anchors[i,0:4]
                    t = boxes[i,0:4]
                    b = bbox_transform_inv(a.reshape(1,4), t.reshape(1,4))
                    #b = clip_boxes(b,img_width,img_height)
                    b = b.reshape(-1)
                    s = scores[i]
                    v = s*255
                    if s<0.75:
                        continue


                    #if scores[i]<0.5: continue
                    cv2.rectangle(img,(a[0], a[1]), (a[2], a[3]), (0,v,v), 1)
                    cv2.rectangle(img,(b[0], b[1]), (b[2], b[3]), (0,0,v), 1)

                imshow('img_predict',img)
                cv2.waitKey(1)







            log.write('%d   %0.5f   %0.5f : \n'%(iter, batch_cls_loss, batch_box_loss))





if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_train()