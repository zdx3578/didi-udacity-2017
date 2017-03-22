from net.utility.file import *
from net.blocks import *
from net.rpn_nms_op import tf_rpn_nms
from net.roipooling_op import roi_pool as tf_roipooling



# temporary net for debugging only. may not follow the paper exactly ....
def top_lidar_feature_net(input, anchors, inds_inside, num_bases):


    stride=1.
    #with tf.variable_scope('top-preprocess') as scope:
    #    input = input

    with tf.variable_scope('top-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')


    with tf.variable_scope('top') as scope:
        #up     = upsample2d(block, factor = 2, has_bias=True, trainable=True, name='1')
        #up     = block
        up      = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        scores  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
        probs   = tf.nn.softmax( tf.reshape(scores,[-1,2]), name='prob')
        deltas  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')

    #<todo> flip to train and test mode nms (e.g. different nms_pre_topn values): use tf.cond
    with tf.variable_scope('top-nms') as scope:    #non-max
        batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
        img_scale = 1
        rois, roi_scores = tf_rpn_nms( probs, deltas, anchors, inds_inside,
                                       stride, img_width, img_height, img_scale,
                                       nms_thresh=0.7, min_size=stride, nms_pre_topn=500, nms_post_topn=100,
                                       name ='nms')

    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block

    print ('top lidar: scale=%f, stride=%d'%(1./stride, stride))
    return feature, scores, probs, deltas, rois, roi_scores



#------------------------------------------------------------------------------
def rgb_feature_net(input):

    stride=1.
    #with tf.variable_scope('rgb-preprocess') as scope:
    #   input = input-128

    with tf.variable_scope('rgb-block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('rgb-block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')


    #<todo> feature = upsample2d(block, factor = 4,  ...)
    feature = block


    print ('rgb : scale=%f, stride=%d'%(1./stride, stride))
    return feature




def fusion_net(features, rois, pools, num_class, out_shape=(8,3)):

    num=len(features)

    input = None
    with tf.variable_scope('fuse-input') as scope:
        for n in range(num):
            feature = features[n]
            roi     = rois[n]
            pool_height = pools[n][0]
            pool_width  = pools[n][1]
            pool_scale  = pools[n][2]

            roi_features,  roi_idxs = tf_roipooling(feature,roi, pool_height, pool_width, pool_scale, name='%d/pool'%n)
            roi_features = flatten(roi_features)
            if input is None:
                input = roi_features
            else:
                input = concat([input,roi_features], axis=1, name='%d/cat'%n)

    with tf.variable_scope('fuse-block-1') as scope:
        block = linear_bn_relu(input, num_hiddens=256, name='1')
        block = linear_bn_relu(block, num_hiddens=256, name='2')

    #include background class
    with tf.variable_scope('fuse') as scope:
        dim = np.product([*out_shape])
        scores  = linear(block, num_hiddens=num_class,     name='score')
        probs   = tf.nn.softmax (scores, name='prob')
        deltas  = linear(block, num_hiddens=dim*num_class, name='box')
        deltas  = tf.reshape(deltas,(-1,num_class,*out_shape))

    return  scores, probs, deltas























def front_lidar_feature_net(input):
    feature  = 0
    return feature



def image_feature_net(input):
    feature  = 0
    return feature



def all_net(top_lidar_shape, front_lidar_shape, rgb_shape):

    top_image   = Input(shape=top_lidar_shape,   dtype='float32', name='top_image')
    front_image = Input(shape=front_lidar_shape, dtype='float32', name='front_image')
    rgb_image   = Input(shape=rgb_shape,         dtype='float32', name='rgb_image')

    top_feature, proposal = top_lidar_feature_net(top_image)
    front_feature = front_lidar_feature_net(front_image)
    rgb_feature = front_lidar_feature_net(rgb_image)

    top_proposal, front_propsal, rbg_proposal = project_proposal(proposal)

    ##https://github.com/endernewton/tf-faster-rcnn
    top   = tf.image.roi_pooling(top_feature, top_proposal)
    front = tf.image.roi_pooling(front_feature, front_propsal)
    rgb   = tf.image.roi_pooling(rgb_feature, rgb_proposal)

    all = Concat(top,front,rgb)  ## merge layer in keras
    scores, boxes = fusion_net(all)

    model = Model(input=[top_image,front_image,rgb_image], output=[scores, boxes])
    return model



# example to run a prediction
def run_demo():

    lidar     = 0 #load from file
    rgb_image = 0 #load from file

    model = 0 #load from file

    # timining starts here! -------------------------------------

    #pre processing
    top_image   = point3d_to_top_image(lidar)
    front_image = point3d_to_front_image(lidar)

    boxes = model.predict( (top_image,front_image, rgb_image), batch_size=1 )

    #post processing
    boxes = do_non_max_suppression(boxes)

    time=0
    # timining stops here! -------------------------------------

    #draw results
    show_results(boxes, lidar, rgb_image, time)



# main ###########################################################################
# to start in tensorboard:
#    /opt/anaconda3/bin
#    ./python tensorboard --logdir /root/share/out/didi/tf
#     http://http://localhost:6006/

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    out_dir='/root/share/out/didi/tf'
    log = Logger('/root/share/out/udacity/00/xxx_log.txt', mode='a')  # log file

    num_bases = 9
    num_classes = 1

    top_lidar_shape = (400,400, 8)
    top_images      = tf.placeholder(shape=[None, *top_lidar_shape], dtype=tf.float32, name='input')
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )  #anchors
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')  #inside_inds

    top_features, top_scores, top_probs, top_boxes, top_rois, top_roi_scores = \
        top_lidar_feature_net(top_images, top_anchors, top_inside_inds, num_bases)


    fuse_scores, fuse_boxes  = \
        fusion_net([(top_features, top_rois),], num_classes, pool_height=6, pool_width=6, pool_scale=1./8)


    # draw graph to check connections
    with tf.Session()  as sess:
        tf.global_variables_initializer().run(feed_dict={IS_TRAIN_PHASE:True})
        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

        print_macs_to_file(log)
    print ('sucess!')