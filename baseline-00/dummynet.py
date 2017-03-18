from net.blocks import *
from net.utility.file import *



# assume fix calibration
#from utility.calibration import * ...



# temporary net for debugging only. may not follow the paper exactly ....
def top_lidar_feature_net(input, num_bases):
    scale=1.

    #with tf.variable_scope('preprocess') as scope:
    #    input = input

    with tf.variable_scope('block-1') as scope:
        block = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        scale *=0.5

    with tf.variable_scope('block-2') as scope:
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        scale *=0.5


    with tf.variable_scope('block-3') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        block = maxpool(block, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        scale *=0.5

    with tf.variable_scope('block-4') as scope:
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        block = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')


    with tf.variable_scope('block-5') as scope:
        #up    = upsample2d(block, factor = 2, has_bias=True, trainable=True, name='1')
        #up    = block
        up    = conv2d_bn_relu(block, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        score = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='pro_score')
        box   = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='pro_box')

    #proposal = non_max_suppresion(score,box)
    #feature = upsample2d(block, factor = 4,  ...)

    #proposal=0
    feature=block

    print ('scale=%f, stride=%0.f'%(scale, 1/scale))
    return feature, score, box



def front_lidar_feature_net(input):
    feature  = 0
    return feature



def image_feature_net(input):
    feature  = 0
    return feature


def fusion_net(input):

    block    = Dense(...)(input)
    block    = Dense(...)(block)
    scores   = Dense(...)(block)  # classification
    boxes    = Dense(...)(block)  #3d box regression ( 8 corners of box )

    return (scores, boxes)


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


    top_lidar_shape = (570,427, 8)   #(704,800, 8)
    top_image  = tf.placeholder(shape=[None, *top_lidar_shape], dtype=tf.float32, name='input')
    top_feature,  score, box = top_lidar_feature_net(top_image,num_bases=9)

    #input   = tf.get_default_graph().get_tensor_by_name('input:0')
    #print(input)

    # draw graph to check connections
    with tf.Session()  as sess:
        tf.global_variables_initializer().run(feed_dict={IS_TRAIN_PHASE:True})
        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

        print_macs_to_file(log)
    print ('sucess!')