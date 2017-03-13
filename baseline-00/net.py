from utility.common import *

# assume fix calibration
#from utility.calibration import * ...




def top_lidar_feature_net(input):
    decay=0.005

    #block
    block    = Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(input)
    block    = Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(block)
    feature  = Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(block)
    proposal = Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(block)

    return feature,proposal

def front_lidar_feature_net(input):
    decay=0.005

    #block
    block    = Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(input)
    block    = Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(block)
    feature  = Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(block)

    return feature



def image_feature_net(input):
    decay=0.005

    input = Input(shape=input_shape, dtype='float32', name='input')
    #block
    block    = Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(input)
    block    = Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(block)
    feature  = Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu', W_regularizer=l2(decay))(block)

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



# example to run a prediction ###########################################################################
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





