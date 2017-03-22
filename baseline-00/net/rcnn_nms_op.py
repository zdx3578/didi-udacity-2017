from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *


## temporay post-processing ....
## <todo> to be updated


def rcnn_nms(probs,  deltas,  rois ):

    threshold=0.75
    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>0.8)[0]

    #post processing
    priors = rois  [idx,1:5]
    deltas = deltas[idx,cls]
    probs  = probs [idx]

    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(priors,deltas)
        return probs,boxes,priors,deltas


    if deltas.shape[1:]==(8,3):
        priors3d = box_to_box3d(priors)
        boxes3d  = box3d_transform_inv(priors3d, deltas)

        num = len(boxes3d)
        for n in range(num):
            boxes3d[n] = regularise_box3d(boxes3d[n])

        return probs, boxes3d, priors, priors3d, deltas