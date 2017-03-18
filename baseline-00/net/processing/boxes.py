from net.configuration import CFG

from net.processing.cython_bbox import bbox_overlaps
from net.processing.cython_bbox import bbox_vote
#from net.processing.gpu_nms import gpu_nms as nms   ##unknown bug ???
from net.processing.cpu_nms import cpu_nms as nms

import numpy as np

#     roi  : i, x1,y1,x2,y2  i=image_index  
#     bbox : x1,y1,x2,y2,  
#     box  : x1,y1,x2,y2, label 
# or  box  : x1,y1,x2,y2, score 
#     det  : x1,y1,x2,y2, score 
#     annotation : box, ... other information, ...




# et_boxes = estimated 
# gt_boxes = ground truth 

def bbox_transform(et_bboxes, gt_bboxes):
    et_ws  = et_bboxes[:, 2] - et_bboxes[:, 0] + 1.0
    et_hs  = et_bboxes[:, 3] - et_bboxes[:, 1] + 1.0
    et_cxs = et_bboxes[:, 0] + 0.5 * et_ws
    et_cys = et_bboxes[:, 1] + 0.5 * et_hs
     
    gt_ws  = gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1.0
    gt_hs  = gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1.0
    gt_cxs = gt_bboxes[:, 0] + 0.5 * gt_ws
    gt_cys = gt_bboxes[:, 1] + 0.5 * gt_hs
     
    dxs = (gt_cxs - et_cxs) / et_ws
    dys = (gt_cys - et_cys) / et_hs
    dws = np.log(gt_ws / et_ws)
    dhs = np.log(gt_hs / et_hs)

    deltas = np.vstack((dxs, dys, dws, dhs)).transpose()
    return deltas



def bbox_transform_inv(bboxes, deltas):

    if bboxes.shape[0] == 0: 
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    bboxes = bboxes.astype(deltas.dtype, copy=False) 
    ws  = bboxes[:, 2] - bboxes[:, 0] + 1.0
    hs  = bboxes[:, 3] - bboxes[:, 1] + 1.0
    cxs = bboxes[:, 0] + 0.5 * ws
    cys = bboxes[:, 1] + 0.5 * hs

    dxs = deltas[:, 0::4]
    dys = deltas[:, 1::4]
    dws = deltas[:, 2::4]
    dhs = deltas[:, 3::4]

    pred_cxs = dxs * ws[:, np.newaxis] + cxs[:, np.newaxis]
    pred_cys = dys * hs[:, np.newaxis] + cys[:, np.newaxis]
    pred_ws = np.exp(dws) * ws[:, np.newaxis]
    pred_hs = np.exp(dhs) * hs[:, np.newaxis]

    pred_bboxes = np.zeros(deltas.shape, dtype=deltas.dtype) 
    pred_bboxes[:, 0::4] = pred_cxs - 0.5 * pred_ws  # x1, y1,x2,y2
    pred_bboxes[:, 1::4] = pred_cys - 0.5 * pred_hs 
    pred_bboxes[:, 2::4] = pred_cxs + 0.5 * pred_ws 
    pred_bboxes[:, 3::4] = pred_cys + 0.5 * pred_hs

    return pred_bboxes

# nms  ###################################################################
def non_max_suppress(bboxes, scores, num_classes, 
                     nms_after_thesh=CFG.TEST.RCNN_NMS_AFTER, 
                     nms_before_score_thesh=0.05, 
                     is_bbox_vote=False,
                     max_per_image=100 ):

   
    # nms_before_thesh = 0.05 ##0.05   # set low number to make roc curve.
                                       # else set high number for faster speed at inference
 
    #non-max suppression 
    nms_boxes = [[]for _ in xrange(num_classes)]
    for j in xrange(1, num_classes): #skip background
        inds = np.where(scores[:, j] > nms_before_score_thesh)[0]
         
        cls_scores = scores[inds, j]
        cls_boxes  = bboxes [inds, j*4:(j+1)*4]
        cls_dets   = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False) 

        # is_bbox_vote=0
        if len(inds)>0:
            keep = nms(cls_dets, nms_after_thesh) 
            dets_NMSed = cls_dets[keep, :] 
            if is_bbox_vote:
                cls_dets = bbox_vote(dets_NMSed, cls_dets)
            else:
                cls_dets = dets_NMSed 

        nms_boxes[j] = cls_dets
      

    ##Limit to MAX_PER_IMAGE detections over all classes
    if max_per_image > 0:
        image_scores = np.hstack([nms_boxes[j][:, -1] for j in xrange(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in xrange(1, num_classes):
                keep = np.where(nms_boxes[j][:, -1] >= image_thresh)[0]
                nms_boxes[j] = nms_boxes[j][keep, :]

    return nms_boxes  

def clip_boxes(bboxes, width, height):
    ''' Clip process to image boundaries. '''

    # x1 >= 0
    bboxes[:, 0::4] = np.maximum(np.minimum(bboxes[:, 0::4], width - 1), 0)
    # y1 >= 0
    bboxes[:, 1::4] = np.maximum(np.minimum(bboxes[:, 1::4], height - 1), 0)
    # x2 < width
    bboxes[:, 2::4] = np.maximum(np.minimum(bboxes[:, 2::4], width - 1), 0)
    # y2 < height
    bboxes[:, 3::4] = np.maximum(np.minimum(bboxes[:, 3::4], height - 1), 0)
    return bboxes