# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Wenhe Jia of priv-lab
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from detectron_cascade.core.config import cfg
from detectron_cascade.datasets import json_dataset
import detectron_cascade.utils.blob as blob_utils
import detectron_cascade.utils.boxes as box_utils


class DecodeBBoxOp(object):
    def __init__(self, train, stage_num):
        self._train = train
        self._stage_num = stage_num

    def forward(self, inputs, outputs):
        # The inputs contains [bbox_pred, cls_prob, rois]
        # rois --> np.array((num,5)), (batch_idx, x1, y2, x2, y2)
        # print('++++++++++++++++++++++ Decode BBox of rcnn stage {} +++++++++++++++++++++++'.format(self._stage_num))
        cls_prob = inputs[0].data[...]
        bbox_pred = inputs[1].data[...]
        rois = inputs[2].data[...]
        if self._train:
            overlaps = inputs[3].data[...]        
            im_info = inputs[4].data
        else:
            im_info = inputs[3].data

        proposals_next = rois[:, 1:5]

        # Use delta with max cls_score as deltas adding to rois
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            delta = bbox_pred[:, 4:bbox_pred.shape[1]]
        else:
            cls_idx = cls_prob.argmax(axis=1)
            delta = np.zeros((bbox_pred.shape[0], 4), dtype=bbox_pred.dtype)
            for i in xrange(cls_idx.shape[0]):
                delta[i, :] = bbox_pred[i, cls_idx[i] * 4:cls_idx[i] * 4 + 4]

        # Add bbox deltas onto rois to generate new rois
        if self._stage_num == 1:
            bbox_reg_weights = cfg.CASCADERCNN.BBOX_REG_WEIGHTS_STAGE1
        elif self._stage_num == 2:
            bbox_reg_weights = cfg.CASCADERCNN.BBOX_REG_WEIGHTS_STAGE2

        new_rois = box_utils.bbox_transform(proposals_next, delta, bbox_reg_weights)
        batch_idxs = rois[:, 0].reshape(rois.shape[0], 1)
        new_rois = np.hstack((batch_idxs, new_rois))

        # remove invalid boxes
        output_rois = remove_invalid_boxes(new_rois)

        if self._train:
            # screen out high IOU boxes, to remove redundant gt boxes
            output_rois = remove_high_iou_boxes(output_rois, overlaps)
        else:
            output_rois = output_rois

        # clip tiled boxes into image
        output_rois = clip_tiled_bboxes(output_rois, im_info[0, :2])
        blob_utils.py_op_copy_blob(output_rois, outputs[0])

def remove_invalid_boxes(boxes):

    ws = (boxes[:, 3] - boxes[:, 1] + 1)
    hs = (boxes[:, 4] - boxes[:, 2] + 1)
    invalid_idx_w = np.where(ws < 0)[0]
    invalid_idx_h = np.where(hs < 0)[0]
    _invalid_idx = np.append(invalid_idx_w, invalid_idx_h)
    invalid_idx = np.unique(_invalid_idx)

    if invalid_idx.shape[0] != 0:
        print('RCNN stage {} --- Distrubute And Fpn Rpn Proposals Op: input rois contain {} invalid boxes, they are:'.format(
               self._stage_num, invalid_idx.shape[0])
        )
        print(boxes[invalid_idx, :])

    new_boxes = np.delete(boxes, invalid_idx, axis=0)
    return new_boxes

def remove_high_iou_boxes(boxes, overlaps):
    assert boxes.shape[0] == overlaps.shape[0], 'number rois do not match number overlaps'

    # print('gt iou thr: {}'.format(cfg.TRAIN.GT_IOU_THR))
    valid_idx = np.where(overlaps < cfg.TRAIN.GT_IOU_THR)[0]
    # print('high iou boxes: {}'.format(boxes.shape[0] - valid_idx.shape[0]))

    new_boxes = boxes[valid_idx, :]
    return new_boxes

def clip_tiled_bboxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 1, 'boxes.shape[1] is {:d}, but must be divisible by 5.'.format(boxes.shape[1])
    # x1 >= 0
    boxes[:, 1::5] = np.maximum(np.minimum(boxes[:, 1::5], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 2::5] = np.maximum(np.minimum(boxes[:, 2::5], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 3::5] = np.maximum(np.minimum(boxes[:, 3::5], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 4::5] = np.maximum(np.minimum(boxes[:, 4::5], im_shape[0] - 1), 0)
    return boxes
