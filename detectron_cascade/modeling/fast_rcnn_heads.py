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
##############################################################################

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron_cascade.core.config import cfg
from detectron_cascade.utils.c2 import const_fill
from detectron_cascade.utils.c2 import gauss_fill
from detectron_cascade.utils.net import get_group_gn
import detectron_cascade.utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_cascade_fast_rcnn_outputs(model, blobs_in, dim, stage_num):
    """Add RoI classification and bounding box regression output ops."""
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    if stage_num == 1:
        model.FC(
            blobs_in[0],
            'cls_score_1st',
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            model.Softmax('cls_score_1st', 'cls_prob_1st', engine='CUDNN')
        model.FC(
            blobs_in[0],
            'bbox_pred_1st',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )

    elif stage_num == 2:
        model.FC(
            blobs_in[0],
            'cls_score_2nd',
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            assert len(blobs_in) == 2, 'during inference, need fc2_2nd and fc2_1st_2nd as in put blobsin rcnn stage 2'
            model.Softmax('cls_score_2nd', 'cls_prob_2nd_2nd', engine='CUDNN')
            cls_prob_2nd_2nd = model.Softmax('cls_score_2nd', 'cls_prob_2nd_2nd', engine='CUDNN')
            model.FCShared(
                blobs_in[1],
                'cls_score_1st_2nd',
                dim,
                model.num_classes,
                weight='cls_score_1st_w',
                bias='cls_score_1st_b'
            )
            cls_prob_1st_2nd = model.Softmax('cls_score_1st_2nd', 'cls_prob_1st_2nd', engine='CUDNN')
            model.Sum([cls_prob_2nd_2nd, cls_prob_1st_2nd], 'cls_prob_2nd')
            model.Scale('cls_prob_2nd', 'cls_prob_2nd', scale=0.5)
        model.FC(
            blobs_in[0],
            'bbox_pred_2nd',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )

    elif stage_num == 3:
        model.FC(
            blobs_in[0],
            'cls_score_3rd',
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        if not model.train:  # == if test
            # Only add softmax when testing; during training the softmax is combined
            # with the label cross entropy loss for numerical stability
            assert len(blobs_in) == 3, 'during inference, need fc2_2nd and fc2_1st_2nd as in put blobsin rcnn stage 3'
            model.Softmax('cls_score_3rd', 'cls_prob_3rd_3rd', engine='CUDNN')
            cls_prob_3rd_3rd = model.Softmax('cls_score_3rd', 'cls_prob_3rd_3rd', engine='CUDNN')

            model.FCShared(
                blobs_in[1],
                'cls_score_1st_3rd',
                dim,
                model.num_classes,
                weight='cls_score_1st_w',
                bias='cls_score_1st_b'
            )
            cls_prob_1st_3rd = model.Softmax('cls_score_1st_3rd', 'cls_prob_1st_3rd', engine='CUDNN')

            model.FCShared(
                blobs_in[2],
                'cls_score_2nd_3rd',
                dim,
                model.num_classes,
                weight='cls_score_2nd_w',
                bias='cls_score_2nd_b'
            )
            cls_prob_2nd_3rd = model.Softmax('cls_score_2nd_3rd', 'cls_prob_2nd_3rd', engine='CUDNN')

            model.Sum([cls_prob_1st_3rd, cls_prob_2nd_3rd, cls_prob_3rd_3rd], 'cls_prob_3rd')
            model.Scale('cls_prob_3rd', 'cls_prob_3rd', scale=0.33333333)
        model.FC(
            blobs_in[0],
            'bbox_pred_3rd',
            dim,
            num_bbox_reg_classes * 4,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0)
        )

def add_cascade_fast_rcnn_losses(model, stage_num):
    """Add losses for RoI classification and bounding box regression."""
    if stage_num == 1:
        cls_prob, loss_cls = model.net.SoftmaxWithLoss(
            ['cls_score_1st', 'labels_int32_1st'], ['cls_prob_1st', 'loss_cls_1st'],
            scale=model.GetLossScale() * cfg.CASCADERCNN.WEIGHT_LOSS_BBOX_STAGE1
        )
        loss_bbox = model.net.SmoothL1Loss(
            [
                'bbox_pred_1st', 'bbox_targets_1st', 'bbox_inside_weights_1st',
                'bbox_outside_weights_1st'
            ],
            'loss_bbox_1st',
            scale=model.GetLossScale() * cfg.CASCADERCNN.WEIGHT_LOSS_BBOX_STAGE1
        )
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
        model.Accuracy(['cls_prob_1st', 'labels_int32_1st'], 'accuracy_cls_1st')
        model.AddLosses(['loss_cls_1st', 'loss_bbox_1st'])
        model.AddMetrics('accuracy_cls_1st')

    elif stage_num == 2:
        cls_prob, loss_cls = model.net.SoftmaxWithLoss(
            ['cls_score_2nd', 'labels_int32_2nd'], ['cls_prob_2nd', 'loss_cls_2nd'],
            scale=model.GetLossScale() * cfg.CASCADERCNN.WEIGHT_LOSS_BBOX_STAGE2
        )
        loss_bbox = model.net.SmoothL1Loss(
            [
                'bbox_pred_2nd', 'bbox_targets_2nd', 'bbox_inside_weights_2nd',
                'bbox_outside_weights_2nd'
            ],
            'loss_bbox_2nd',
            scale=model.GetLossScale() * cfg.CASCADERCNN.WEIGHT_LOSS_BBOX_STAGE2
        )
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
        model.Accuracy(['cls_prob_2nd', 'labels_int32_2nd'], 'accuracy_cls_2nd')
        model.AddLosses(['loss_cls_2nd', 'loss_bbox_2nd'])
        model.AddMetrics('accuracy_cls_2nd')

    elif stage_num == 3:
        cls_prob, loss_cls = model.net.SoftmaxWithLoss(
            ['cls_score_3rd', 'labels_int32_3rd'], ['cls_prob_3rd', 'loss_cls_3rd'],
            scale=model.GetLossScale() * cfg.CASCADERCNN.WEIGHT_LOSS_BBOX_STAGE3
        )
        loss_bbox = model.net.SmoothL1Loss(
            [
                'bbox_pred_3rd', 'bbox_targets_3rd', 'bbox_inside_weights_3rd',
                'bbox_outside_weights_3rd'
            ],
            'loss_bbox_3rd',
            scale=model.GetLossScale() * cfg.CASCADERCNN.WEIGHT_LOSS_BBOX_STAGE3
        )
        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
        model.Accuracy(['cls_prob_3rd', 'labels_int32_3rd'], 'accuracy_cls_3rd')
        model.AddLosses(['loss_cls_3rd', 'loss_bbox_3rd'])
        model.AddMetrics('accuracy_cls_3rd')

    return loss_gradients

# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_cascade_2mlp_head(model, blob_in, dim_in, spatial_scale, stage_num):
    """Add cascade ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if stage_num == 1:
        roi_feat = model.RoIFeatureTransform(
            blob_in,
            'roi_feat_1st',
            blob_rois='rois_1st',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale
        )
        model.FC(roi_feat, 'fc1' + '_1st', dim_in * roi_size * roi_size, hidden_dim, weight_init=("MSRAFill", {}), bias_init=const_fill(0.0))
        model.Relu('fc1' + '_1st', 'fc1' + '_1st')
        model.FC('fc1' + '_1st', 'fc2' + '_1st', hidden_dim, hidden_dim, weight_init=("MSRAFill", {}), bias_init=const_fill(0.0))
        model.Relu('fc2' + '_1st', 'fc2' + '_1st')
        return ['fc2' + '_1st'], hidden_dim

    elif stage_num == 2:
        roi_feat = model.RoIFeatureTransform(
            blob_in,
            'roi_feat_2nd',
            blob_rois='rois_2nd',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale
        )
        model.FC(roi_feat, 'fc1' + '_2nd', dim_in * roi_size * roi_size, hidden_dim, weight_init=("MSRAFill", {}), bias_init=const_fill(0.0))
        model.Relu('fc1' + '_2nd', 'fc1' + '_2nd')
        model.FC('fc1' + '_2nd', 'fc2' + '_2nd', hidden_dim, hidden_dim, weight_init=("MSRAFill", {}), bias_init=const_fill(0.0))
        model.Relu('fc2' + '_2nd', 'fc2' + '_2nd')

        if not model.train:
            model.FCShared(roi_feat, 'fc1' + '_1st' + '_2nd', dim_in * roi_size * roi_size, hidden_dim, weight='fc1_1st_w', bias='fc1_1st_b')
            model.Relu('fc1' + '_1st' + '_2nd', 'fc1' + '_1st' + '_2nd')
            model.FCShared('fc1' + '_1st' + '_2nd', 'fc2' + '_1st' + '_2nd', hidden_dim, hidden_dim, weight='fc2_1st_w', bias='fc2_1st_b')
            model.Relu('fc2' + '_1st' + '_2nd', 'fc2' + '_1st' + '_2nd')
        return ['fc2' + '_2nd', 'fc2' + '_1st' + '_2nd'], hidden_dim

    elif stage_num == 3:
        roi_feat = model.RoIFeatureTransform(
            blob_in,
            'roi_feat_3rd',
            blob_rois='rois_3rd',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=roi_size,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale
        )
        model.FC(roi_feat, 'fc1' + '_3rd', dim_in * roi_size * roi_size, hidden_dim, weight_init=("MSRAFill", {}), bias_init=const_fill(0.0))
        model.Relu('fc1' + '_3rd', 'fc1' + '_3rd')
        model.FC('fc1' + '_3rd', 'fc2' + '_3rd', hidden_dim, hidden_dim, weight_init=("MSRAFill", {}), bias_init=const_fill(0.0))
        model.Relu('fc2' + '_3rd', 'fc2' + '_3rd')

        if not model.train:
            model.FCShared(roi_feat, 'fc1' + '_1st' + '_3rd', dim_in * roi_size * roi_size, hidden_dim, weight='fc1_1st_w', bias='fc1_1st_b')
            model.Relu('fc1' + '_1st' + '_3rd', 'fc1' + '_1st' + '_3rd')
            model.FCShared('fc1' + '_1st' + '_3rd', 'fc2' + '_1st' + '_3rd', hidden_dim, hidden_dim, weight='fc2_1st_w', bias='fc2_1st_b')
            model.Relu('fc2' + '_1st' + '_3rd', 'fc2' + '_1st' + '_3rd')

            model.FCShared(roi_feat, 'fc1' + '_2nd' + '_3rd', dim_in * roi_size * roi_size, hidden_dim, weight='fc1_2nd_w', bias='fc1_2nd_b')
            model.Relu('fc1' + '_2nd' + '_3rd', 'fc1' + '_2nd' + '_3rd')
            model.FCShared('fc1' + '_2nd' + '_3rd', 'fc2' + '_2nd' + '_3rd', hidden_dim, hidden_dim, weight='fc2_2nd_w', bias='fc2_2nd_b')
            model.Relu('fc2' + '_2nd' + '_3rd', 'fc2' + '_2nd' + '_3rd')
        return ['fc2' + '_3rd', 'fc2' + '_1st' + '_2nd', 'fc2' + '_2nd' + '_3rd'], hidden_dim

def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim

def add_roi_cascade_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale, stage_num):
    """Add cascade X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:
        if stage_num == 1:
            roi_feat = model.RoIFeatureTransform(
                blob_in, 'roi_feat_1st',
                blob_rois='rois_1st',
                method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
                resolution=roi_size,
                sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
                spatial_scale=spatial_scale
            )

            current = roi_feat
            for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
                current = model.ConvGN(
                    current, 'head_conv' + str(i + 1) + '_1st', dim_in, hidden_dim, 3,
                    group_gn=get_group_gn(hidden_dim),
                    stride=1, pad=1,
                    weight_init=('MSRAFill', {}),
                    bias_init=('ConstantFill', {'value': 0.}))
                current = model.Relu(current, current)
                dim_in = hidden_dim

            fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
            model.FC(current, 'fc6_1st', dim_in * roi_size * roi_size, fc_dim)
            model.Relu('fc6_1st', 'fc6_1st')
            return ['fc6_1st'], fc_dim

        elif stage_num == 2:
            roi_feat = model.RoIFeatureTransform(
                blob_in, 'roi_feat_2nd',
                blob_rois='rois_2nd',
                method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
                resolution=roi_size,
                sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
                spatial_scale=spatial_scale
            )

            current = roi_feat
            for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
                current = model.ConvGN(
                    current, 'head_conv' + str(i + 1) + '_2nd', dim_in, hidden_dim, 3,
                    group_gn=get_group_gn(hidden_dim),
                    stride=1, pad=1,
                    weight_init=('MSRAFill', {}),
                    bias_init=('ConstantFill', {'value': 0.}))
                current = model.Relu(current, current)
                dim_in = hidden_dim

            fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
            model.FC(current, 'fc6_2nd', dim_in * roi_size * roi_size, fc_dim)
            model.Relu('fc6_2nd', 'fc6_2nd')

            if not model.train:
                current_1st_2nd = roi_feat
                for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
                    current_1st_2nd = model.ConvGNShared(
                        current_1st_2nd, 'head_conv' + str(i + 1) + '_1st_2nd', dim_in, hidden_dim, 3,
                        group_gn=get_group_gn(hidden_dim),
                        stride=1, pad=1,
                        conv_weight_init='head_conv' + str(i + 1) + '_1st_w',
                        conv_bias_init=('ConstantFill', {'value': 0.}),
                        gn_bias_init='head_conv' + str(i + 1) + '_1st_b',
                        gn_scale_innit='head_conv' + str(i + 1) + '_1st_s')
                    current_1st_2nd = model.Relu(current_1st_2nd, current_1st_2nd)
                    dim_in = hidden_dim

                fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
                model.FCShared(current_1st_2nd, 'fc6_1st_2nd', dim_in * roi_size * roi_size, fc_dim)
                model.Relu('fc6_1st_2nd', 'fc6_1st_2nd')
            return ['fc6_2nd', 'fc6_1st_2nd'], fc_dim
            # return ['fc6_2nd'], fc_dim

        elif stage_num == 3:
            roi_feat = model.RoIFeatureTransform(
                blob_in, 'roi_feat_3rd',
                blob_rois='rois_3rd',
                method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
                resolution=roi_size,
                sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
                spatial_scale=spatial_scale
            )

            current = roi_feat
            for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
                current = model.ConvGN(
                    current, 'head_conv' + str(i + 1) + '_3rd', dim_in, hidden_dim, 3,
                    group_gn=get_group_gn(hidden_dim),
                    stride=1, pad=1,
                    weight_init=('MSRAFill', {}),
                    bias_init=('ConstantFill', {'value': 0.}))
                current = model.Relu(current, current)
                dim_in = hidden_dim

            fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
            model.FC(current, 'fc6_3rd', dim_in * roi_size * roi_size, fc_dim)
            model.Relu('fc6_3rd', 'fc6_3rd')

            if not model.train:
                current_1st_3rd = roi_feat
                for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
                    current_1st_3rd = model.ConvGNShared(
                        current_1st_3rd, 'head_conv' + str(i + 1) + '_1st_3rd', dim_in, hidden_dim, 3,
                        group_gn=get_group_gn(hidden_dim),
                        stride=1, pad=1,
                        conv_weight_init='head_conv' + str(i + 1) + '_1st_w',
                        conv_bias_init=('ConstantFill', {'value': 0.}),
                        gn_bias_init='head_conv' + str(i + 1) + '_1st_b',
                        gn_scale_innit='head_conv' + str(i + 1) + '_1st_s')
                    current_1st_3rd = model.Relu(current_1st_3rd, current_1st_3rd)
                    dim_in = hidden_dim

                fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
                model.FCShared(current_1st_3rd, 'fc6_1st_3rd', dim_in * roi_size * roi_size, fc_dim)
                model.Relu('fc6_1st_3rd', 'fc6_1st_3rd')

                current_2nd_3rd = roi_feat
                for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
                    current_2nd_3rd = model.ConvGNShared(
                        current_2nd_3rd, 'head_conv' + str(i + 1) + '_1st_3rd', dim_in, hidden_dim, 3,
                        group_gn=get_group_gn(hidden_dim),
                        stride=1, pad=1,
                        conv_weight_init='head_conv' + str(i + 1) + '_1st_w',
                        conv_bias_init=('ConstantFill', {'value': 0.}),
                        gn_bias_init='head_conv' + str(i + 1) + '_2nd_b',
                        gn_scale_innit='head_conv' + str(i + 1) + '_2nd_s'                    )
                    current_2nd_3rd = model.Relu(current_2nd_3rd, current_2nd_3rd)
                    dim_in = hidden_dim

                fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
                model.FCShared(current_2nd_3rd, 'fc6_2nd_3rd', dim_in * roi_size * roi_size, fc_dim)
                model.Relu('fc6_2nd_3rd', 'fc6_2nd_3rd')
            return ['fc6_3rd', 'fc6_1st_3rd', 'fc6_2nd_3rd'], fc_dim
            # return ['fc6_3rd'], fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim
