# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)

from torch import nn 
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)

from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference

from ubteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers


import numpy as np
from detectron2.modeling.poolers import ROIPooler, convert_boxes_to_pooler_format
from detectron2.modeling.matcher import Matcher


@ROI_HEADS_REGISTRY.register()
class R3ROIHEADS(CascadeROIHeads):
    """
    The ROI heads that implement :paper:`Cascade R-CNN`.
    """
    @configurable
    def __init__(
        #cascade originale chiama le variabili che sono def qui
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        stages,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            box_pooler (ROIPooler): pooler that extracts region features from given boxes
            box_heads (list[nn.Module]): box head for each cascade stage
            box_predictors (list[nn.Module]): box predictor for each cascade stage
            proposal_matchers (list[Matcher]): matcher with different IoU thresholds to
                match boxes with ground truth for each stage. The first matcher matches
                RPN proposals with ground truth, the other matchers use boxes predicted
                by the previous stage as proposals and match them with ground truth.
        """
        assert "proposal_matcher" not in kwargs, (
            "CascadeROIHeads takes 'proposal_matchers=' for each stage instead "
            "of one 'proposal_matcher='."
        )
        #The first matcher matches RPN proposals with ground truth, done in the base class
        self.stages=stages

        assert "proposal_matcher" not in kwargs, (
            "CascadeROIHeads takes 'proposal_matchers=' for each stage instead "
            "of one 'proposal_matcher='."
        )
        # The first matcher matches RPN proposals with ground truth, done in the base class
        kwargs["proposal_matcher"] = proposal_matchers[0]
        #num_stages = self.num_cascade_stages = len(box_heads)
        box_heads = nn.ModuleList(box_heads)
        box_predictors = nn.ModuleList(box_predictors)

        super(CascadeROIHeads, self).__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_heads,
            box_predictor=box_predictors,
            **kwargs,
        )
        self.proposal_matchers = proposal_matchers
        self.num_cascade_stages=len(stages)

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        stages                   = cfg.MODEL.ROI_STAGES
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        
        for i in range(len(set(stages))):
            # FIXME check bbox_reg_weights
            bbox_reg_weights = cascade_bbox_reg_weights[i]
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                FastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
        for match_iou in cascade_ious:
            proposal_matchers.append(Matcher([match_iou], [0,1], allow_low_quality_matches=False))

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
            "stages":stages,
        }


    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
            #do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, proposals, targets, compute_loss, compute_val_loss, branch
            )
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, targets, compute_loss, compute_val_loss, branch
            )
            return pred_instances, predictions

    def _forward_box(
        self,
        features,
        proposals,
        targets=None,
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = ""):
        """
        Args:
            features, targets: the same as in
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  #(predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        
        for k in range(self.num_cascade_stages):
            
            idx=self.stages[k]
            
            if k > 0:
                # The output boxes of the previous stage are used to create the input
                # proposals of the next stage.
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                
                if (self.training and compute_loss) or compute_val_loss:          #Distingue le fasi di trading e il calcolo della loss 
                    proposals = self._match_and_label_boxes(proposals, k, targets)

            predictions = self._run_stage(features, proposals, idx)
            prev_pred_boxes = self.box_predictor[idx].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[idx], predictions, proposals))


        if (self.training and compute_loss) or compute_val_loss:  #apply if training loss or val loss
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses, predictions
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            # Use the boxes of the last head
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self,
        proposals: List[Instances],
        targets: List[Instances],
        branch: str = ""
    ) -> List[Instances]:
        #Same as StandardROIHeadsPseudoLab 
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt

def build_pre_processing(cfg):
    
    conv_type=cfg.MODEL.ROI_CONV_PRE

    if conv_type == "Conv3":
        pre_module=torch.nn.Conv2d(256,256,kernel_size=3,padding=1)
    
    elif conv_type == "Conv5":
        pre_module=torch.nn.Conv2d(256,256,kernel_size=5,padding=2)
    
    elif conv_type == "Conv7":
        pre_module=torch.nn.Conv2d(256,256,kernel_size=7,padding=3)
        
    elif conv_type == "NL":
        
        pre_module=NONLocalBlock2D(256, sub_sample=False, kernel_size = cfg.MODEL.NN_LOCAL_BLOCK_PRE[0], padding=cfg.MODEL.NN_LOCAL_BLOCK_PRE[1])
    
    return pre_module

def build_post_processing(cfg):
    
    conv_type=cfg.MODEL.ROI_CONV_POST

    if conv_type == "Conv3":
        post_module=torch.nn.Conv2d(256,256,kernel_size=3,padding=1)
    
    elif conv_type == "Conv5":
        post_module=torch.nn.Conv2d(256,256,kernel_size=5,padding=2)
    
    elif conv_type == "Conv7":
        post_module=torch.nn.Conv2d(256,256,kernel_size=7,padding=3)
        
    elif conv_type == "NL":
        post_module=NONLocalBlock2D(256, sub_sample=False, kernel_size = cfg.MODEL.NN_LOCAL_BLOCK_POST[0], padding=cfg.MODEL.NN_LOCAL_BLOCK_POST[1])

    return post_module



def build_roi_extractor(cfg, input_shape):

    model_type= cfg.MODEL.ROI_BOX_EXTRACTOR

    if model_type == "ROIPooler":

        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

    else:
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        box_pooler = GRoIE(
            cfg=cfg,
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
    return box_pooler

class GRoIE(ROIPooler): 

    def __init__ (
        self,
        cfg,
        *args,**kwargs
    ):
        super().__init__(*args,**kwargs)
        self.pre_processing=build_pre_processing(cfg)
        self.post_processing=build_post_processing(cfg)

    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        
        num_level_assignments = len(x)
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        roi_feats = x[0].new_zeros(
            pooler_fmt_boxes.size(0), num_channels, output_size, output_size)
    
        
        #Apply the pooler to a levele and extract the features
        for i in range(num_level_assignments):
            roi_features_t = self.level_poolers[i](x[i], pooler_fmt_boxes) #x is feats[i], pooler_fmt_boxes is rois
            
            #apply pre-processing to a RoI extracted from each layer
            roi_features_t = self.pre_processing(roi_features_t)

            roi_features_t = nn.ReLU()(roi_features_t)

            # and sum them all
            roi_feats += roi_features_t
        #apply post-processing (sum) before return the result
        roi_feats = self.post_processing(roi_feats)

        roi_feats = nn.ReLU()(roi_feats)

        return roi_feats


class _NonLocalBlockND(nn.Module):
    def __init__(self, 
                in_channels, 
                inter_channels=None, 
                dimension=3, 
                sub_sample=True, 
                bn_layer=True,
                kernel_size=3,
                padding=1):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3: #Non serve
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=kernel_size, stride=1, padding=padding)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=kernel_size, stride=1, padding=padding),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=kernel_size, stride=1, padding=padding)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=kernel_size, stride=1, padding=padding)
                             
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=kernel_size, stride=1, padding=padding)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, kernel_size=3, padding=1):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              kernel_size=kernel_size,
                                              padding=padding)


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        #pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        #sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        # fmt: on
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = build_roi_extractor(
            cfg,
            input_shape
        )
        
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if (self.training and compute_loss) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self,
        proposals: List[Instances],
        targets: List[Instances],
        branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt