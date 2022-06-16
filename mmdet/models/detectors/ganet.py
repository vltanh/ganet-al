# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Yinchao Ma
# @Email   : imyc@mail.ustc.edu.cn
# --------------------------------------------------------

import torch
from mmdet.core import build_assigner
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

from .single_stage import SingleStageDetector
from ..builder import DETECTORS, build_loss


@DETECTORS.register_module
class GANet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss='LaneLossAggress',
                 loss_weights={},
                 output_scale=4,
                 num_classes=1,
                 point_scale=True,
                 sample_gt_points=[11, 11, 11, 11],
                 assigner_cfg=dict(type='LaneAssigner'),
                 use_smooth=False):
        super(GANet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained,
        )
        self.sample_gt_points = sample_gt_points
        self.num_classes = num_classes
        self.head = head
        self.use_smooth = use_smooth
        self.assigner_cfg = assigner_cfg
        self.loss_weights = loss_weights
        self.point_scale = point_scale
        if test_cfg is not None and 'out_scale' in test_cfg.keys():
            self.output_scale = test_cfg['out_scale']
        else:
            self.output_scale = output_scale
        self.loss = build_loss(loss)
        if self.assigner_cfg:
            self.assigner = build_assigner(self.assigner_cfg)

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        if img_metas is None:
            return self.test_inference(img, **kwargs)
        elif return_loss:
            # img shape [8 3 320 800]
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        # kwargs -> ['exist_mask', 'instance_mask', 'gauss_mask', 'hm_down_scale', 'lane_points']
        # shape [8 128 80 200]  swin [B C 40 100]
        # print(img_metas[0]['filename'])
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)  # features, deform_points
        if self.head:
            [cpts_hm, kpts_hm, pts_offset, int_offset] = \
                self.bbox_head.forward_train(
                    output['features'],
                    output.get("aux_feat", None),
            )
        cpts_hm = torch.clamp(torch.sigmoid(cpts_hm), min=1e-4, max=1 - 1e-4)
        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)

        loss_items = [
            {
                "type": "focalloss",
                "gt": kwargs['gt_cpts_hm'],
                "pred": cpts_hm,
                "weight": self.loss_weights["center"],
                "name": "center_focalloss",
            },
            {
                "type": "focalloss",
                "gt": kwargs['gt_kpts_hm'],
                "pred": kpts_hm,
                "weight": self.loss_weights["point"],
                "name": "point_focalloss",
            },
        ]
        if not self.use_smooth:
            loss_items.append({
                "type": "regl1kploss",
                "gt": kwargs['int_offset'],
                "pred": int_offset,
                "mask": kwargs['offset_mask'],
                "weight": self.loss_weights["error"],
                "name": "error_regl1kploss",
            })
            loss_items.append({
                "type": "regl1kploss",
                "gt": kwargs['pts_offset'],
                "pred": pts_offset,
                "mask": kwargs['offset_mask_weight'],
                "weight": self.loss_weights["offset"],
                "name": "offset_regl1kploss",
            })
        else:
            loss_items.append({
                "type": "smoothl1loss",
                "gt": kwargs['int_offset'],
                "pred": int_offset,
                "mask": kwargs['offset_mask'],
                "weight": self.loss_weights["error"],
                "name": "error_smoothl1loss",
            })
            loss_items.append({
                "type": "smoothl1loss",
                "gt": kwargs['pts_offset'],
                "pred": pts_offset,
                "mask": kwargs['offset_mask_weight'],
                "weight": self.loss_weights["offset"],
                "name": "offset_smoothl1loss",
            })

        if "deform_points" in output.keys() and self.loss_weights["aux"] != 0:
            for i, points in enumerate(output['deform_points']):
                if points is None:
                    continue
                gt_points = kwargs[f'lane_points_l{i}']
                gt_matched_points, pred_matched_points = \
                    self.assigner.assign(
                        points,
                        gt_points,
                        sample_gt_points=self.sample_gt_points[i],
                    )
                if self.point_scale:
                    loss_item = {
                        "type": "smoothl1loss",
                        "gt": gt_matched_points / (2 ** (3 - i)),
                        "pred": pred_matched_points / (2 ** (3 - i)),
                        "weight": self.loss_weights["aux"],
                        "name": "aux_smoothl1loss",
                    }
                else:
                    loss_item = {
                        "type": "smoothl1loss",
                        "gt": gt_matched_points,
                        "pred": pred_matched_points,
                        "weight": self.loss_weights["aux"],
                        "name": "aux_smoothl1loss",
                    }
                loss_items.append(loss_item)

        losses = self.loss(loss_items)
        return losses

    def test_inference(self, img, hack_seeds=None, **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)  # shape [8 64 80 200]
        if self.head:
            [cpts_hm, kpts_hm, pts_offset, int_offset] = self.bbox_head.forward_train(
                output['features'],
                output.get("aux_feat", None),
            )
            seeds, hm = self.bbox_head.forward_test(
                output['features'],
                output.get("aux_feat", None),
                hack_seeds,
                kwargs['thr'],
                kwargs['kpt_thr'],
                kwargs['cpt_thr'],
            )
        output['cpts_hm'] = cpts_hm
        output['kpts_hm'] = kpts_hm
        output['pts_offset'] = pts_offset
        output['int_offset'] = int_offset
        output['deform_points'] = output['deform_points']
        output['seeds'] = seeds
        output['hm'] = hm
        return output

    def forward_test(self, img, img_metas,
                     hack_seeds=None,
                     **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(
                output['features'],
                output.get("aux_feat", None),
                hack_seeds,
                kwargs['thr'],
                kwargs['kpt_thr'],
                kwargs['cpt_thr'],
            )
        return [seeds, hm]

    def forward_dummy(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        x = self.bbox_head.forward_train(
            x['features'], x.get("aux_feat", None))
        return x


class Teacher:
    def __init__(self, cfg, ckpt) -> None:
        cfg = Config.fromfile(cfg)
        self.teacher = build_detector(cfg.model)
        load_checkpoint(self.teacher, ckpt)
        self.teacher.cuda()
        for param in self.teacher.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def gen_label(self, img):
        self.teacher.eval()
        output_t = self.teacher.backbone(img.type(torch.cuda.FloatTensor))
        output_t = self.teacher.neck(output_t)
        if self.teacher.head:
            [cpts_hm_t, kpts_hm_t, pts_offset_t, int_offset_t] = \
                self.teacher.bbox_head.forward_train(
                    output_t['features'],
                    output_t.get("aux_feat", None),
            )
        cpts_hm_t = torch.clamp(
            torch.sigmoid(cpts_hm_t),
            min=1e-4,
            max=1 - 1e-4,
        )
        kpts_hm_t = torch.clamp(
            torch.sigmoid(kpts_hm_t),
            min=1e-4,
            max=1 - 1e-4,
        )
        return cpts_hm_t, kpts_hm_t, pts_offset_t, int_offset_t


@DETECTORS.register_module
class GANet_KD(SingleStageDetector):
    def __init__(self,
                 teacher_cfg=None,
                 teacher_ckpt=None,
                 backbone=None,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss='LaneLossAggress',
                 loss_weights={},
                 output_scale=4,
                 num_classes=1,
                 point_scale=True,
                 sample_gt_points=[11, 11, 11, 11],
                 assigner_cfg=dict(type='LaneAssigner'),
                 use_smooth=False):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained,
        )
        self.sample_gt_points = sample_gt_points
        self.num_classes = num_classes
        self.head = head
        self.use_smooth = use_smooth
        self.assigner_cfg = assigner_cfg
        self.loss_weights = loss_weights
        self.point_scale = point_scale
        if test_cfg is not None and 'out_scale' in test_cfg.keys():
            self.output_scale = test_cfg['out_scale']
        else:
            self.output_scale = output_scale
        self.loss = build_loss(loss)
        if self.assigner_cfg:
            self.assigner = build_assigner(self.assigner_cfg)

        if teacher_cfg is not None and teacher_ckpt is not None:
            self.teacher = Teacher(teacher_cfg, teacher_ckpt)

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        if img_metas is None:
            return self.test_inference(img, **kwargs)
        elif return_loss:
            # img shape [8 3 320 800]
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        # kwargs -> ['exist_mask', 'instance_mask', 'gauss_mask', 'hm_down_scale', 'lane_points']
        # shape [8 128 80 200]  swin [B C 40 100]

        # Teacher's prediction
        cpts_hm_t, kpts_hm_t, pts_offset_t, int_offset_t = \
            self.teacher.gen_label(img)

        # Student's prediction
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)  # features, deform_points
        if self.head:
            [cpts_hm, kpts_hm, pts_offset, int_offset] = \
                self.bbox_head.forward_train(
                    output['features'],
                    output.get("aux_feat", None),
            )
        cpts_hm = torch.clamp(torch.sigmoid(cpts_hm), min=1e-4, max=1 - 1e-4)
        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)

        loss_items = [
            {
                "type": "focalloss",
                "gt": kwargs['gt_cpts_hm'],
                "pred": cpts_hm,
                "weight": self.loss_weights["center"],
                "name": "center_focalloss",
            },
            {
                "type": "focalloss",
                "gt": kwargs['gt_kpts_hm'],
                "pred": kpts_hm,
                "weight": self.loss_weights["point"],
                "name": "point_focalloss",
            },

            # KD-loss
            {
                "type": "focalloss",
                "gt": cpts_hm_t,
                "pred": cpts_hm,
                "weight": self.loss_weights["center"],
                "name": "center_focalloss_kd",
            },
            {
                "type": "focalloss",
                "gt": kpts_hm_t,
                "pred": kpts_hm,
                "weight": self.loss_weights["point"],
                "name": "point_focalloss_kd",
            },
        ]
        if not self.use_smooth:
            loss_items.append({
                "type": "regl1kploss",
                "gt": kwargs['int_offset'],
                "pred": int_offset,
                "mask": kwargs['offset_mask'],
                "weight": self.loss_weights["error"],
                "name": "error_regl1kploss",
            })
            loss_items.append({
                "type": "regl1kploss",
                "gt": kwargs['pts_offset'],
                "pred": pts_offset,
                "mask": kwargs['offset_mask_weight'],
                "weight": self.loss_weights["offset"],
                "name": "offset_regl1kploss",
            })

            # KD-loss
            loss_items.append({
                "type": "regl1kploss",
                "gt": int_offset_t,
                "pred": int_offset,
                "mask": kwargs['offset_mask'],
                "weight": self.loss_weights["error"],
                "name": "error_regl1kploss_kd",
            })
            loss_items.append({
                "type": "regl1kploss",
                "gt": pts_offset_t,
                "pred": pts_offset,
                "mask": kwargs['offset_mask_weight'],
                "weight": self.loss_weights["offset"],
                "name": "offset_regl1kploss_kd",
            })
        else:
            loss_items.append({
                "type": "smoothl1loss",
                "gt": kwargs['int_offset'],
                "pred": int_offset,
                "mask": kwargs['offset_mask'],
                "weight": self.loss_weights["error"],
                "name": "error_smoothl1loss",
            })
            loss_items.append({
                "type": "smoothl1loss",
                "gt": kwargs['pts_offset'],
                "pred": pts_offset,
                "mask": kwargs['offset_mask_weight'],
                "weight": self.loss_weights["offset"],
                "name": "offset_smoothl1loss",
            })

            # KD-loss
            loss_items.append({
                "type": "smoothl1loss",
                "gt": int_offset_t,
                "pred": int_offset,
                "mask": kwargs['offset_mask'],
                "weight": self.loss_weights["error"],
                "name": "error_smoothl1loss_kd",
            })
            loss_items.append({
                "type": "smoothl1loss",
                "gt": pts_offset_t,
                "pred": pts_offset,
                "mask": kwargs['offset_mask_weight'],
                "weight": self.loss_weights["offset"],
                "name": "offset_smoothl1loss_kd",
            })

        if "deform_points" in output.keys() and self.loss_weights["aux"] != 0:
            for i, points in enumerate(output['deform_points']):
                if points is None:
                    continue
                gt_points = kwargs[f'lane_points_l{i}']
                gt_matched_points, pred_matched_points = \
                    self.assigner.assign(
                        points,
                        gt_points,
                        sample_gt_points=self.sample_gt_points[i],
                    )
                if self.point_scale:
                    loss_item = {
                        "type": "smoothl1loss",
                        "gt": gt_matched_points / (2 ** (3 - i)),
                        "pred": pred_matched_points / (2 ** (3 - i)),
                        "weight": self.loss_weights["aux"],
                        "name": "aux_smoothl1loss",
                    }
                else:
                    loss_item = {
                        "type": "smoothl1loss",
                        "gt": gt_matched_points,
                        "pred": pred_matched_points,
                        "weight": self.loss_weights["aux"],
                        "name": "aux_smoothl1loss",
                    }
                loss_items.append(loss_item)

        losses = self.loss(loss_items)
        return losses

    def test_inference(self, img, hack_seeds=None, **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)  # shape [8 64 80 200]
        if self.head:
            [cpts_hm, kpts_hm, pts_offset, int_offset] = self.bbox_head.forward_train(
                output['features'],
                output.get("aux_feat", None),
            )
            seeds, hm = self.bbox_head.forward_test(
                output['features'],
                output.get("aux_feat", None),
                hack_seeds,
                kwargs['thr'],
                kwargs['kpt_thr'],
                kwargs['cpt_thr'],
            )
        output['cpts_hm'] = cpts_hm
        output['kpts_hm'] = kpts_hm
        output['pts_offset'] = pts_offset
        output['int_offset'] = int_offset
        output['deform_points'] = output['deform_points']
        output['seeds'] = seeds
        output['hm'] = hm
        return output

    def forward_test(self, img, img_metas,
                     hack_seeds=None,
                     **kwargs):
        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(
                output['features'],
                output.get("aux_feat", None),
                hack_seeds,
                kwargs['thr'],
                kwargs['kpt_thr'],
                kwargs['cpt_thr'],
            )
        return [seeds, hm]

    def forward_dummy(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        x = self.bbox_head.forward_train(
            x['features'], x.get("aux_feat", None))
        return x
