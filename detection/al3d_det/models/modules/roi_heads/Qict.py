import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from al3d_utils import common_utils, voxel_aggregation_utils, density_utils
from al3d_utils.ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG, StackSAModuleMSGAttention

from inspect import isfunction
from al3d_det.utils import loss_utils, box_coder_utils
from al3d_det.utils.model_nms_utils import class_agnostic_nms
from al3d_det.utils.attention_utils import TransformerEncoder, get_positional_encoder
from al3d_det.models import fusion_modules 
from .proposal_target_layer import ProposalTargetLayer

class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:
        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)
        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
            
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)
        Returns:
        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        if cls_preds is None:
            batch_cls_preds = None
        else:
            batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds


class VoxelAggregationHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        layer_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.ffn = nn.Sequential(
            nn.Conv1d(7, 64 // 2, 1),
            nn.BatchNorm1d(64 // 2),
            nn.ReLU(64 // 2),
            nn.Conv1d(64 // 2, 64, 1),
        )
        self.up_ffn = nn.Sequential(
            nn.Conv1d(64, 192 // 2, 1),
            nn.BatchNorm1d(192 // 2),
            nn.ReLU(192 // 2),
            nn.Conv1d(192 // 2, 192, 1),
        )
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for i, src_name in enumerate(self.pool_cfg.FEATURE_LOCATIONS):
            mlps = layer_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [self.model_cfg.VOXEL_AGGREGATION.NUM_FEATURES[i]] + mlps[k]
            stack_sa_module_msg = StackSAModuleMSGAttention if self.pool_cfg.get('ATTENTION', {}).get('ENABLED') else StackSAModuleMSG
            pool_layer = stack_sa_module_msg(
                radii=layer_cfg[src_name].POOL_RADIUS,
                nsamples=layer_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method=layer_cfg[src_name].POOL_METHOD,
                use_density=layer_cfg[src_name].get('USE_DENSITY')
            )

            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            assert self.pool_cfg.ATTENTION.NUM_FEATURES == c_out, f'ATTENTION.NUM_FEATURES must equal voxel aggregation output dimension of {c_out}.'
            pos_encoder = get_positional_encoder(self.pool_cfg)
            self.attention_head = TransformerEncoder(self.pool_cfg.ATTENTION, pos_encoder)
            for p in self.attention_head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        self.crossattention_head = fusion_modules.__all__[self.pool_cfg.FUSION.NAME](
                fuse_mode=self.pool_cfg.FUSION.FUSE_MODE, 
                interpolate=False, 
                voxel_size=voxel_size, 
                pc_range=point_cloud_range, 
                image_list=self.pool_cfg.FUSION.CAMERAS, 
                image_scale=self.pool_cfg.FUSION.IMAGE_SCALE, 
                depth_thres=self.pool_cfg.FUSION.DEPTH_THRES, 
                layer_channel=self.pool_cfg.FUSION.LAYER_CHANNEL,
                mid_channels=self.pool_cfg.FUSION.MID_CHANNELS,
                double_flip=False, 
                dropout_ratio=0,
                activate_out=True,
                fuse_out=True
        )
        self.crossattention_pointhead = fusion_modules.__all__[self.pool_cfg.FUSION.NAME](
                fuse_mode=self.pool_cfg.FUSION.FUSE_MODE, 
                interpolate=False, 
                voxel_size=voxel_size, 
                pc_range=point_cloud_range, 
                image_list=self.pool_cfg.FUSION.CAMERAS, 
                image_scale=self.pool_cfg.FUSION.IMAGE_SCALE, 
                depth_thres=self.pool_cfg.FUSION.DEPTH_THRES, 
                layer_channel=self.pool_cfg.FUSION.LAYER_CHANNEL,
                mid_channels=self.pool_cfg.FUSION.MID_CHANNELS,
                double_flip=False, 
                dropout_ratio=0,
                activate_out=True,
                fuse_out=True
        )
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """
        batch_size = batch_dict['batch_size']
        batch_rois = batch_dict['rois']

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            batch_dict, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
        new_xyz = global_roi_grid_points.view(-1, 3)

        pooled_features_list = []
        ball_idxs_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURE_LOCATIONS):
            point_coords = batch_dict['point_coords'][src_name]
            point_features = batch_dict['point_features'][src_name]
            
            pool_layer = self.roi_grid_pool_layers[k]

            xyz = point_coords[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            batch_idx = point_coords[:, 0]
            for k in range(batch_size):
                xyz_batch_cnt[k] = (batch_idx == k).sum()

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
            pool_output = pool_layer(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features.contiguous(),
            )  # (M1 + M2 ..., C)
            if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
                _, pooled_features, ball_idxs = pool_output
            else:
                _, pooled_features = pool_output

            pooled_features = pooled_features.view(
                -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)

            if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
                ball_idxs = ball_idxs.view(
                    -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE **3,
                    ball_idxs.shape[-1]
                )
                ball_idxs_list.append(ball_idxs)

        all_pooled_features = torch.cat(pooled_features_list, dim=-1)
        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            all_ball_idxs = torch.cat(ball_idxs_list, dim=-1)
        else:
            all_ball_idxs = []
        return all_pooled_features, global_roi_grid_points, local_roi_grid_points, all_ball_idxs

    def get_global_grid_points_of_roi(self, batch_dict, grid_size):
        rois = batch_dict['rois']
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)

        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_point_voxel_features(self, batch_dict):
        raise NotImplementedError

    def get_localgrid_input(self, points, rois, local_roi_grid_points):
        points_per_part = density_utils.find_num_points_per_part_multi(points,
                                                                       rois,
                                                                       self.model_cfg.ROI_GRID_POOL.GRID_SIZE,
                                                                       self.pool_cfg.DENSITYQUERY.MAX_NUM_BOXES,
                                                                       return_centroid=True)
        points_per_part_num_features = 1 if len(points_per_part.shape) <= 5 else points_per_part.shape[-1]
        points_per_part = points_per_part.view(points_per_part.shape[0]*points_per_part.shape[1], -1, points_per_part_num_features).float()
        # First feature is density, other potential features are xyz
        points_per_part[..., 0] = torch.log10(points_per_part[..., 0] + 0.5) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
        if self.pool_cfg.DENSITYQUERY.POSITIONAL_ENCODER == 'grid_points':
            positional_input = local_roi_grid_points
        elif self.pool_cfg.DENSITYQUERY.POSITIONAL_ENCODER == 'density':
            positional_input = points_per_part
        elif self.pool_cfg.DENSITYQUERY.POSITIONAL_ENCODER == 'density_grid_points':
            positional_input = torch.cat((local_roi_grid_points, points_per_part), dim=-1)
        else:
            positional_input = None
        return positional_input

    def get_positional_input(self, points, rois, local_roi_grid_points):
        points_per_part = density_utils.find_num_points_per_part_multi(points,
                                                                       rois,
                                                                       self.model_cfg.ROI_GRID_POOL.GRID_SIZE,
                                                                       self.pool_cfg.ATTENTION.MAX_NUM_BOXES,
                                                                       return_centroid=self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_centroid')
        points_per_part_num_features = 1 if len(points_per_part.shape) <= 5 else points_per_part.shape[-1]
        points_per_part = points_per_part.view(points_per_part.shape[0]*points_per_part.shape[1], -1, points_per_part_num_features).float()
        # First feature is density, other potential features are xyz
        points_per_part[..., 0] = torch.log10(points_per_part[..., 0] + 0.5) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
        if self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'grid_points':
            positional_input = local_roi_grid_points
        elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density':
            positional_input = points_per_part
        elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_grid_points':
            positional_input = torch.cat((local_roi_grid_points, points_per_part), dim=-1)
        else:
            positional_input = None
        return positional_input

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        batch_dict['point_features'], batch_dict['point_coords'] = self.get_point_voxel_features(batch_dict)

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        # RoI aware pooling
        pooled_features, global_roi_grid_points, local_roi_grid_points, ball_idxs = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        batch_size_rcnn = pooled_features.shape[0]
        if self.pool_cfg.get('DENSITYQUERY', {}).get('ENABLED'):
            src_key_padding_mask = None
            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE

            localgrid_densityfeat = self.get_localgrid_input(batch_dict['points'], batch_dict['rois'], local_roi_grid_points)
            localgrid_densityfeat = self.ffn(localgrid_densityfeat.permute(0, 2, 1))
            localgrid_densityfeat = localgrid_densityfeat.reshape(-1, 64)
            grid_coords_idlist = []
            for idx in range(batch_dict['batch_size']):
                batch_idx = torch.ones([localgrid_densityfeat.shape[0]//batch_dict['batch_size'], 1], dtype=batch_dict['points'][:, 0].dtype) * idx
                grid_coords_idlist.append(batch_idx)
            grid_coordid = torch.cat(grid_coords_idlist, dim=0).to(global_roi_grid_points.device)
            grid_coords = torch.cat((grid_coordid, global_roi_grid_points.view(-1, 3)), dim=-1)
            localgrid_densityfeat_fuse = self.crossattention_pointhead(batch_dict, point_features=localgrid_densityfeat, point_coords=grid_coords, layer_name="layer1")
            localgrid_densityfeat_fuse = localgrid_densityfeat_fuse.reshape(pooled_features.shape[0], pooled_features.shape[1], 64)
            localgrid_densityfeat_fuse = self.up_ffn(localgrid_densityfeat_fuse.permute(0, 2, 1))
            if self.pool_cfg.DENSITYQUERY.get('COMBINE'):
                pooled_features = pooled_features + localgrid_densityfeat_fuse.permute(0, 2, 1)

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            src_key_padding_mask = None
            if self.pool_cfg.ATTENTION.get('MASK_EMPTY_POINTS'):
                src_key_padding_mask = (ball_idxs == 0).all(-1)

            positional_input = self.get_positional_input(batch_dict['points'], batch_dict['rois'], local_roi_grid_points)
            # Attention
            attention_output = self.attention_head(pooled_features, positional_input, src_key_padding_mask) # (BxN, 6x6x6, C)

            if self.pool_cfg.ATTENTION.get('COMBINE'):
                attention_output = pooled_features + attention_output

            # Permute
            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            batch_size_rcnn = attention_output.shape[0]
            pooled_features = attention_output.permute(0, 2, 1).\
                contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size) # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        rcnn_cls = self.cls_layers(shared_features)

        rcnn_cls = rcnn_cls.transpose(1, 2).contiguous().squeeze(dim=1)
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict



class SELF(VoxelAggregationHead):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, point_cloud_range, voxel_size, num_class, kwargs=kwargs)

    def default(val, default_val):
        default_val = default_val() if isfunction(default_val) else default_val
        return val if val is not None else default_val

    def cast_tuple(el):
        return el if isinstance(el, tuple) else (el,)

    # tensor related helper functions

    def top1(t):
        values, index = t.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def cumsum_exclusive(t, dim=-1):
        num_dims = len(t.shape)
        num_pad_dims = - dim - 1
        pre_padding = (0, 0) * num_pad_dims
        pre_slice   = (slice(None),) * num_pad_dims
        padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
        return padded_t[(..., slice(None, -1), *pre_slice)]

    # pytorch one hot throws an error if there are out of bound indices.
    # tensorflow, in contrast, does not throw an error
    def safe_one_hot(indexes, max_length):
        max_index = indexes.max() + 1
        return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

    def init_(t):
        dim = t.shape[-1]
        std = 1 / math.sqrt(dim)
        return t.uniform_(-std, std)

    # activations

    class GELU_(nn.Module):
        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

    # expert class

    class Experts(nn.Module):
        def __init__(self,
            dim,
            num_experts = 16,
            hidden_dim = None,
            activation = GELU):
            super().__init__()

            hidden_dim = default(hidden_dim, dim * 4)
            num_experts = cast_tuple(num_experts)

            w1 = torch.zeros(*num_experts, dim, hidden_dim)
            w2 = torch.zeros(*num_experts, hidden_dim, dim)

            w1 = init_(w1)
            w2 = init_(w2)

            self.w1 = nn.Parameter(w1)
            self.w2 = nn.Parameter(w2)
            self.act = activation()

        def forward(self, x):
            hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
            hidden = self.act(hidden)
            out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
            return out

    # the below code is almost all transcribed from the official tensorflow version, from which the papers are written
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py

    # gating network

    class Top2Gating(nn.Module):
        def __init__(
            self,
            dim,
            num_gates,
            eps = 1e-9,
            outer_expert_dims = tuple(),
            second_policy_train = 'random',
            second_policy_eval = 'random',
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,
            capacity_factor_eval = 2.):
            super().__init__()

            self.eps = eps
            self.num_gates = num_gates
            self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

            self.second_policy_train = second_policy_train
            self.second_policy_eval = second_policy_eval
            self.second_threshold_train = second_threshold_train
            self.second_threshold_eval = second_threshold_eval
            self.capacity_factor_train = capacity_factor_train
            self.capacity_factor_eval = capacity_factor_eval

        def forward(self, x, importance = None):
            *_, b, group_size, dim = x.shape
            num_gates = self.num_gates

            if self.training:
                policy = self.second_policy_train
                threshold = self.second_threshold_train
                capacity_factor = self.capacity_factor_train
            else:
                policy = self.second_policy_eval
                threshold = self.second_threshold_eval
                capacity_factor = self.capacity_factor_eval

            raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
            raw_gates = raw_gates.softmax(dim=-1)

            # FIND TOP 2 EXPERTS PER POSITON
            # Find the top expert for each position. shape=[batch, group]

            gate_1, index_1 = top1(raw_gates)
            mask_1 = F.one_hot(index_1, num_gates).float()
            density_1_proxy = raw_gates

            if importance is not None:
                equals_one_mask = (importance == 1.).float()
                mask_1 *= equals_one_mask[..., None]
                gate_1 *= equals_one_mask
                density_1_proxy = density_1_proxy * equals_one_mask[..., None]
                del equals_one_mask

            gates_without_top_1 = raw_gates * (1. - mask_1)

            gate_2, index_2 = top1(gates_without_top_1)
            mask_2 = F.one_hot(index_2, num_gates).float()

            if importance is not None:
                greater_zero_mask = (importance > 0.).float()
                mask_2 *= greater_zero_mask[..., None]
                del greater_zero_mask

            # normalize top2 gate scores
            denom = gate_1 + gate_2 + self.eps
            gate_1 /= denom
            gate_2 /= denom

            # BALANCING LOSSES
            # shape = [batch, experts]
            # We want to equalize the fraction of the batch assigned to each expert
            density_1 = mask_1.mean(dim=-2)
            # Something continuous that is correlated with what we want to equalize.
            density_1_proxy = density_1_proxy.mean(dim=-2)
            loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

            # Depending on the policy in the hparams, we may drop out some of the
            # second-place experts.
            if policy == "all":
                pass
            elif policy == "none":
                mask_2 = torch.zeros_like(mask_2)
            elif policy == "threshold":
                mask_2 *= (gate_2 > threshold).float()
            elif policy == "random":
                probs = torch.zeros_like(gate_2).uniform_(0., 1.)
                mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
            else:
                raise ValueError(f"Unknown policy {policy}")

            # Each sequence sends (at most?) expert_capacity positions to each expert.
            # Static expert_capacity dimension is needed for expert batch sizes
            expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
            expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
            expert_capacity_f = float(expert_capacity)

            # COMPUTE ASSIGNMENT TO EXPERTS
            # [batch, group, experts]
            # This is the position within the expert's mini-batch for this sequence
            position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
            # Remove the elements that don't fit. [batch, group, experts]
            mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
            # [batch, experts]
            # How many examples in this sequence go to this expert
            mask_1_count = mask_1.sum(dim=-2, keepdim=True)
            # [batch, group] - mostly ones, but zeros where something didn't fit
            mask_1_flat = mask_1.sum(dim=-1)
            # [batch, group]
            position_in_expert_1 = position_in_expert_1.sum(dim=-1)
            # Weight assigned to first expert.  [batch, group]
            gate_1 *= mask_1_flat

            position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
            position_in_expert_2 *= mask_2
            mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
            mask_2_flat = mask_2.sum(dim=-1)

            position_in_expert_2 = position_in_expert_2.sum(dim=-1)
            gate_2 *= mask_2_flat
            
            # [batch, group, experts, expert_capacity]
            combine_tensor = (
                gate_1[..., None, None]
                * mask_1_flat[..., None, None]
                * F.one_hot(index_1, num_gates)[..., None]
                * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
                gate_2[..., None, None]
                * mask_2_flat[..., None, None]
                * F.one_hot(index_2, num_gates)[..., None]
                * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
            )

            dispatch_tensor = combine_tensor.bool().to(combine_tensor)
            return dispatch_tensor, combine_tensor, loss

    # plain mixture of experts

    class MoE(nn.Module):
        def __init__(self,
            dim,
            num_experts = 16,
            hidden_dim = None,
            activation = nn.ReLU,
            second_policy_train = 'random',
            second_policy_eval = 'random',
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,
            capacity_factor_eval = 2.,
            loss_coef = 1e-2,
            experts = None):
            super().__init__()

            self.num_experts = num_experts

            gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
            self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
            self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
            self.loss_coef = loss_coef

        def forward(self, inputs, **kwargs):
            b, n, d, e = *inputs.shape, self.num_experts
            dispatch_tensor, combine_tensor, loss = self.gate(inputs)
            expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

            # Now feed the expert inputs through the experts.
            orig_shape = expert_inputs.shape
            expert_inputs = expert_inputs.reshape(e, -1, d)
            expert_outputs = self.experts(expert_inputs)
            expert_outputs = expert_outputs.reshape(*orig_shape)

            output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
            return output, loss * self.loss_coef

    # 2-level heirarchical mixture of experts

    class HeirarchicalMoE(nn.Module):
        def __init__(self,
            dim,
            num_experts = (4, 4),
            hidden_dim = None,
            activation = nn.ReLU,
            second_policy_train = 'random',
            second_policy_eval = 'random',
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,
            capacity_factor_eval = 2.,
            loss_coef = 1e-2,
            experts = None):
            super().__init__()

            assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
            num_experts_outer, num_experts_inner = num_experts
            self.num_experts_outer = num_experts_outer
            self.num_experts_inner = num_experts_inner

            gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

            self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
            self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

            self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
            self.loss_coef = loss_coef

        def forward(self, inputs, **kwargs):
            b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
            dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
            expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

            # we construct an "importance" Tensor for the inputs to the second-level
            # gating.  The importance of an input is 1.0 if it represents the
            # first-choice expert-group and 0.5 if it represents the second-choice expert
            # group.  This is used by the second-level gating.
            importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
            importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

            dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
            expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

            # Now feed the expert inputs through the experts.
            orig_shape = expert_inputs.shape
            expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
            expert_outputs = self.experts(expert_inputs)
            expert_outputs = expert_outputs.reshape(*orig_shape)

            # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
            # expert_output has shape [y0, x1, h, d, n]

            expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
            output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
            return output, (loss_outer + loss_inner) * self.loss_coef
        

class GAT(VoxelAggregationHead):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, point_cloud_range, voxel_size, num_class, kwargs=kwargs)

    class GAT(nn.Module):
        def __init__(self, dim, num_heads = 8, hidden_dim = None, activation = nn.ReLU):
            super().__init__()

            hidden_dim = default(hidden_dim, dim * 4)
            self.num_heads = num_heads

            self.to_qkv = nn.Linear(dim, 3 * dim, bias = False)
            self.to_out = nn.Linear(dim, dim)

            self.w = nn.Parameter(torch.randn(num_heads, dim, dim))
            self.b = nn.Parameter(torch.zeros(num_heads, 1, dim))

            self.act = activation()

    class LiDARImageDataset(Dataset):
        def __init__(self, lidar_data, image_data, transform=None):
            self.lidar_data = lidar_data
            self.image_data = image_data
            self.transform = transform

        def __len__(self):
            return len(self.lidar_data)

        def __getitem__(self, idx):
            lidar = self.lidar_data[idx]
            image = Image.fromarray(self.image_data[idx])
            if self.transform:
                image = self.transform(image)
            return lidar, image

    class AdiabaticReversibleBlock(nn.Module):
        def __init__(self, model_dim):
            super(AdiabaticReversibleBlock, self).__init__()
            self.F_block = nn.Linear(model_dim, model_dim)
            self.G_block = nn.Linear(model_dim, model_dim)
            self.initialize_hardware()

        def initialize_hardware(self):
            pass

        def forward(self, x1, x2):
            y1 = x1 + self.adi_forward_F(x2)
            y2 = x2 + self.adi_forward_G(y1)
            return y1, y2

        def backward(self, y1, y2):
            x2 = y2 - self.adi_backward_G(y1)
            x1 = y1 - self.adi_backward_F(x2)
            return x1, x2

        def adi_forward_F(self, x):
            return self.F_block(x)

        def adi_forward_G(self, x):
            return self.G_block(x)

        def adi_backward_F(self, x):
            return self.F_block(x)

        def adi_backward_G(self, x):
            return self.G_block(x)

    class AdiabaticTransformerLayer(nn.Module):
        def __init__(self, model_dim):
            super(AdiabaticTransformerLayer, self).__init__()
            self.reversible_block = AdiabaticReversibleBlock(model_dim)

        def forward(self, x):
            x1, x2 = torch.chunk(x, 2, dim=-1)
            x1, x2 = self.reversible_block(x1, x2)
            return torch.cat((x1, x2), dim=-1)

        def backward(self, y):
            y1, y2 = torch.chunk(y, 2, dim=-1)
            y1, y2 = self.reversible_block.backward(y1, y2)
            return torch.cat((y1, y2), dim=-1)

    class AdiabaticTransformer(nn.Module):
        def __init__(self, num_layers, model_dim):
            super(AdiabaticTransformer, self).__init__()
            self.layers = nn.ModuleList([
                AdiabaticTransformerLayer(model_dim) for _ in range(num_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

        def backward(self, y):
            for layer in reversed(self.layers):
                y = layer.backward(y)
            return y

    class CompleteModel(nn.Module):
        def __init__(self, model_dim, num_layers):
            super(CompleteModel, self).__init__()
            self.lidar_linear = nn.Linear(3, model_dim * 2)
            self.image_linear = nn.Linear(3, model_dim * 2)
            self.transformer = AdiabaticTransformer(num_layers, model_dim * 2)
            self.output_linear = nn.Linear(model_dim * 2, 3)

        def forward(self, lidar, image):
            lidar_embedded = self.lidar_linear(lidar)
            image_embedded = self.image_linear(image)
            transformer_output = self.transformer(lidar_embedded + image_embedded)
            output = self.output_linear(transformer_output)
            return output

