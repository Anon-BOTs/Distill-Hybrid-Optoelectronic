import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from mmdet.models import HEADS, build_loss
from mamba_ssm.models.mixer_seq_simple import create_block


def get_hilbert_index_3d_mamba_lite(template, coors, batch_size, z_dim, hilbert_spatial_size, shift=(0, 0, 0)):
    '''
    coors: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    # new 3D
    hil_size_z, hil_size_y, hil_size_x = hilbert_spatial_size

    x = coors[:, 3] + shift[2]
    y = coors[:, 2] + shift[1]
    z = coors[:, 1] + shift[0]

    flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])
        # inds_next_to_curt[name] = torch.argsort(inds_curt_to_next[name])

    index_info = {}
    index_info['inds_curt_to_next'] = inds_curt_to_next
    index_info['inds_next_to_curt'] = inds_next_to_curt

    return index_info



def get_hilbert_index_2d_mamba_lite(template, coors, batch_size, hilbert_spatial_size, shift=(0, 0)):
    '''
    coors: (b, z, y, x)
    shift: (shift_z, shift_y, shift_x)
    hilbert_spatial_size: [z, y, x]
    '''
    # new 3D
    _, hil_size_y, hil_size_x = hilbert_spatial_size

    x = coors[:, 3] + shift[1]
    y = coors[:, 2] + shift[0]
    # z = coors[:, 1] + shift[0]

    # flat_coors = (z * hil_size_y * hil_size_x + y * hil_size_x + x).long()
    flat_coors = (y * hil_size_x + x).long()
    hil_inds = template[flat_coors].long()

    inds_curt_to_next = {}
    inds_next_to_curt = {}
    for i in range(batch_size):
        batch_mask = coors[:, 0] == i
        inds_curt_to_next[i] = torch.argsort(hil_inds[batch_mask])
        inds_next_to_curt[i] = torch.argsort(inds_curt_to_next[i])
        # inds_next_to_curt[name] = torch.argsort(inds_curt_to_next[name])

    index_info = {}
    index_info['inds_curt_to_next'] = inds_curt_to_next
    index_info['inds_next_to_curt'] = inds_next_to_curt

    return index_info

class MambaBlockV1(nn.Module):

    def __init__(self, 
                 d_model, 
                 ssm_cfg, 
                 norm_epsilon, 
                 rms_norm,
                 down_kernel_size,
                 down_stride,
                 downsample_lvl,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 device=None,
                 dtype=torch.float32):
        super().__init__()

        # ssm_cfg = {}
        factory_kwargs = {'device': device, 'dtype':dtype}

        # mamba layer
        mamba_encoder_1 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=0,
            **factory_kwargs,
        )

        mamba_encoder_2 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=1,
            **factory_kwargs,
        )

        self.mamba_encoder_list = nn.ModuleList([mamba_encoder_1, mamba_encoder_2])

        # downsampling operation #
        if len(down_stride):
            self.conv_encoder = nn.ModuleList()
            for idx in range(len(down_stride)):
                self.conv_encoder.append(None)

            # upsampling operation #
            downsample_times = len(down_stride[1:])
            self.conv_decoder = nn.ModuleList()
            self.conv_decoder_norm = nn.ModuleList()
            
        self.downsample_lvl = downsample_lvl

        norm_cls = partial(
            nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        )
        self.norm = norm_cls(d_model)
        self.norm_back = norm_cls(d_model)

    def forward(
        self,
        bev_features,
        curve_template,
        hilbert_spatial_size,
        num_stage,
        ):
        import pdb
        pdb.set_trace()

        mamba_layer1 = self.mamba_encoder_list[0]
        mamba_layer2 = self.mamba_encoder_list[1]
        
        features = []
        for conv in self.conv_encoder:
            x = conv(bev_features)
            features.append(x)
        
        feats_s1 = features[0]
        feats_s2 = features[1]
        batch_size = len(feats_d)

        clvl_cruve_template_s1 = curve_template['curve_template_rank9']
        clvl_hilbert_spatial_size_s1 = hilbert_spatial_size['curve_template_rank9']
        index_info_s1 = get_hilbert_index_2d_mamba_lite(clvl_cruve_template_s1, coords_s1, batch_size, x_s1.spatial_shape[0], \
                                                        clvl_hilbert_spatial_size_s1, shift=(num_stage, num_stage, num_stage))
        inds_curt_to_next_s1 = index_info_s1['inds_curt_to_next']
        inds_next_to_curt_s1 = index_info_s1['inds_next_to_curt']

        clvl_cruve_template_s2 = curve_template[self.downsample_lvl]
        clvl_hilbert_spatial_size_s2 = hilbert_spatial_size[self.downsample_lvl]
        index_info_s2 = get_hilbert_index_2d_mamba_lite(clvl_cruve_template_s2, coords_s2, batch_size, x_s2.spatial_shape[0], 
                                                        clvl_hilbert_spatial_size_s2, shift=(num_stage, num_stage, num_stage))
        inds_curt_to_next_s2 = index_info_s2['inds_curt_to_next']
        inds_next_to_curt_s2 = index_info_s2['inds_next_to_curt']

        new_features = []
        # Low Resolution
        out_feat_s2 = torch.zeros_like(feats_s2)
        out_feat_s1 = torch.zeros_like(feats_s1)

        # forward SSMs
        for i in range(batch_size):
            b_mask_m2 = coords_s2[:, 0] == i
            feat_m2 = feats_s2[b_mask_m2][inds_curt_to_next_s2[i]][None]
            out_feat_m2 = mamba_layer1(feat_m2, None)
            out_feat_s2[b_mask_m2] = (out_feat_m2[0]).squeeze(0)[inds_next_to_curt_s2[i]]

        x_s2 = self.norm(out_feat_s2)

        # Backward SSMs
        for i in range(batch_size):
            b_mask_m1 = coords_s1[:, 0] == i
            feat_m1 = feats_s1[b_mask_m1][inds_curt_to_next_s1[i]][None]
            feat_back = feat_m1.flip(1)
            out_feat_back = mamba_layer2(feat_back, None)
            out_feat_s1[b_mask_m1] = (out_feat_back[0]).squeeze(0).flip(0)[inds_next_to_curt_s1[i]]

        x_s1 = self.norm_back(out_feat_s1)

        # new_features.append(features[0])
        new_features.append(x_s1)
        new_features.append(x_s2)

        x = x_s2
        return x
    

@HEADS.register_module()
class MambaPredHead(nn.Module):
    def __init__(self,
                curve_template_path_rank9='../data/hilbert/curve_template_3d_rank_9.pth',
                curve_template_path_rank8='../data/hilbert/curve_template_3d_rank_8.pth',
                curve_template_path_rank7='../data/hilbert/curve_template_3d_rank_7.pth',

                # for mamba
                d_model=128,
                fused_add_norm=True,
                rms_norm=True,
                norm_epsilon=0.00001,
                num_blocks=6,
                use_shift=False,
                downsample_lvl=['curve_template_rank9', 'curve_template_rank8', 'curve_template_rank7'],
                **kwargs,
                ):
        super().__init__(init_cfg)

        self.fuse_block = FuseBlock(in_channels, out_channels)
        blocks = []
        for i in range(num_blocks):
            blocks.append(MambaBlockV1(d_model, None, norm_epsilon, rms_norm,
                                       down_kernel_size[i], down_stride[i], device='cuda',
                                       downsample_lvl=self.downsample_lvl[i],))
        self.block_list = nn.ModuleList(blocks)

        # Build Hilbert tempalte 
        self.curve_template = {}
        self.hilbert_spatial_size = {}
        self.load_template(self.model_cfg.INPUT_LAYER.curve_template_path_rank9, 9)
        self.load_template(self.model_cfg.INPUT_LAYER.curve_template_path_rank8, 8)
        self.load_template(self.model_cfg.INPUT_LAYER.curve_template_path_rank7, 7)

        self.use_shift = use_shift

    def load_template(self, path, rank):
        template = torch.load(path)
        if isinstance(template, dict):
            self.curve_template[f'curve_template_rank{rank}'] = template['data'].reshape(-1)
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = template['size'] 
        else:
            self.curve_template[f'curve_template_rank{rank}'] = template.reshape(-1)
            spatial_size = 2 ** rank
            self.hilbert_spatial_size[f'curve_template_rank{rank}'] = (1, spatial_size, spatial_size) #[z, y, x]

    def forward(self, bev_features, frame_index, img_metas):
        bev_feat = self.fuse_block(bev_features)
        if self.use_shift:
            num_stage = 2
        else:
            num_stage = 0

        for i, block in enumerate(self.block_list):
            bev_feat = block(bev_features, self.curve_template, self.hilbert_spatial_size, num_stage)
        return bev_feat