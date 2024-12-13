# FreeNet
# Z. Zheng, Y. Zhong, A. Ma, and L. Zhang, “FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End
# Hyperspectral Image Classification,” IEEE Trans. Geosci. Remote Sens., vol. 58, no. 8, pp. 5612–5626, 2020.

# Reference code: https://github.com/Z-Zheng/FreeNet

import torch
from torch import nn
import torch.nn.functional as F
from feature_fusion import FeatureFusionBlock

def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
                         nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                         nn.GroupNorm(num_group, out_channel),
                         nn.ReLU(inplace=False),
                        )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
                         nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                         nn.ReLU(inplace=False)
                        )


def repeat_block(block_channel, r, n):
    layers = [
        nn.Sequential(
            conv3x3_gn_relu(block_channel, block_channel, r)
        )
        for _ in range(n)]
    return nn.Sequential(*layers)
    # 循环n遍


''' ---------------------------------------------------------------------------------------------------------------- '''
# FreeNet 模块化


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        r = int(16 * self.config['reduction_ratio'])
        block1_channels = int(self.config['block_channels'][0] * self.config['reduction_ratio'] / r) * r
        block2_channels = int(self.config['block_channels'][1] * self.config['reduction_ratio'] / r) * r
        block3_channels = int(self.config['block_channels'][2] * self.config['reduction_ratio'] / r) * r
        block4_channels = int(self.config['block_channels'][3] * self.config['reduction_ratio'] / r) * r

        self.Module_dict = nn.ModuleDict({
            '0': nn.ModuleList([
                conv3x3_gn_relu(self.config['in_channels'], block1_channels, r),
                repeat_block(block1_channels, r, self.config['num_blocks'][0]),
                nn.Identity(),
            ]),

            '1': nn.ModuleList([
                downsample2x(block1_channels, block2_channels),
                repeat_block(block2_channels, r, self.config['num_blocks'][1]),
                nn.Identity(),
            ]),

            '2': nn.ModuleList([
                downsample2x(block2_channels, block3_channels),
                repeat_block(block3_channels, r, self.config['num_blocks'][2]),
                nn.Identity(),
            ]),

            '3': nn.ModuleList([
                downsample2x(block3_channels, block4_channels),
                repeat_block(block4_channels, r, self.config['num_blocks'][3]),
                nn.Identity(),
            ])
        })

    def forward(self, data, index):

        feature_ops = self.Module_dict[str(index)]

        for op in feature_ops:
            data = op(data)

        return data


class Reduce_Conv(nn.Module):
    def __init__(self, config):
        super(Reduce_Conv, self).__init__()
        self.config = config
        r = int(16 * self.config['reduction_ratio'])
        block1_channels = int(self.config['block_channels'][0] * self.config['reduction_ratio'] / r) * r
        block2_channels = int(self.config['block_channels'][1] * self.config['reduction_ratio'] / r) * r
        block3_channels = int(self.config['block_channels'][2] * self.config['reduction_ratio'] / r) * r
        block4_channels = int(self.config['block_channels'][3] * self.config['reduction_ratio'] / r) * r

        inner_dim = int(self.config['inner_dim'] * self.config['reduction_ratio'])
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])

    def forward(self, feat_list):
        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        return inner_feat_list


def top_down(top):
    top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
    return top2x


class BranchFusion(nn.Module):
    def __init__(self, channel, reduction):
        super(BranchFusion, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.inter_channels = channel // reduction
        self.conv_data = nn.Conv2d(in_channels=channel,
                                   out_channels=channel // reduction,
                                   kernel_size=(1, 1),
                                   stride=(1, 1))
        self.conv_data_spec = nn.Conv2d(in_channels=channel,
                                        out_channels=channel // reduction,
                                        kernel_size=(1, 1),
                                        stride=(1, 1))
        self.conv_quary = nn.Conv2d(in_channels=channel,
                                    out_channels=channel // reduction,
                                    kernel_size=(1, 1),
                                    stride=(1, 1))
        self.attend = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels=channel // reduction,
                                  out_channels=channel,
                                  kernel_size=(1, 1),
                                  stride=(1, 1))

    def forward(self, data, data_spec, feat):
        # B, C, H, W = data.shape
        data_conv = self.conv_data(data).permute(0, 2, 3, 1).contiguous()   # (B, C, H, W) -> (B, Q, H, W) -> (B, H, W, Q)
        data_spec_conv = self.conv_data_spec(data_spec).permute(0, 2, 3, 1).contiguous()   # (B, C, H, W) -> (B, Q, H, W) -> (B, H, W, Q)
        data_data_spec_concat = torch.cat((data_conv.unsqueeze(-1), data_spec_conv.unsqueeze(-1)), dim=-1)   # (B, H, W, Q, 2)
        quary = feat.clone()
        quary_conv = self.conv_quary(quary).permute(0, 2, 3, 1).contiguous().unsqueeze(-1)   # (B, C, H, W) -> (B, Q, H, W) -> (B, H, W, Q) -> (B, H, W, Q, 1)
        PixelWiseWeight = torch.einsum('bhwcn,bhwcm->bhwnm', [quary_conv, data_data_spec_concat])   # (B, H, W, 1, 2)
        scale = self.inter_channels ** 0.5
        PixelWiseWeight = PixelWiseWeight / scale
        PixelWiseWeight = self.attend(PixelWiseWeight)
        return PixelWiseWeight


class Fus(nn.Module):
    def __init__(self, config):
        super(Fus, self).__init__()
        self.config = config
        inner_dim = int(self.config['inner_dim'] * self.config['reduction_ratio'])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.FusionWeight = nn.Sequential(
            BranchFusion(channel=128, reduction=4),
            BranchFusion(channel=128, reduction=4),
            BranchFusion(channel=128, reduction=4),
            BranchFusion(channel=128, reduction=4),
        )

    def forward(self, inner_feat_list_data, inner_feat_list_data_spec):

        feat = inner_feat_list_data[0] + inner_feat_list_data_spec[0]
        PixelWiseWeight = self.FusionWeight[0](data=inner_feat_list_data[0],
                                               data_spec=inner_feat_list_data_spec[0],
                                               feat=feat)

        data_feat_weight = PixelWiseWeight[..., 0].permute(0, 3, 1, 2).contiguous()
        data_spec_feat_weight = PixelWiseWeight[..., 1].permute(0, 3, 1, 2).contiguous()
        inner_feat = inner_feat_list_data[0] * data_feat_weight + \
                     inner_feat_list_data_spec[0] * data_spec_feat_weight

        out_feat_list = [self.fuse_3x3convs[0](inner_feat)]

        for i in range(len(inner_feat_list_data) - 1):
            feat = top_down(out_feat_list[i])
            PixelWiseWeight = self.FusionWeight[0](data=inner_feat_list_data[i + 1],
                                                   data_spec=inner_feat_list_data_spec[i + 1],
                                                   feat=feat)

            data_feat_weight = PixelWiseWeight[..., 0].permute(0, 3, 1, 2).contiguous()
            data_spec_feat_weight = PixelWiseWeight[..., 1].permute(0, 3, 1, 2).contiguous()
            inner_feat = inner_feat_list_data[i + 1] * data_feat_weight + \
                         inner_feat_list_data_spec[i + 1] * data_spec_feat_weight

            inner = feat + inner_feat
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        return out_feat_list


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        inner_dim = int(self.config['inner_dim'] * self.config['reduction_ratio'])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])

    def forward(self, inner_feat_list):
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = top_down(out_feat_list[i]) + inner_feat_list[i + 1]
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        return out_feat_list


''' ---------------------------------------------------------------------------------------------------------------- '''
# Loss

class FeatureMap(nn.Module):
    def __init__(self):
        super(FeatureMap, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, data, label):

        B, H, W = label.shape
        data = F.interpolate(data, (H, W), mode='nearest')

        not_ignore_spatial_mask = label.int() != -1
        label = label * not_ignore_spatial_mask

        one_hot_label = torch.nn.functional.one_hot(label, num_classes=24)  # [B, H, W, class]
        one_hot_label = one_hot_label * not_ignore_spatial_mask.unsqueeze(-1)
        label_map_class = one_hot_label.permute(0, 3, 1, 2).contiguous()
        label_map_class = label_map_class.to(torch.float32)

        feature = torch.einsum('bchw,bnhw->bcn', [data, label_map_class])
        label_num_sum = label_map_class.sum(dim=-1).sum(dim=-1).unsqueeze(1)
        feature = feature / (label_num_sum + 1e-8)
        label_map = self.avg_pool(label_map_class).squeeze(-1).squeeze(-1)
        label_map[label_map > 0] = 1
        mask = label_map

        return feature, mask


''' ---------------------------------------------------------------------------------------------------------------- '''


class CSCN(nn.Module):
    def __init__(self, config):
        super(CSCN, self).__init__()
        self.config = config
        self.Encoder = Encoder(self.config)
        self.Reduce_Conv = Reduce_Conv(self.config)
        self.Fus = Fus(self.config)
        self.cls_pred_conv = nn.Conv2d(128, self.config['num_classes'], 1)
        self.Encoder_spec = Encoder(self.config)
        self.Reduce_Conv_spec = Reduce_Conv(self.config)
        r = int(16 * self.config['reduction_ratio'])
        block1_channels = int(self.config['block_channels'][0] * self.config['reduction_ratio'] / r) * r
        block2_channels = int(self.config['block_channels'][1] * self.config['reduction_ratio'] / r) * r
        block3_channels = int(self.config['block_channels'][2] * self.config['reduction_ratio'] / r) * r
        block4_channels = int(self.config['block_channels'][3] * self.config['reduction_ratio'] / r) * r
        self.uff_data = nn.Sequential(
            FeatureFusionBlock(xyz_dim=block1_channels, rgb_dim=block1_channels),
            FeatureFusionBlock(xyz_dim=block2_channels, rgb_dim=block2_channels),
            FeatureFusionBlock(xyz_dim=block3_channels, rgb_dim=block3_channels),
            FeatureFusionBlock(xyz_dim=block4_channels, rgb_dim=128),
        )
        self.uff_data_spec = nn.Sequential(
            FeatureFusionBlock(xyz_dim=block1_channels, rgb_dim=block1_channels),
            FeatureFusionBlock(xyz_dim=block2_channels, rgb_dim=block2_channels),
            FeatureFusionBlock(xyz_dim=block3_channels, rgb_dim=block3_channels),
            FeatureFusionBlock(xyz_dim=block4_channels, rgb_dim=128),
        )
        self.feature = FeatureMap()
        self.Decoder = Decoder(self.config)
        self.Decoder_spec = Decoder(self.config)
        self.cls_weak = nn.Conv2d(128, self.config['num_classes'], 1)
        self.cls_weak_spec = nn.Conv2d(128, self.config['num_classes'], 1)
        self.Softmax = nn.Softmax(dim=1)
        self.Sigmoid = nn.Sigmoid()
        self.NLLLoss = nn.NLLLoss(ignore_index=-1)
        self.BCELoss = nn.BCELoss(reduction='none')

    def forward(self, data, data_spec, label, training=True):
        feat_list_data = []
        feat_list_data_spec = []
        for i in range(4):
            data = self.Encoder(data=data, index=i)
            data_spec = self.Encoder_spec(data=data_spec, index=i)

            feat_list_data.append(data)
            feat_list_data_spec.append(data_spec)

        inner_feat_list_data = self.Reduce_Conv(feat_list_data)
        inner_feat_list_data_spec = self.Reduce_Conv_spec(feat_list_data_spec)

        # ------------------------------------------------------------------------------------------------------------ #

        inner_feat_list_data.reverse()
        inner_feat_list_data_spec.reverse()

        out_feat_list = self.Fus(inner_feat_list_data, inner_feat_list_data_spec)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        loss_uff = torch.tensor(0.0, requires_grad=True, device=data.device)
        loss_con = torch.tensor(0.0, requires_grad=True, device=data.device)
        if training:

            out_feat_list_data = self.Decoder(inner_feat_list_data)
            out_feat_list_data_spec = self.Decoder_spec(inner_feat_list_data_spec)

            logits_data = self.cls_weak(out_feat_list_data[-1])
            logits_data_spec = self.cls_weak_spec(out_feat_list_data_spec[-1])

            ''' UFF '''
            data_uff_0 = out_feat_list_data[3]
            data_spec_uff_0 = out_feat_list_data_spec[3]
            data_uff_3 = feat_list_data[3]
            data_spec_uff_3 = feat_list_data_spec[3]

            feature_data_uff_0, mask = self.feature(data_uff_0, label)
            feature_data_spec_uff_0, _ = self.feature(data_spec_uff_0, label)
            feature_data_uff_3, _ = self.feature(data_uff_3, label)
            feature_data_spec_uff_3, _ = self.feature(data_spec_uff_3, label)

            feature_data_uff_0, feature_data_spec_uff_0, feature_data_uff_3, feature_data_spec_uff_3 = \
                feature_data_uff_0.permute(0, 2, 1).contiguous(), feature_data_spec_uff_0.permute(0, 2, 1).contiguous(), \
                feature_data_uff_3.permute(0, 2, 1).contiguous(), feature_data_spec_uff_3.permute(0, 2, 1).contiguous()

            loss_con = self.uff_data[3](feature_data_uff_3, feature_data_uff_0, mask) + \
                       self.uff_data_spec[3](feature_data_spec_uff_3, feature_data_spec_uff_0, mask) + \
                       loss_con

            pro_data = self.Softmax(logits_data)
            pro_data_spec = self.Softmax(logits_data_spec)
            pro = torch.cat((pro_data.unsqueeze(-1), pro_data_spec.unsqueeze(-1)), dim=-1).max(-1)[0]
            pro_log = torch.log(pro)
            loss_decoder = self.NLLLoss(pro_log, label)
            loss_uff = loss_con + loss_decoder

        return logit, loss_uff