import torch
import torch.nn as nn
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeatureFusionBlock(nn.Module):
    def __init__(self, xyz_dim, rgb_dim, mlp_ratio=4.):
        super().__init__()

        self.xyz_dim = xyz_dim
        self.rgb_dim = rgb_dim

        self.xyz_norm = nn.LayerNorm(xyz_dim)
        self.xyz_mlp = Mlp(in_features=xyz_dim, hidden_features=int(xyz_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)

        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_mlp = Mlp(in_features=rgb_dim, hidden_features=int(rgb_dim * mlp_ratio), act_layer=nn.GELU, drop=0.)

        self.rgb_head = nn.Linear(rgb_dim, 256)
        self.xyz_head = nn.Linear(xyz_dim, 256)

        self.T = 1

    def feature_fusion(self, xyz_feature, rgb_feature):
        xyz_feature = self.xyz_norm(xyz_feature)
        xyz_feature = self.xyz_mlp(xyz_feature)
        rgb_feature = self.rgb_norm(rgb_feature)
        rgb_feature = self.rgb_mlp(rgb_feature)

        feature = torch.cat([xyz_feature, rgb_feature], dim=2)

        return feature

    def contrastive_loss(self, q, k, mask):

        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]
        labels = (torch.arange(N, dtype=torch.long)).to(logits.device)
        labels[mask == 0] = -1
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss_CrossEntropy = criterion(logits, labels)

        return loss_CrossEntropy * self.T

    def forward(self, xyz_feature, rgb_feature, xyz_mask):
        mask = xyz_mask.to(xyz_feature.device)

        feature = self.feature_fusion(xyz_feature, rgb_feature)

        feature_xyz = feature[:, :, :self.xyz_dim]
        feature_rgb = feature[:, :, self.xyz_dim:]

        q = self.rgb_head(feature_rgb.view(-1, feature_rgb.shape[2]))
        k = self.xyz_head(feature_xyz.view(-1, feature_xyz.shape[2]))
        mask_ignore = mask.contiguous().view(-1)
        loss = self.contrastive_loss(q, k, mask_ignore)

        return loss
