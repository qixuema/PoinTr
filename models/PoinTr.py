import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from .Transformer import PCTransformer
from .build import MODELS


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc


class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512): # in_channel=384, step=8, hidden_dim=256
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1) # [1, 64]
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1) # 也是 [1, 64], 不过和 a 的排列方式不一样
        self.folding_seed = torch.cat([a, b], dim=0).cuda() # [2, 64]

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x): # x: [bs_in*224, 384]
        num_sample = self.step * self.step
        bs = x.size(0) # bs = bs_in*224
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample) # [bs, 384, 1] expand [bs, 384, 64]
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device) # [1, 2, 64] expand [bs, 2, 64]

        x = torch.cat([seed, features], dim=1) # [bs, 386, 64]
        fd1 = self.folding1(x)  # [bs, 3, 64]
        x = torch.cat([fd1, features], dim=1)  # [bs, 387, 64]
        fd2 = self.folding2(x)  # [bs, 3, 64]

        return fd2

@MODELS.register_module()
class PoinTr(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim # 384
        self.knn_layer = config.knn_layer # 1，也就是只用了一层的几何感知
        self.num_pred = config.num_pred # 14336
        self.num_query = config.num_query # for PCN_model, num_query = 224

        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5) # 向下取整，为 8
        self.base_model = PCTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [6, 8], drop_rate = 0., num_query = self.num_query, knn_layer = self.knn_layer)
        
        self.foldingnet = Fold(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()


    def get_loss(self, ret, gt, epoch=0):
        loss_coarse = self.loss_func(ret[0], gt) # 计算"中心点"与 gt 的 loss；监督预测中心点的目的是为了让其符合完整真实点云的分布：
        loss_fine = self.loss_func(ret[1], gt) # 计算"预测点云"与 gt 的 loss
        return loss_coarse, loss_fine

    def forward(self, xyz):
        # 这里的 base_model 就是 PCTransformer
        q, coarse_point_cloud = self.base_model(xyz) # B M C and B M 3 ; [bs, 224, 384], [bs, 224, 3]
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024, 为了提取 global feature，我们先将 q 升维到1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024，然后使用 max pooling 得到 global feature

        # 将 q 的 global feature、q 自身以及 coarse_point_cloud 进行合并
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C [bs, 224, 1411=1024+384+3]

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C [bs*224, 384]
        # # NOTE: try to rebuild pc
        # coarse_point_cloud = self.refine_coarse(rebuild_feature).reshape(B, M, 3)

        # NOTE: foldingNet
        # 将上述合并特征输入 FoldingNet 预测相对位置，每个中心点对应预测 64 个邻近点；TODO: 另外 foldingnet 真的有这么神奇吗？这部分是单独训练吗？
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S [bs, 224, 3, 64] 这里的作用是，为每一个预测的中心点，生成64个周围点
        # 将相对位置变成绝对位置，即得到预测的缺失部分的点云
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3 ; [bs, 224*64, 3]

        # NOTE: fc
        # relative_xyz = self.refine(rebuild_feature)  # BM 3S
        # rebuild_points = (relative_xyz.reshape(B,M,3,-1) + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)

        # cat the input
        # 原始中心点
        inp_sparse = fps(xyz, self.num_query)
        # 将原始中心点与预测的中心点合并在一起，得到整个物体的中心点
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        # 将预测的残缺部分点云和输入的点云合并在一起
        rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()

        ret = (coarse_point_cloud, rebuild_points) # 输出中心点和全部点
        return ret

