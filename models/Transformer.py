import torch
import torch.nn as nn

from timm.models.layers import DropPath,trunc_normal_

from .dgcnn_group import DGCNN_Grouper
from utils.logger import *
import numpy as np
# from knn_cuda import KNN
# knn = KNN(k=8, transpose_mode=False)

def knn_point(nsample, xyz, query_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        query_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(query_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    计算两组点集中, 所有点, 两两之间的距离平方;
    Calculate Euclid distance between each two points.
    because: 
        dist = (xs - xd)^2 + (ys - yd)^2 + (zs - zd)^2
    and:
        src^T * dst = xs * xd + ys * yd + zs * zd;
        sum(src**2, dim=-1) = xs*xs + ys*ys + zs*zs;
        sum(dst**2, dim=-1) = xd*xd + yd*yd + zd*zd;
    therefore:
        dist = sum(src**2,dim=-1) + sum(dst**2,dim=-1) - 2*src^T*dst
    
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist   

def get_knn_index(coor_q, coor_k=None): # 该函数被调用了 3 次，encoder 之前调用了一次，decoder 之前调用了 2 次
    """从 coor_k 中获取 coor_q 的 k 个近邻的索引, 默认 k = 8; 作者自己手写了一个 knn 算法

    Args:
        coor_q (torch.tensor([bs, 3, np_q])): query points
        coor_k (torch.tensor([bs, 3, np_k]), optional): 被 query 的 points. Defaults to None.

    Returns:
        torch.tensor([bs*k*np_q]): 返回 query points 的索引, 并展开为 1 维的形式
    """
    coor_k = coor_k if coor_k is not None else coor_q #[1, 3, 128]
    # coor: bs, 3, np
    batch_size, _, num_points_q = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        # _, idx = knn(coor_k, coor_q)  # bs k np
        k = 8
        idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M [bs, 128, 8]
        idx = idx.transpose(-1, -2).contiguous() # [bs, 8, 128]
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k # 这里为什么要加一个 idx_base, 也就是为什么要把第 i 个 batch 的点的 idx 加上 i* num_points_k 呢？
        idx = idx + idx_base
        idx = idx.view(-1) # 这里又为什么把所有 batch 的 idx 展开成 1 行呢？
    
    return idx  # bs*k*np

def get_graph_feature(x, knn_index, x_q=None):
    """合并相对位置特征与绝对位置特征

    Args:
        x (torch.tensor([bs, np, c])): 绝对位置特征
        knn_index (torch.tensor([bs*k*np])): knn 索引
        x_q (_type_, optional): _description_. Defaults to None.

    Returns:
        torch.tensor([bs, k, np, c]): "相对位置特征"和"绝对位置特征"合并之后的特征
    """
    # x: bs, np, c 
    # knn_index: bs*k*np
    k = 8
    batch_size, num_points, num_dims = x.size() # [bs, 128, 384]
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :] # 这样方便直接摘取 k 近邻的 feature
    feature = feature.view(batch_size, k, num_query, num_dims) # [1, 8, 128, 384]
    x = x_q if x_q is not None else x
    x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1) # [1, 8, 128, 384]
    feature = torch.cat((feature - x, x), dim=-1) # feature - x 为相对位置特征, x 为绝对位置特征
    return feature  # b k np c [bs, 8, 128, 768=2*384]

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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim*2, dim)

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)
        
        q = q + self.drop_path(q_1) # [bs, 224, 384]

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None: # 这个是解码器独有的 cross_knn
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q) # 首先根据 cross_knn_index，从 norm_v 中找到对应的 feature，然后用这个 feature 与 norm_q 进行相减，得到相对feature，然后把相对feature与norm_q进行合并，就得到了 knn_f
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q

# transformer block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index = None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)

        if knn_index is not None: # 第一层几何感知
            knn_f = get_graph_feature(norm_x, knn_index) # 实际上就是合并相对位置和绝对位置特征（局部和全局）[bs, 8, 128, 768]
            knn_f = self.knn_map(knn_f) # 线性映射 [bs, 8, 128, 384]
            knn_f = knn_f.max(dim=1, keepdim=False)[0] # [bs, 128, 384]
            x_1 = torch.cat([x_1, knn_f], dim=-1) # 将 attention 模块的输出与几何感知模块的输出进行合并 [bs, 128, 768]
            x_1 = self.merge_map(x_1) # 再映射回原来的维度 [bs, 128, 384]
        # 然后是 residual 模块 
        x = x + self.drop_path(x_1) # [bs, 128, 384]
        x = x + self.drop_path(self.mlp(self.norm2(x))) # [bs, 128, 384]
        return x

class PCTransformer(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 6], num_heads=6, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        num_query = 224, knn_layer = -1):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        
        self.knn_layer = knn_layer

        print_log(' Transformer with knn_layer %d' % self.knn_layer, logger='MODEL')

        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )
        # self.pos_embed_wave = nn.Sequential(
        #     nn.Conv1d(60, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(128, embed_dim, 1)
        # )

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])

        # self.increase_dim = nn.Sequential(
        #     nn.Linear(embed_dim,1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024)
        # )

        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.num_query = num_query # 这个在 PCN model 里默认是 224
        
        # 用来预测输入到解码器的初始位置序列, 对应结构图中的 Query Generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_query)
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor): # 这里是作者尝试的工作，在最终版本并没有使用
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

    def forward(self, in_pc):
        '''
            in_pc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        # build point proxy
        bs = in_pc.size(0)
        coor_center_input, f = self.grouper(in_pc.transpose(1,2).contiguous()) # 分别得到中心点坐标及中心点特征; [bs, C_coor=3, num=128], [bs, C_f=128, num=128]
        knn_index = get_knn_index(coor_center_input) # NOTE: 这里 knn_index 的 shape is [bs*8*num_q], 是一个 1 维 tensor, knn 是为了实现几何感知功能
        # NOTE: try to use a sin wave  coor B 3 N, change the pos_embed input dim
        # pos = self.pos_encoding_sin_wave(coor).transpose(1,2)
        pos =  self.pos_embed(coor_center_input).transpose(1,2) # 通过 MLP 对残缺点云的中心点位置进行编码 [bs, 128, 384] TODO: 这里以后有时间得研究一下他这个位置编码具体是怎么实现的
        x = self.input_proj(f).transpose(1,2) # 将特征通过一个 MLP, 得到新的特征 x; [bs, 128, 384], 即 feature 变成了 384 维
        # cls_pos = self.cls_pos.expand(bs, -1, -1)
        # cls_token = self.cls_pos.expand(bs, -1, -1)
        # x = torch.cat([cls_token, x], dim=1)
        # pos = torch.cat([cls_pos, pos], dim=1)
        
        # encoder
        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer: # 作者默认 self.knn_layer = 1, 所以关于 knn 的网络层，只有一层
                x = blk(x + pos, knn_index)   # B N C [bs, 128, 384]
            else:
                x = blk(x + pos) # [bs, 128, 384]；这里最后得到的 x 将作为 decoder 里面 Q、K、V 中的 V
        # build the query feature for decoder
        # global_feature  = x[:, 0] # B C

        global_feature = self.increase_dim(x.transpose(1,2)) # B 1024 N ，因为要得到的是 global feature，所以维度越大越好，因此先进行升维 [bs, 1024, 128]
        global_feature = torch.max(global_feature, dim=-1)[0] # B 1024 [bs, 1024], 提取输出的残缺点云的所有中心点的全局特征

        # 1. Query Generator 模块
        # 1.1 预测序列的位置特征, 这个也就是预测的缺失部分点云的中心点坐标了
        coarse_point_cloud = self.coarse_pred(global_feature).reshape(bs, -1, 3)  #  B M C(3) [bs, 224, 3]；根据 global feature 得到 指定个数(224)的 query point 坐标

        # 1.2 将全局特征和 query point 的位置特征合并, 然后 mlp 到 384 维度
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query, -1), 
            coarse_point_cloud], dim=-1) # B M C+3 [bs, 224, 1027=1024+3]
        q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C ，然后再经过一个 mlp，得到 224 个 query point 的 384 维度特征 [bs, 224, 384]
        
        # 用于解码器第一层几何感知
        new_knn_index = get_knn_index(coarse_point_cloud.transpose(1, 2).contiguous()) # 找自身的 k 近邻索引
        cross_knn_index = get_knn_index(coor_k=coor_center_input, coor_q=coarse_point_cloud.transpose(1, 2).contiguous()) # 从残缺点云的中心点集中找 k 近邻索引，所以叫 cross knn
        
        # decoder
        for i, blk in enumerate(self.decoder):
            if i < self.knn_layer:
                q = blk(q, x, new_knn_index, cross_knn_index)   # B M C
            else:
                q = blk(q, x)

        return q, coarse_point_cloud # 这部分也要输出; [bs, 224, 384], [bs, 224, 3]

