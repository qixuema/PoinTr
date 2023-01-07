from tokenize import group
import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
# knn = KNN(k=16, transpose_mode=False)


def knn_point_dg(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C], C = 3
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance_dg(new_xyz, xyz) # 从 xyz 中找距离 new_xyz 最近的 16 个点的索引
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance_dg(src, dst):
    """
    Calculate Euclid distance between each two points. 这里计算的都是三维欧式空间中的距离，暂时没涉及到特征空间中的计算
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
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


class DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        # NOTE pytorch 中只能对倒数第2维数据进行卷积
        self.input_trans = nn.Conv1d(3, 8, 1) # in_channels=3, out_channels=8, kernel_size=1， 将 channel 3 变为 channel 8，其他不变

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    
    @staticmethod # 以上实例声明了静态方法 f，从而可以实现实例化使用 C().f()，当然也可以不实例化调用该方法 C.f()
    def fps_downsample(coor, f, num_group): # 我这里用 coor 表示三维空间坐标，用 f 表示特征
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group) # [bs, num_group]

        combined_f = torch.cat([coor, f], dim=1) # [bs, C_coor + C_f, np], 其中 C_coor=3

        new_combined_f = (
            pointnet2_utils.gather_operation(
                combined_f, fps_idx
            )
        ) # [bs, C_coor + C_f, num_group]
        # 然后，从 new_combined_x 中将 coor 和 f 拆分开
        new_coor = new_combined_f[:, :3] # [bs, C_coor, num_group]
        new_f = new_combined_f[:, 3:] # [bs, C_f, num_group]

        return new_coor, new_f

    @staticmethod
    def get_graph_feature_dg(coor_q, f_q, coor_k, f_k): # 根据 coor_k，从 f_k 中找到距离 f_q 的 coor_q 最近的 16 个 feature，计算相对特征，然后将相对特征与绝对特征进行合并，然后输出

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = f_k.size(0)
        num_points_k = f_k.size(2) # 2048, 2048, 512, 512
        num_points_q = f_q.size(2) # 2048, 512,  512, 128

        with torch.no_grad():
#             _, idx = knn(coor_k, coor_q)  # bs k np
            idx = knn_point_dg(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M [bs, 2048/512/512/128, 16]
            idx = idx.transpose(-1, -2).contiguous() # [bs, 16, 2048/512/512/128]
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=f_q.device).view(-1, 1, 1) * num_points_k # 原来这个是为了区分开不同 batch 的索引
            idx = idx + idx_base # [bs, 16, 2048/512/512/128]
            idx = idx.view(-1)
        num_dims_k = f_k.size(1) # 8/32/64/64
        f_k = f_k.transpose(2, 1).contiguous() # [bs, 8/32/64/64, np_k]->[bs, np_k, 8/32/64/64]
        feature = f_k.view(batch_size * num_points_k, -1)[idx, :] # 这样方便直接摘取 k 近邻的 feature
        feature = feature.view(batch_size, k, num_points_q, num_dims_k).permute(0, 3, 2, 1).contiguous() # [bs, C, np_q, k] 即 [bs, 8/32/64/64, 2048/512/512/128, 16]
        f_q = f_q.view(batch_size, num_dims_k, num_points_q, 1).expand(-1, -1, -1, k) # [bs, C, np_q, 1] 即 [bs, 8/32/64/64, 2048/512/512/128, 1]
        
        # 整合全局和局部相对特征
        feature = torch.cat((feature - f_q, f_q), dim=1) # [bs, 2*C, np_q, 16] 即 [bs, 16/64/128/128, 2048/512/512/128, 16]
        return feature

    def forward(self, x):

        # x: bs, 3, np(2048)

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x # [bs, 3, 2048]
        f = self.input_trans(x) # [bs, 8, np], 即将 channel 3 变为 channel 8

        # first layer, no fps_downsample, knn 中的 k = 16
        f = self.get_graph_feature_dg(coor, f, coor, f) # dg means dgcnn version; [bs, 2*8, np_q=2048, k=16]
        # 接下来一个模块就是卷积+批标准化+激活+最大化
        f = self.layer1(f) # [bs, 32, 2048, 16]
        f = f.max(dim=-1, keepdim=False)[0] # TODO: 这里显式地取了第一个 batch，不晓得是为什么？虽然我在测试的时候，bs 确实=1

        # second layer
        coor_q, f_q = self.fps_downsample(coor, f, 512) # 从 incomplete point cloud 中 get 512 个中心点，以及这些中心点的 feature
        f = self.get_graph_feature_dg(coor_q, f_q, coor, f) # [bs, 64, 512, 16]
        f = self.layer2(f) # [bs, 64, 512, 16]
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q # [bs, 3, 512]

        # third layer, no fps_downsample
        f = self.get_graph_feature_dg(coor, f, coor, f) # [bs, 128, 512, 16]
        f = self.layer3(f) # [bs, 64, 512, 16]
        f = f.max(dim=-1, keepdim=False)[0]

        # fourth layer
        coor_q, f_q = self.fps_downsample(coor, f, 128) # 从 coor 和 f 中 get 128 个中心点，以及这些中心点的 feature; [bs, 3, 128], [bs, 64, 128]
        f = self.get_graph_feature_dg(coor_q, f_q, coor, f) # [bs, 128, 128, 16]
        f = self.layer4(f) # [bs, 128, 128, 16]
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q # [bs, 3, 128]

        return coor, f # 得到了 N 个局部区域的中心点及其 feature; [bs, 3, 128], [bs, 128, 128]
