# -*-coding:utf-8-*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common.initializer import TruncatedNormal


class SharedMLP(nn.Cell):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            pad_mode='same',
            bn=False,
            activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.Conv2dTranspose if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode='valid',
            has_bias=True,
            weight_init=TruncatedNormal(sigma=1e-3)
        )
        self.has_bn = bn
        self.batch_norm = None
        if self.has_bn:
            self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99)
        self.activation_fn = activation_fn

    def construct(self, x):
        r"""
            construct method

            Parameters
            ----------
            x: ms.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            ms.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(x)
        if self.has_bn:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Cell):
    def __init__(self, in_channel=10, out_channel=1, use_pos_encoding=True):
        super(LocalSpatialEncoding, self).__init__()

        self.mlp = SharedMLP(in_channel, out_channel, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.d = out_channel
        self.use_pos_encoding = use_pos_encoding

    def construct(self, coords, features, neighbor_idx):
        r"""
            construct method

            Parameters
            ----------
            coords: ms.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: ms.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbor_idx: ms.Tensor, shape (B, N, K)

            Returns
            -------
            ms.Tensor, shape (B, 2*d, N, K)
        """

        idx = neighbor_idx
        extended_idx = P.Tile()(idx.expand_dims(1), (1, 3, 1, 1))
        cat = P.Concat(-3)
        if self.use_pos_encoding:
            # finding neighboring points
            xyz_tile = P.Tile()(coords.transpose(0, 2, 1).expand_dims(-1),
                                (1, 1, 1, idx.shape[-1]))
            neighbor_xyz = P.GatherD()(xyz_tile, 2, extended_idx.astype(ms.int32))
            relative_xyz = xyz_tile - neighbor_xyz  # relative_xyz
            relative_dist = P.Sqrt()(P.ReduceSum(keep_dims=True)(P.Square()(relative_xyz), -3))

            # relative point position encoding
            f_xyz = cat((
                relative_dist,
                relative_xyz,
                xyz_tile,
                neighbor_xyz,
            ))
        else:
            f_xyz = coords
        f_xyz = self.mlp(f_xyz)

        f_tile = P.Tile()(features, (1, 1, 1, idx.shape[-1]))
        extended_idx_for_feat = P.Tile()(idx.expand_dims(1), (1, f_xyz.shape[1], 1, 1))
        f_neighbours = P.GatherD()(f_tile, 2, extended_idx_for_feat.astype(ms.int32))

        f_concat = cat([f_xyz, f_neighbours])

        if self.use_pos_encoding:
            return f_xyz, f_concat

        return f_concat


class AttentivePooling(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.SequentialCell([
            nn.Dense(in_channels, in_channels, has_bias=False),
            nn.Softmax(-2)
        ])
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

    def construct(self, x):
        r"""
            construct method

            Parameters
            ----------
            x: ms.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            ms.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

        # sum over the neighbors
        features = scores * x
        features = P.ReduceSum(keep_dims=True)(features, -1)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Cell):
    def __init__(self, d_in, d_out):
        super(LocalFeatureAggregation, self).__init__()

        self.mlp1 = SharedMLP(d_in, d_out // 2, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out, bn=True)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(in_channel=10, out_channel=d_out // 2, use_pos_encoding=True)
        self.lse2 = LocalSpatialEncoding(in_channel=d_out // 2, out_channel=d_out // 2, use_pos_encoding=False)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU(0.2)

    def construct(self, coords, features, neighbor_idx):
        r"""
            construct method

            Parameters
            ----------
            coords: ms.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: ms.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbor_idx: ms.Tensor, shape (B, N, 16)
                knn neighbor idx
            Returns
            -------
            ms.Tensor, shape (B, 2*d_out, N, 1)
        """
        f_pc = self.mlp1(features)

        f_xyz, f_concat = self.lse1(coords, f_pc, neighbor_idx)
        f_pc_agg = self.pool1(f_concat)

        f_concat = self.lse2(f_xyz, f_pc_agg, neighbor_idx)
        f_pc_agg = self.pool2(f_concat)

        return self.lrelu(self.mlp2(f_pc_agg) + self.shortcut(features))


def get_param_groups(network):
    """Param groups for optimizer."""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]
