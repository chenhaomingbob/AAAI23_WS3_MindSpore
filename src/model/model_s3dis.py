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
from mindspore import dtype as mstype
from mindspore import Tensor

from .base_model import SharedMLP, LocalFeatureAggregation


class WS3(nn.Cell):
    def __init__(self, d_in, num_classes):
        super(WS3, self).__init__()

        self.fc_start = nn.Dense(d_in, 8)
        self.bn_start = nn.SequentialCell([
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ])

        # encoding layers
        self.encoder = nn.CellList([
            LocalFeatureAggregation(8, 16),
            LocalFeatureAggregation(32, 64),
            LocalFeatureAggregation(128, 128),
            LocalFeatureAggregation(256, 256),
            LocalFeatureAggregation(512, 512)
        ])

        self.mlp = SharedMLP(1024, 1024, bn=True, activation_fn=nn.LeakyReLU(0.2))

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.LeakyReLU(0.2)
        )
        self.decoder = nn.CellList([
            SharedMLP(1536, 512, **decoder_kwargs),
            SharedMLP(768, 256, **decoder_kwargs),
            SharedMLP(384, 128, **decoder_kwargs),
            SharedMLP(160, 32, **decoder_kwargs),
            SharedMLP(64, 32, **decoder_kwargs)
        ])

        self.fc_end_fc1 = SharedMLP(32, 64, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.fc_end_fc2 = SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.fc_end_drop = nn.Dropout()
        self.fc_end_fc3 = SharedMLP(32, num_classes)

    def construct(self, xyz, feature, neighbor_idx, sub_idx, interp_idx):
        r"""
            construct method

            Parameters
            ----------
            xyz: list of ms.Tensor, shape (num_layer, B, N_layer, 3), each layer xyz
            feature: ms.Tensor, shape (B, N, d), input feature [xyz ; feature]
            neighbor_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer knn neighbor idx
            sub_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 16), each layer pooling idx
            interp_idx: list of ms.Tensor, shape (num_layer, B, N_layer, 1), each layer interp idx

            Returns
            -------
            ms.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        feature = self.fc_start(feature).swapaxes(-2, -1).expand_dims(-1)  # (B, N, 6) -> (B, 8, N, 1)
        feature = self.bn_start(feature)  # shape (B, 8, N, 1)
        # <<<<<<<<<< ENCODER
        f_stack = []
        for i in range(5):
            f_encoder_i = self.encoder[i](xyz[i], feature,
                                          neighbor_idx[i])  # (B,40960,3)  (4, 8, 40960, 1) (4, 40960, 16)
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                f_stack.append(f_encoder_i)
            f_stack.append(f_sampled_i)
        # >>>>>>>>>> ENCODER
        feature = self.mlp(f_stack[-1])  # [B, d, N, 1]
        # <<<<<<<<<< DECODER
        f_decoder_list = []
        for j in range(5):
            f_interp_i = self.random_sample(feature, interp_idx[-j - 1])  # [B, d, n, 1]
            cat = P.Concat(1)
            f_decoder_i = self.decoder[j](cat((f_stack[-j - 2], f_interp_i)))
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # >>>>>>>>>> DECODER
        f_layer_fc1 = self.fc_end_fc1(f_decoder_list[-1])
        f_layer_fc2 = self.fc_end_fc2(f_layer_fc1)
        f_layer_drop = self.fc_end_drop(f_layer_fc2)
        f_layer_fc3 = self.fc_end_fc3(f_layer_drop)

        f_layer_fc2, f_layer_fc3 = f_layer_fc2.swapaxes(1, 3), f_layer_fc3.swapaxes(1, 3)
        f_layer_out = P.Concat(axis=-1)([f_layer_fc3, f_layer_fc2])
        f_out = f_layer_out.squeeze(1)  # (B,N_points,13+32)

        return f_out  # (B,N_points,45)

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, d, N, 1] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, d, N', 1] pooled features matrix
        """
        b, d = feature.shape[:2]
        n_ = pool_idx.shape[1]
        # [B, N', max_num] --> [B, d, N', max_num]
        pool_idx = P.Tile()(pool_idx.expand_dims(1), (1, feature.shape[1], 1, 1))
        # [B, d, N', max_num] --> [B, d, N'*max_num]
        pool_idx = pool_idx.reshape((b, d, -1))
        pool_features = P.GatherD()(feature.squeeze(-1), -1, pool_idx.astype(ms.int32))
        pool_features = pool_features.reshape((b, d, n_, -1))
        pool_features = P.ReduceMax(keep_dims=True)(pool_features, -1)  # [B, d, N', 1]
        return pool_features


class WS3WithLoss(nn.Cell):

    def __init__(self, network, weights, num_classes, ignored_label_indexs, c_epoch, topk):
        super(WS3WithLoss, self).__init__()
        self.network = network
        self.weights = Tensor(weights, dtype=mstype.float32)
        self.num_classes = num_classes
        self.ignored_label_inds = ignored_label_indexs
        self.c_epoch = c_epoch
        self.topk = topk
        self.c_epoch_k = Tensor(self.c_epoch, dtype=mstype.float32)
        self.onehot = nn.OneHot(depth=num_classes, dtype=mstype.float32)
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, feature, feature2, labels, input_inds, cloud_inds,
                  p0, p1, p2, p3, p4, n0, n1, n2, n3, n4, pl0,
                  pl1, pl2, pl3, pl4, u0, u1, u2, u3, u4):
        xyz = [p0, p1, p2, p3, p4]
        neighbor_idx = [n0, n1, n2, n3, n4]
        sub_idx = [pl0, pl1, pl2, pl3, pl4]
        interp_idx = [u0, u1, u2, u3, u4]

        logits_embed = self.network(xyz, feature, neighbor_idx, sub_idx, interp_idx)
        labels = labels  # (B,N)

        logits = logits_embed[..., :self.num_classes]  # (B,N,45) -> (B,N,13)
        pred_embed = logits_embed[..., self.num_classes:]  # (B,N,45) -> (B,N,32)
        logits = logits.reshape((-1, self.num_classes))
        pred_embed = pred_embed.reshape((-1, 32))
        labels = labels.reshape((-1,))  # (b,n) -> (b*n)

        # Boolean mask of points that should be ignored
        ignore_mask = P.zeros_like(labels).astype(mstype.bool_)
        for ign_label in self.ignored_label_inds:
            ignore_mask = P.logical_or(ignore_mask, P.equal(labels, ign_label))  #
        valid_mask = P.logical_not(ignore_mask)  # (B*N,)

        one_hot_labels = self.onehot(labels)  # (B*N,) -> (B*N,13)
        weights = self.weights * one_hot_labels * valid_mask.reshape(-1, 1)  # (B*N,13)
        weights = P.ReduceSum()(weights, 1)
        unweighted_loss = self.loss_fn(logits, one_hot_labels)
        weighted_loss = unweighted_loss * weights
        weighted_loss = weighted_loss * valid_mask
        ce_loss = P.ReduceSum()(weighted_loss)
        num_valid_points = P.ReduceSum()(valid_mask.astype(mstype.float32))
        ce_loss = ce_loss / num_valid_points

        if self.c_epoch_k == 0:
            loss = ce_loss
        else:
            sp_loss = self.get_sp_loss_by_mask(pred_embed, logits, one_hot_labels, valid_mask,
                                               self.topk) * self.c_epoch_k
            loss = ce_loss + sp_loss

        return loss

    @staticmethod
    def get_sp_loss_by_mask(embed, logits, one_hot_label, valid_mask, topk):
        invalid_mask = P.logical_not(valid_mask)  # (B*N,)
        num_invalid_points = int(P.count_nonzero(invalid_mask.astype(mstype.int32)))
        topk += num_invalid_points
        num_class = one_hot_label.shape[1]  # scalar: 13

        valid_one_hot_label = one_hot_label * valid_mask.reshape(-1, 1)  # (B*N,13)
        valid_embed = embed * valid_mask.reshape(-1, 1)  # (B*N,32)
        invalid_embed = embed * invalid_mask.reshape(-1, 1)  # (B*N,32)
        valid_one_hot_label_T = P.transpose(valid_one_hot_label, (1, 0))
        sum_embed = P.matmul(valid_one_hot_label_T, valid_embed)
        mean_embed = sum_embed / (P.reduce_sum(valid_one_hot_label_T, axis=1).reshape(-1, 1) + 0.001)
        adj_matrix = WS3WithLoss.double_feature(invalid_embed, mean_embed)

        neg_adj = -adj_matrix  # (B*N,13)
        neg_adj_t = P.transpose(neg_adj, (1, 0))  # (13,B*N)
        _, nn_idx = P.TopK()(neg_adj_t, topk)
        s = P.shape(neg_adj_t)  # (M,N)
        row_idx = P.tile(P.expand_dims(P.arange(s[0]), 1), (1, topk))
        ones_idx = P.Stack(axis=1)([row_idx.reshape([-1]), nn_idx.reshape([-1])])
        res = P.scatter_nd(ones_idx, P.ones(s[0] * topk, neg_adj_t.dtype), s)
        nn_idx_multi_hot = P.transpose(res, (1, 0))  # [N,M]

        new_valid_mask = P.reduce_sum(nn_idx_multi_hot, axis=1) > 0  # (B*N,)
        new_valid_mask = new_valid_mask.reshape(-1, 1)  # (B*N,1)
        num_new_valid_mask = int(P.count_nonzero(new_valid_mask.astype(mstype.int32)))

        w_ij = P.exp(-1.0 * adj_matrix)  # (B*N,13)
        w_ij = w_ij * new_valid_mask  # (B*N,13)
        w_ij_label = nn_idx_multi_hot * new_valid_mask  # (B*N,13)
        w_ij = P.mul(w_ij, w_ij_label)  # (B*N,13)

        new_soft_label_hot = nn.Softmax(axis=-1)(w_ij)  # (B*N,13)
        top1 = new_soft_label_hot.argmax(axis=-1)
        soft_label_mask = P.OneHot()(top1, num_class, Tensor(1.0), Tensor(0.0))
        new_soft_label_hot = P.mul(new_soft_label_hot, soft_label_mask)

        logits = logits * new_valid_mask
        new_soft_label_hot = new_soft_label_hot * new_valid_mask
        loss = nn.SoftmaxCrossEntropyWithLogits()(logits, new_soft_label_hot)
        loss = loss.sum() / num_new_valid_mask

        return loss

    @staticmethod
    def double_feature(point_feature1, point_feature2):
        """
        Compute pairwise distance of a point cloud.
        """
        point2_transpose = P.transpose(point_feature2, (1, 0))  # [C, M]
        point_inner = P.matmul(point_feature1, point2_transpose)  # [N,M]
        point_inner = -2 * point_inner

        point1_square = P.ReduceSum(keep_dims=True)(P.square(point_feature1), axis=-1)
        point2_square = P.ReduceSum(keep_dims=True)(P.square(point_feature2), axis=-1)

        point2_square_transpose = P.transpose(point2_square, (1, 0))  # [1,M]
        adj_matrix = point1_square + point_inner + point2_square_transpose

        return adj_matrix
