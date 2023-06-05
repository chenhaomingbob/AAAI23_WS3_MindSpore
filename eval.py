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
import os
import argparse
import ast
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix

from mindspore import context, load_checkpoint, load_param_into_net, ops
from mindspore import dtype as mstype

from src.data.S3DIS_dataset_test import dataloader, ms_map
from src.utils.tools import DataProcessing as DP
from src.utils.tools import ConfigS3DIS as cfg

from src.utils.logger import get_logger
from src.utils.helper_ply import write_ply

from tqdm import tqdm


def run_eval(params):
    context.set_context(mode=context.PYNATIVE_MODE, device_target=params.device_target, device_id=params.device_id)
    logger = get_logger(params.outputs_dir, params.rank)
    _, val_ds, dataset = dataloader(dataset_dir=params.dataset_dir, num_parallel_workers=8, shuffle=False)
    input_columns = ["xyz", "colors", "labels", "q_idx", "c_idx"]
    output_columns = ["features", "labels", "input_inds", "cloud_inds",
                      "p0", "p1", "p2", "p3", "p4",
                      "n0", "n1", "n2", "n3", "n4",
                      "pl0", "pl1", "pl2", "pl3", "pl4",
                      "u0", "u1", "u2", "u3", "u4"]
    val_loader = val_ds.batch(batch_size=params.batch_size,
                              per_batch_map=ms_map,
                              input_columns=input_columns,
                              output_columns=output_columns,
                              drop_remainder=True)
    val_ds_size = val_loader.get_dataset_size()
    val_loader = val_loader.create_dict_iterator()

    if params.device_target == 'Ascend':
        from src.model.model_s3dis_remove_bias import WS3
    else:
        from src.model.model_s3dis import WS3

    network = WS3(6, cfg.num_classes)
    network.set_train(False)

    if params.float16:
        print("network uses float16")
        network.to_float(mstype.float16)

    if '.ckpt' in params.model_path:
        ckpts = [params.model_path]
    else:
        ckpt_path = Path(os.path.join(params.model_path, 'ckpt'))
        ckpts = ckpt_path.glob('*.ckpt')
        ckpts = sorted(ckpts, key=lambda ckpt: ckpt.stem.split("_")[0].split("-")[1])

    best_miou, best_ckpt = 0.0, ckpts[0]
    ckpt_bar = tqdm(total=len(ckpts), leave=False, desc='Step', dynamic_ncols=True)
    logger.info('==========begin test===============')
    for ckpt_i, ckpt in enumerate(ckpts):
        logger.info('load ckpt from:{}'.format(str(ckpt)))
        param_dict = load_checkpoint(str(ckpt))
        load_param_into_net(network, param_dict)

        val_proportions = np.zeros(cfg.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1
        test_probs = [np.zeros(shape=[l.shape[0], cfg.num_classes], dtype=np.float32)
                      for l in dataset.input_labels['validation']]

        # Smoothing parameter for votes
        test_smooth = 0.95
        step_bar = tqdm(total=val_ds_size, leave=False, desc='Step', dynamic_ncols=True)
        for step_i, data in enumerate(val_loader):
            features = data['features']
            labels = data['labels']  # (B,N)
            xyz = [data['p0'], data['p1'], data['p2'], data['p3'], data['p4']]
            neigh_idx = [data['n0'], data['n1'], data['n2'], data['n3'], data['n4']]
            sub_idx = [data['pl0'], data['pl1'], data['pl2'], data['pl3'], data['pl4']]
            interp_idx = [data['u0'], data['u1'], data['u2'], data['u3'], data['u4']]
            point_idx = data['input_inds'].asnumpy()
            cloud_idx = data['cloud_inds'].asnumpy()

            logits = network(xyz, features, neigh_idx, sub_idx, interp_idx)  # [b, num_classes, N]
            logits = logits[..., :cfg.num_classes]

            prob_logits = ops.Softmax(-1)(logits)

            prob_logits = prob_logits.asnumpy()
            for j in range(np.shape(prob_logits)[0]):
                probs = prob_logits[j]
                p_idx = point_idx[j, :]
                c_i = cloud_idx[j][0]
                test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs

            correct = np.sum(np.argmax(prob_logits, axis=-1) == labels.asnumpy())
            acc = correct / float(np.prod(np.shape(labels)))
            msg = f'Step: {str(step_i)}; acc: {str(acc)}'
            step_bar.set_postfix_str(msg, refresh=False)
            step_bar.update()

        best_ckpt, best_miou = cal_metric(best_ckpt, best_miou, ckpt, ckpt_i, dataset, logger, params, test_probs,
                                          val_ds, val_proportions)
        ckpt_bar.update()

    logger.info('All ckpt test end. Best MIOU: {:.1f} . Best ckpt: {}'.format(100 * best_miou, str(best_ckpt)))


def cal_metric(best_ckpt, best_miou, ckpt, ckpt_i, dataset, logger, params, test_probs, val_ds, val_proportions):
    last_min, num_votes = -0.5, 100
    while last_min < num_votes:
        new_min = np.min(val_ds.source.min_possibility['validation'])
        logger.info(f"Epoch {ckpt_i}, end.  Min possibility = {new_min:.1f}")
        if last_min + 1 < new_min:
            last_min += 1
            logger.info('Confusion on sub clouds')
            confusion_list = []

            num_val = len(dataset.input_labels['validation'])

            for i_test in range(num_val):
                probs = test_probs[i_test]
                preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                labels = dataset.input_labels['validation'][i_test]
                confusion_list += [confusion_matrix(labels, preds, labels=dataset.label_values)]

            # Regroup confusions
            C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
            # Rescale with the right number of point per class
            C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
            # Compute IoUs
            IoUs = DP.IoU_from_confusions(C)
            m_IoU = np.mean(IoUs)
            s = '{:5.2f} | '.format(100 * m_IoU)
            for IoU in IoUs:
                s += '{:5.2f} '.format(100 * IoU)
            logger.info(s + '\n')
            if int(np.ceil(new_min)) % 1 == 0:
                # Project predictions
                logger.info('Reproject Vote #{:d}'.format(int(np.floor(new_min))))
                proj_probs_list = []
                for i_val in range(num_val):
                    # Reproject probs back to the evaluations points
                    proj_idx = dataset.val_proj[i_val]
                    probs = test_probs[i_val][proj_idx, :]
                    proj_probs_list += [probs]
                # Show vote results
                logger.info('Confusion on full clouds')
                confusion_list = []
                for i_test in range(num_val):
                    # Get the predicted labels
                    preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                    # Confusion
                    labels = dataset.val_labels[i_test]
                    acc = np.sum(preds == labels) / len(labels)
                    logger.info(dataset.input_names['validation'][i_test] + ' Acc:' + str(acc))
                    confusion_list += [confusion_matrix(y_true=labels, y_pred=preds, labels=dataset.label_values)]
                    name = dataset.input_names['validation'][i_test] + '.ply'
                    write_ply(os.path.join(params.outputs_dir, 'val_preds', name), [preds, labels],
                              ['pred', 'label'])
                # Regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0)
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                if m_IoU > best_miou:
                    best_miou = m_IoU
                    best_ckpt = ckpt
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                logger.info('-' * len(s))
                logger.info(s)
                logger.info('-' * len(s) + '\n')
                logger.info('==========end test===============')
                break
    return best_ckpt, best_miou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='WS3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--batch_size', type=int, help='val batch size', default=20)
    parser.add_argument('--val_area', type=str, help='area to validate', default='Area_5')
    parser.add_argument('--model_path', type=str, help='model saved path', default='runs')
    parser.add_argument('--device_target', type=str, help='Ascend | GPU', default='Ascend', choices=['Ascend', 'GPU'])
    parser.add_argument('--device_id', type=int, help='GPU id to use', default=0)
    parser.add_argument('--rank', type=int, help='rank', default=0)
    parser.add_argument('--dataset_dir', type=str, help='path of dataset', default='./datasets/S3DIS')
    parser.add_argument('--outputs_dir', type=str, help='path of output', default='test_outputs')
    parser.add_argument('--float16', type=ast.literal_eval, default=False)

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.model_path = os.path.join(base_dir, args.model_path)

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    val_pred_path = os.path.join(args.outputs_dir, 'val_preds')
    if not os.path.exists(val_pred_path):
        os.makedirs(val_pred_path)

    run_eval(args)
