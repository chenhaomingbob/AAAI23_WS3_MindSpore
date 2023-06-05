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
import datetime
import argparse
import pickle
import ast
from pathlib import Path
import numpy as np

from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net, nn, set_seed
from mindspore.nn import Adam
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig, Callback
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore import dtype as mstype

from src.data.S3DIS_dataset import dataloader, ms_map
from src.model.base_model import get_param_groups
from src.utils.tools import DataProcessing as DP
from src.utils.tools import ConfigS3DIS as config
from src.utils.logger import get_logger


class UpdateLossEpoch(Callback):
    def __init__(self, cur_num_training_ep0=30, logger=None):
        super(UpdateLossEpoch, self).__init__()
        self.training_ep = {i: np.exp(i / 100 - 1.0) - np.exp(-1.0) for i in range(0, 100)}
        self.training_ep.update({i: 0 for i in range(0, cur_num_training_ep0)})
        self.logger = logger

    # msv1.7: epoch_begin
    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()

        train_network_with_loss = cb_params.network

        cur_epoch_num = cb_params.cur_epoch_num
        train_network_with_loss.c_epoch_k += self.training_ep[cur_epoch_num - 1]

        self.logger.info(
            f"UpdateLossEpoch ==>  cur_epoch_num:{cur_epoch_num}, "
            f"cur_training_ep:{self.training_ep[cur_epoch_num]}, "
            f"loss_fn.c_epoch_k:{train_network_with_loss.c_epoch_k}")

    # msv1.8: on_train_epoch_begin
    def on_train_epoch_begin(self, run_context):
        self.epoch_begin(run_context)


class S3DISLossMonitor(Callback):
    def __init__(self, per_print_times=1, logger=None):
        super(S3DISLossMonitor, self).__init__()
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.logger = logger

    # msv1.7: step_end
    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(f"epoch: {cb_params.cur_epoch_num} "
                             f"step: {cur_step_in_epoch}. "
                             f"Invalid loss {loss}, terminating training.")

        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -= \
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num

            msg = f"epoch: {cb_params.cur_epoch_num} " \
                  f"step: {cur_step_in_epoch}, " \
                  f"loss is {loss} "
            self.logger.info(msg)

    # msv1.8: on_train_step_end
    def on_train_step_end(self, run_context):
        self.step_end(run_context)


def prepare_network(weights, cfg, args):
    """Prepare Network"""

    if args.device_target == 'Ascend':
        from src.model.model_s3dis_remove_bias import WS3
        from src.model.model_s3dis_remove_bias import WS3WithLoss
    else:
        from src.model.model_s3dis import WS3
        from src.model.model_s3dis import WS3WithLoss

    d_in = 6  # xyzrgb
    network = WS3(d_in, cfg.num_classes)
    if args.ss_pretrain:
        print(f"Load scannet pretrained ckpt from {args.ss_pretrain}")
        param_dict = load_checkpoint(args.ss_pretrain)
        whitelist = ["encoder"]
        new_param_dict = dict()
        for key, val in param_dict.items():
            if key.split(".")[0] == 'network' and key.split(".")[1] in whitelist:
                new_key = ".".join(key.split(".")[1:])
                new_param_dict[new_key] = val
        load_param_into_net(network, new_param_dict, strict_load=True)

    network = WS3WithLoss(network,
                          weights,
                          cfg.num_classes,
                          cfg.ignored_label_indexs,
                          cfg.c_epoch,
                          cfg.topk)

    if args.retrain_model:
        print(f"Load S3DIS pretrained ckpt from {args.retrain_model}")
        param_dict = load_checkpoint(args.retrain_model)
        load_param_into_net(network, param_dict, strict_load=True)

    return network


def train(cfg, args):
    if cfg.graph_mode:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)

    logger = get_logger(args.outputs_dir, args.rank)

    logger.info("============ Args =================")
    for arg in vars(args):
        logger.info('%s: %s' % (arg, getattr(args, arg)))
    logger.info("============ Cfg =================")
    for c in vars(cfg):
        logger.info('%s: %s' % (c, getattr(cfg, c)))

    train_loader, _, dataset = dataloader(cfg, shuffle=False, num_parallel_workers=8)
    ignored_label_indexs = [getattr(dataset, 'label_to_idx')[ign_label] for ign_label in
                            getattr(dataset, 'ignored_labels')]
    cfg.ignored_label_indexs = ignored_label_indexs
    weights = DP.get_class_weights("S3DIS")

    network = prepare_network(weights, cfg, args)
    decay_lr = nn.ExponentialDecayLR(cfg.learning_rate, cfg.lr_decays, decay_steps=cfg.train_steps, is_stair=True)
    opt = Adam(
        params=get_param_groups(network),
        learning_rate=decay_lr,
        loss_scale=cfg.loss_scale
    )

    log = {'cur_epoch': 1, 'cur_step': 1, 'best_epoch': 1, 'besr_miou': 0.0}
    if not os.path.exists(args.outputs_dir + '/log.pkl'):
        f = open(args.outputs_dir + '/log.pkl', 'wb')
        pickle.dump(log, f)
        f.close()

    if args.resume:
        f = open(args.resume + '/log.pkl', 'rb')
        log = pickle.load(f)
        print(f"log of resume file {log}")
        f.close()
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(network, param_dict)

    # dataloader
    train_loader = train_loader.batch(batch_size=cfg.batch_size,
                                      per_batch_map=ms_map,
                                      input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
                                      output_columns=["features", "features2", "labels", "input_inds", "cloud_inds",
                                                      "p0", "p1", "p2", "p3", "p4",
                                                      "n0", "n1", "n2", "n3", "n4",
                                                      "pl0", "pl1", "pl2", "pl3", "pl4",
                                                      "u0", "u1", "u2", "u3", "u4",
                                                      ],
                                      drop_remainder=True)

    logger.info('==========begin training===============')

    loss_scale = cfg.loss_scale
    loss_scale_manager = FixedLossScaleManager(loss_scale) if args.scale or loss_scale != 1.0 else None
    print('loss_scale:', loss_scale)

    if cfg.float16:
        print("network uses float16")
        network.to_float(mstype.float16)

    if args.scale:
        model = Model(network,
                      loss_scale_manager=loss_scale_manager,
                      loss_fn=None,
                      optimizer=opt)
    else:
        model = Model(network,
                      loss_fn=None,
                      optimizer=opt,
                      )

    # callback for loss & time cost
    loss_cb = S3DISLossMonitor(50, logger)
    time_cb = TimeMonitor(data_size=cfg.train_steps)
    cbs = [loss_cb, time_cb]

    # callback for saving ckpt
    config_ckpt = CheckpointConfig(save_checkpoint_steps=cfg.train_steps, keep_checkpoint_max=100)
    ckpt_cb = ModelCheckpoint(prefix='ws3',
                              directory=os.path.join(args.outputs_dir, 'ckpt'),
                              config=config_ckpt)
    cbs += [ckpt_cb]

    update_loss_epoch_cb = UpdateLossEpoch(args.num_training_ep0, logger)
    cbs += [update_loss_epoch_cb]

    logger.info(f"Outputs_dir:{args.outputs_dir}")
    logger.info(f"Total number of epoch: {cfg.max_epoch}; "
                f"Dataset capacity: {train_loader.get_dataset_size()}")

    model.train(cfg.max_epoch,
                train_loader,
                callbacks=cbs,
                dataset_sink_mode=False)
    logger.info('==========end training===============')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='WS3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--epochs', type=int, help='max epochs', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size', default=6)
    parser.add_argument('--dataset_dir', type=str, help='path of dataset', default='./datasets/S3DIS')
    parser.add_argument('--outputs_dir', type=str, help='path of output', default='outputs')
    parser.add_argument('--val_area', type=str, help='area to validate', default='Area_5')
    parser.add_argument('--resume', type=str, help='model to resume', default=None)
    parser.add_argument('--scale', type=ast.literal_eval, help='scale or not', default=False)
    parser.add_argument('--device_target', type=str, help='Ascend | GPU', default='Ascend', choices=['Ascend', 'GPU'])
    parser.add_argument('--device_id', type=int, help='GPU id to use', default=0)
    parser.add_argument('--rank', type=int, help='rank', default=0)
    parser.add_argument('--name', type=str, help='name of the experiment', default=None)
    parser.add_argument('--ss_pretrain', type=str, help='name of the experiment', default=None)
    parser.add_argument('--retrain_model', type=str, help='name of the experiment', default=None)
    parser.add_argument('--float16', type=ast.literal_eval, default=False)
    parser.add_argument('--train_steps', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_decays', type=float, default=0.95)
    parser.add_argument('--loss_scale', type=float, default=1.0)
    parser.add_argument('--topk', type=int, default=500)
    parser.add_argument('--num_training_ep0', type=int, default=30)
    parser.add_argument('--labeled_percent', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=888)
    parser.add_argument('--graph_mode', type=ast.literal_eval, default=False)

    arguments = parser.parse_args()

    config.dataset_dir = arguments.dataset_dir
    config.batch_size = arguments.batch_size
    config.max_epoch = arguments.epochs
    config.train_steps = arguments.train_steps
    config.learning_rate = arguments.learning_rate
    config.lr_decays = arguments.lr_decays
    config.loss_scale = arguments.loss_scale
    config.topk = arguments.topk
    num_training_ep0 = arguments.num_training_ep0
    config.training_ep0 = {i: 0 for i in range(0, num_training_ep0)}
    config.training_ep = {i: np.exp(i / 100 - 1.0) - np.exp(-1.0) for i in range(0, 100)}
    config.training_ep.update(config.training_ep0)
    config.labeled_percent = arguments.labeled_percent
    config.random_seed = arguments.random_seed
    config.graph_mode = arguments.graph_mode
    config.float16 = arguments.float16

    if arguments.name is None:
        if arguments.resume:
            arguments.name = Path(arguments.resume).split('/')[-1]
        else:
            time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
            arguments.name = f'TSteps{config.train_steps}_MaxEpoch{config.max_epoch}_BatchS{config.batch_size}' \
                             f'_TopK{config.topk}_NumTrainEp0{num_training_ep0}' \
                             f'_LP{config.labeled_percent}_RS{config.random_seed}'
            if config.graph_mode:
                arguments.name += "_GraphM"
            else:
                arguments.name += "_PyNateiveM"
            arguments.name += f'_{time_str}'

    np.random.seed(config.random_seed)
    set_seed(config.random_seed)

    arguments.outputs_dir = os.path.join(arguments.outputs_dir, arguments.name)

    print(f"outputs_dir:{arguments.outputs_dir}")
    if not os.path.exists(arguments.outputs_dir):
        os.makedirs(arguments.outputs_dir)

    if arguments.resume:
        arguments.outputs_dir = arguments.resume

    train(config, arguments)
