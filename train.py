from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from datetime import datetime as dt

from collections import deque
from model import CPCI_Action_30
from data_utils import SubTrajectory, TrajectoryBuffer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import utils


import environment

def cycle(iterable, batch_size):
    while True:
        for x in iterable:
            yield x

def generate_trajectory(env, device, agent, trajectory_len):
    obs_list = []
    pos_list = []
    orientation_list = []
    one_hot_action_list = []
    new_obs_list = []
    new_pos_list = []
    new_orientation_list = []
    step = 0
    sub_trajectory = SubTrajectory(trajectory_len)
    while step < trajectory_len:
        one_hot_action = agent.step()
        obs, pos, orientation = env.get_observation()
        _ = env.step(np.argmax(one_hot_action))

        new_obs, new_pos, new_orientation = env.get_observation()
        obs_list.append(obs)
        pos_list.append(pos)
        orientation_list.append(orientation)
        one_hot_action_list.append(one_hot_action)
        new_obs_list.append(new_obs)
        new_pos_list.append(new_pos)
        new_orientation_list.append(new_orientation)
        step += 1
    obs_list = torch.from_numpy(np.concatenate(obs_list)).type('torch.FloatTensor').to(device)
    new_obs_list = torch.from_numpy(np.concatenate(new_obs_list)).type('torch.FloatTensor').to(device)
    #for i in range(trajectory_len):
    #    sub_trajectory.add(obs_list[i], pos_list[i], orientation_list[i], one_hot_action_list[i], new_obs_list[i], new_pos_list[i], new_orientation_list[i])
    sub_trajectory.obs = obs_list
    sub_trajectory.new_obs = new_obs_list
    sub_trajectory.new_pos = new_pos_list
    sub_trajectory.pos = pos_list
    sub_trajectory.ori = orientation_list
    sub_trajectory.new_ori = new_orientation_list
    sub_trajectory.action = torch.from_numpy(np.concatenate(one_hot_action_list)).type('torch.FloatTensor').to(device)
    sub_trajectory.position_list = utils.get_pos_grid(list(sub_trajectory.new_pos), device).view(trajectory_len, -1, 12, 12)
    sub_trajectory.orig_position = utils.get_pos_grid([sub_trajectory.pos[0]], device).view(1, 12, 12)
    sub_trajectory.orientation_list = utils.one_hot_orientation(list(sub_trajectory.new_ori), device).view(trajectory_len, -1, 4)
    sub_trajectory.orig_orientation = utils.one_hot_orientation([sub_trajectory.ori[0]], device)
    return sub_trajectory

def fill_buffer(replay_buffer, env, device, agent, trajectory_len):
    replay_buffer.clear()
    add_to_buffer(replay_buffer, env, device,agent, trajectory_len, replay_buffer.maxSize)

def add_to_buffer(replay_buffer, env, device,agent, trajectory_len, n):
    print("\nFilling buffer...", flush=True)
    pbar = tqdm(total = n)
    for i in range(n):
        new_trajectory = generate_trajectory(env, device, agent, trajectory_len)

        replay_buffer.add_trajectory(new_trajectory)
        env.reset()
        if i % 50 == 0 and i != 0:
            pbar.update(50)

def load_buffer(file_path, buffer):
    npz = np.load(file_path)
    buffer.obs = torch.from_numpy(npz['obs'])
    buffer.pos = torch.from_numpy(npz['pos'])
    buffer.ori = torch.from_numpy(npz['ori'])
    buffer.action = torch.from_numpy(npz['action'])
    buffer.new_obs = torch.from_numpy(npz['new_obs'])
    buffer.new_pos = torch.from_numpy(npz['new_pos'])
    buffer.new_ori = torch.from_numpy(npz['new_ori'])


def prep_data(data, batch_size):
    ds = torch.utils.data.TensorDataset(data.obs, data.pos, data.ori, data.action, data.new_obs, data.new_pos,
                                              data.new_ori)
    loader = iter(cycle(torch.utils.data.DataLoader(ds, batch_size=batch_size, drop_last=True, pin_memory=True,
                                                          shuffle=True), batch_size=batch_size))
    return loader



def train(env, model, args, device):
    time_str = dt.now().strftime('%Y_%m_%d_%H_%M')
    train_writer = SummaryWriter(log_dir=f'/mnt/data/erac/{time_str}/train')
    test_writer = SummaryWriter(log_dir=f'/mnt/data/erac/{time_str}/test')
    train_data_size = args.train_data_size
    test_data_size = args.test_data_size
    batch_size = args.batch_size
    trajectory_len = args.trajectory_len

    agent = environment.GridWorldAgent(env)
    max_cpc_steps = args.num_cpc_steps
    max_eval_steps = args.num_eval_steps
    num_eval_batches = args.num_eval_batches
    num_batches_until_eval = args.num_batches_until_eval
    tensorboard_print_cycle = args.cpc_tb_print_cycle

    test_data = TrajectoryBuffer(test_data_size, trajectory_len, obs_dim=(5, 5), pos_dim=(1, 12,12), ori_dim=(1, 4), action_dim=(4,))
    train_data = TrajectoryBuffer(train_data_size, trajectory_len, obs_dim=(5,5), pos_dim=(1, 12,12), ori_dim=(1, 4), action_dim=(4,))
    env.reset()

    step = 0

    pbar = tqdm(total = max_cpc_steps)
    fill_buffer(train_data, env, device, agent, trajectory_len)
    fill_buffer(test_data, env, device, agent, trajectory_len)

    train_loader = prep_data(train_data, batch_size)
    test_loader = prep_data(test_data, batch_size)


    loss_pos_list = [None for _ in range(tensorboard_print_cycle)]
    loss_neg_list = [None for _ in range(tensorboard_print_cycle)]
    while step < max_cpc_steps:
        batch = [tensor.to(device) for tensor in next(train_loader)]
        # Train using train_data
        loss_pos, loss_neg, z_neg_dist = model.update(batch)
        loss_pos_list[step % tensorboard_print_cycle] = loss_pos
        loss_neg_list[step % tensorboard_print_cycle] = loss_neg
        train_writer.add_histogram('z_pos_vs_neg_dist', z_neg_dist, step)


        if step % tensorboard_print_cycle == 0 and step != 0:
            loss_pos_concat = torch.stack(loss_pos_list).detach().cpu().numpy()
            loss_neg_concat = torch.stack(loss_neg_list).detach().cpu().numpy()
            train_writer.add_scalar('loss_pos', loss_pos_concat.mean(), step)
            train_writer.add_scalar('loss_neg', loss_neg_concat.mean(), step)
            train_writer.flush()
            print("Loss pos: ", loss_pos_concat.mean(), "loss neg:", loss_neg_concat.mean(), flush=True)
            pbar.update(tensorboard_print_cycle)
        if step % num_batches_until_eval == 0 and step != 0:
            model.eval()
            test_loss_pos, test_loss_neg = None, None
            for i in range(num_eval_batches):
                test_batch = [tensor.to(device) for tensor in next(test_loader)]
                if test_loss_pos is None:
                    test_loss_pos, test_loss_neg, z_neg_dist = model.update(test_batch, save=False, train=False)
                    test_loss_pos = test_loss_pos.detach().cpu().numpy() / num_eval_batches
                    test_loss_neg = test_loss_neg.detach().cpu().numpy() / num_eval_batches
                else:
                    test_loss_pos_, test_loss_neg_, z_neg_dist = model.update(test_batch, save=False, train=False)
                    test_loss_pos_ = test_loss_pos_.detach().cpu().numpy() / num_eval_batches
                    test_loss_neg_ = test_loss_neg_.detach().cpu().numpy() / num_eval_batches
                    test_loss_pos = test_loss_pos + test_loss_pos_
                    test_loss_neg = test_loss_neg + test_loss_neg_
            test_writer.add_scalar('loss_pos', test_loss_pos, step)
            test_writer.add_scalar('loss_neg', test_loss_neg, step)
            test_writer.flush()
            model.train()
        step += 1
    pbar = tqdm(total=max_eval_steps)
    step = 0
    for p in model.mlp.parameters():
        p.requires_grad = False
    for p in model.observation_to_latent_mapper.parameters():
        p.requires_grad = False
    for p in model.belief_gru.parameters():
        p.requires_grad = False
    for p in model.action_gru.parameters():
        p.requires_grad = False
    torch.save(model, f'/mnt/data/erac/{time_str}/model_cpt')
    eval_tensorboard_print_cycle = args.eval_tb_print_cycle
    loss_position_list = [None for _ in range(eval_tensorboard_print_cycle)]
    loss_orientation_list = [None for _ in range(eval_tensorboard_print_cycle)]
    while step < max_eval_steps:
        train_batch = [tensor.to(device) for tensor in next(train_loader)]
        save = (step == max_eval_steps -1)
        loss_evalmlp_pos, loss_evalmlp_ori, position_pred = model.update(train_batch, save=save, eval_mlp=True)
        loss_position_list[step % eval_tensorboard_print_cycle] = loss_evalmlp_pos
        loss_orientation_list[step % eval_tensorboard_print_cycle] = loss_evalmlp_ori
        if step % eval_tensorboard_print_cycle == 0 and step != 0:
            position = train_batch[5].detach()
            utils.position_tracker_to_tb(args, train_writer, position, position_pred, step)
            loss_position_concat = torch.stack(loss_position_list).detach().cpu().numpy()
            loss_orientation_concat = torch.stack(loss_orientation_list).detach().cpu().numpy()
            train_writer.add_scalar('loss_position', loss_position_concat.mean(), step)
            train_writer.add_scalar('loss_orientation', loss_orientation_concat.mean(), step)
            train_writer.flush()
            print("Eval loss position: ", loss_evalmlp_pos.data, "Eval loss orientation",  loss_evalmlp_ori.data,
                  flush=True)
            pbar.update(eval_tensorboard_print_cycle)

        if step % num_batches_until_eval == 0 and step != 0:
            model.eval()
            test_eval_mlp_pos, test_eval_mlp_orientation = None, None
            for i in range(num_eval_batches):
                test_batch = [tensor.to(device) for tensor in next(test_loader)]
                if test_eval_mlp_pos is None:
                    test_eval_mlp_pos, test_eval_mlp_orientation, position_pred = model.update(test_batch, save=False, eval_mlp=True, train=False)
                    test_eval_mlp_pos = test_eval_mlp_pos.detach().cpu().numpy() / num_eval_batches
                    test_eval_mlp_orientation = test_eval_mlp_orientation.detach().cpu().numpy() / num_eval_batches
                else:
                    test_eval_mlp_pos_, test_eval_mlp_orientation_, position_pred = model.update(test_batch, save=False, eval_mlp=True, train=False)
                    test_eval_mlp_pos_  = test_eval_mlp_pos_.detach().cpu().numpy() / num_eval_batches
                    test_eval_mlp_orientation_ = test_eval_mlp_orientation_.detach().cpu().numpy() / num_eval_batches
                    test_eval_mlp_pos = test_eval_mlp_pos + test_eval_mlp_pos_
                    test_eval_mlp_orientation = test_eval_mlp_orientation + test_eval_mlp_orientation_
            position = test_batch[5].detach()
            utils.position_tracker_to_tb(args, test_writer, position, position_pred, step)
            test_writer.add_scalar('loss_position', test_eval_mlp_pos, step)
            test_writer.add_scalar('loss_orientation', test_eval_mlp_orientation, step)
            test_writer.flush()
            model.train()
        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Minibatch size for subtrajectories')
    parser.add_argument('--num-cpc-steps', type=int, default=int(6e4), help='Number of training episodes')
    parser.add_argument('--num-eval-steps', type=int, default=int(6e4), help='Number of eval episodes')
    parser.add_argument('--trajectory_len', type=int, default=int(80), help='Trajectory length')
    parser.add_argument('--negative_sampling_factor', type=int, default=int(1), help='Number of negative samples for CPC')
    parser.add_argument('--look_ahead', type=int, default=int(1))
    parser.add_argument('--grid_size_x', type=int, default=int(12))
    parser.add_argument('--grid_size_y', type=int, default=int(12))
    parser.add_argument('--train_data_size', type=int, default=int(1e5))
    parser.add_argument('--test_data_size', type=int, default=int(2e4))
    parser.add_argument('--cpc_tb_print_cycle', type=int, default=int(50))
    parser.add_argument('--eval_tb_print_cycle', type=int, default=int(200))
    parser.add_argument('--num_eval_batches', type=int, default=int(300))
    parser.add_argument('--cpc_lr', type=float, default=2e-4)
    parser.add_argument('--eval_lr', type=float, default=2e-4)
    parser.add_argument('--num_batches_until_eval', type=int, default=1e3)
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPCI_Action_30(device, args.trajectory_len, args.negative_sampling_factor, cpc_lr=args.cpc_lr,
                           eval_lr=args.eval_lr, look_ahead=args.look_ahead, grid_x=args.grid_size_x,
                           grid_y=args.grid_size_y)
    model.to(device)
    env = environment.GridWorld(args.grid_size_x, args.grid_size_y)
    train(env, model, args, device)
    # position_pred = np.load('./position_pred.npy')
    # position = np.load('./position_true.npy')
    # X_SIZE = 12
    # Y_SIZE = 12
    # import matplotlib.pyplot as plt
    # for i in range(80):
    #     fig, ax = plt.subplots(1, 2)
    #
    #     ax[0].set_title(f'Actual Position, step {i}')
    #     ax[1].set_title(f'Position Belief, step {i}')
    #     ax[0].imshow(1. - position[1,i,...].squeeze(0), cmap='gray', origin="lower",vmin=0, vmax=1)
    #     ax[1].imshow(1. - torch.softmax(torch.from_numpy(position_pred[1,i,...]),dim=-1).reshape(12,12).numpy(),cmap='gray', origin="lower",vmin=0, vmax=1)
    #     # optionally add grid
    #     ax[0].set_xticks(np.arange(X_SIZE + 1) - 0.5, minor=True)
    #     ax[0].set_yticks(np.arange(Y_SIZE + 1) - 0.5, minor=True)
    #     ax[0].grid(which="minor")
    #     ax[0].tick_params(which="minor", size=0)
    #     ax[1].set_xticks(np.arange(X_SIZE + 1) - 0.5, minor=True)
    #     ax[1].set_yticks(np.arange(Y_SIZE + 1) - 0.5, minor=True)
    #     ax[1].grid(which="minor")
    #     ax[1].tick_params(which="minor", size=0)
    #     plt.show()
    #     plt.close()
    #
    #
