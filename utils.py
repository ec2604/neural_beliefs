import numpy as np
import torch
import matplotlib.pyplot as plt

def sample_negatives(z_batch, negative_sampling_factor):
    cdist = torch.cdist(z_batch.reshape(-1, z_batch.shape[-1]), z_batch.reshape(-1, z_batch.shape[-1]))
    numerator = (cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1))
    denom = (cdist > cdist.min(dim=-1).values.view(cdist.shape[0], 1)).sum(dim=-1).view(cdist.shape[0], 1)
    obs_neg = z_batch.reshape(-1, z_batch.shape[-1])[
        torch.multinomial(numerator / denom, negative_sampling_factor)]
    return obs_neg.reshape(z_batch.shape[0], z_batch.shape[1], negative_sampling_factor, z_batch.shape[-1])

def get_pos_grid(pos, device):
    X_SIZE = 12
    Y_SIZE = 12
    grid = torch.zeros(len(pos), X_SIZE*Y_SIZE, dtype=torch.float32, device=device)
    idx = torch.tensor(np.array([pos_[0]*(Y_SIZE) + pos_[1] for pos_ in pos]),device=device).view(-1, 1)
    grid.scatter_(dim=1, index=idx, src=torch.full_like(grid, fill_value=1))
    return grid.view(-1, X_SIZE, Y_SIZE)


def one_hot_orientation(orientation, device):
    ori_grid = torch.zeros(len(orientation), 4, dtype=torch.float32, device=device)
    idx = torch.tensor(np.array([int(round(ori / 90)%4) for ori in orientation]), device=device).view(-1, 1)
    ori_grid.scatter_(dim=1, index=idx, src=torch.full_like(ori_grid, fill_value=1))
    return ori_grid

def plot_grid(pos, j=0, save=False):
    X_SIZE = 9
    Y_SIZE = 10
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(1.- pos, cmap='gray', origin="lower")

    # optionally add grid
    ax.set_xticks(np.arange(X_SIZE+1)-0.5, minor=True)
    ax.set_yticks(np.arange(Y_SIZE+1)-0.5, minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)
    if save:
        plt.savefig(f'grid_{j}.png')
    else:
        plt.show()

def position_tracker_to_tb(args, writer, position, position_pred, step):
    fig, ax = plt.subplots(1, 3)
    rand_traj = np.random.choice(np.arange(args.batch))
    rand_step = np.random.choice(np.arange(10, args.trajectory_len))
    ax[0].set_title(f'Actual Position, step {rand_step}')
    ax[1].set_title(f'Position Belief, step {rand_step}')
    ax[2].set_title(f'History until step {rand_step}')

    history = (position[rand_traj, :rand_step, ...].sum(dim=0) > 0).type(torch.float32)

    ax[0].imshow(1. - position[rand_traj, rand_step, ...].squeeze(0).detach().cpu().numpy(), cmap='gray',
                 origin="lower", vmin=0, vmax=1)
    ax[1].imshow(1. - torch.softmax(position_pred[rand_traj, rand_step, ...], dim=-1).reshape(args.grid_size_x,
                                                                                              args.grid_size_y).detach().cpu().numpy(),
                 cmap='gray', origin="lower", vmin=0, vmax=1)
    ax[2].imshow(1. - history.detach().cpu().squeeze(0).numpy(), cmap='gray', origin='lower', vmin=0, vmax=1)
    # optionally add grid
    ax[0].set_xticks(np.arange(args.grid_size_x + 1) - 0.5, minor=True)
    ax[0].set_yticks(np.arange(args.grid_size_y + 1) - 0.5, minor=True)
    ax[0].grid(which="minor")
    ax[0].tick_params(which="minor", size=0)
    ax[1].set_xticks(np.arange(args.grid_size_x + 1) - 0.5, minor=True)
    ax[1].set_yticks(np.arange(args.grid_size_y + 1) - 0.5, minor=True)
    ax[1].grid(which="minor")
    ax[1].tick_params(which="minor", size=0)
    ax[2].set_xticks(np.arange(args.grid_size_x + 1) - 0.5, minor=True)
    ax[2].set_yticks(np.arange(args.grid_size_y + 1) - 0.5, minor=True)
    ax[2].grid(which="minor")
    ax[2].tick_params(which="minor", size=0)
    fig.tight_layout()
    writer.add_figure('position_tracker', fig, global_step=step)
    plt.close(fig)