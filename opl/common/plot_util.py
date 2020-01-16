import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import os
traj_limitation = [5, 10, 15, 20, 25]
COLORS = ['darkgreen', 'blue', 'red', 'orange', 'cyan', 'yellow', 'magenta', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'green', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']


def argsparser():
    parser = argparse.ArgumentParser("plot")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    return parser.parse_args()


def plot_return_bc(args, file_path):
    rets = np.load(file_path)
    plt.figure(dpi=300)
    fig, axs = plt.subplots()
    axs.plot(traj_limitation, rets, marker='*', markersize=8, ls='--', color=COLORS[0])
    axs.set_xticks(traj_limitation)
    ymin = np.min(rets)
    ymax = 1.2 * np.max(rets)
    axs.set_ylim([ymin, ymax])
    axs.set_title(args.env_id)
    path = 'plots'
    os.makedirs(path, exist_ok=True)
    fig.text(0.55, 0.04, 'Number of Expert Trajectories', fontsize=10, ha='center')
    fig.text(0.04, 0.55, 'Return', fontsize=10, ha='center', rotation='vertical')
    fig.legend(
        ['Behavioral Cloning'],
        loc='upper center',
        fancybox=True,
    )
    file_name = osp.join(path, args.env_id + '.pdf')
    fig.savefig(file_name)


def main(args):
    file_path = osp.join('results', args.env_id + '.npy')
    plot_return_bc(args, file_path)


if __name__ == '__main__':
    args = argsparser()
    main(args=args)