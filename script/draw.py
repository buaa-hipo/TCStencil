import itertools
import matplotlib.pyplot as plt
import numpy as np
from param import FLAG_DICT

font_family = 'Times New Roman'


def load_tile_csv(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    result_dict = {}
    for stencil_type in FLAG_DICT['stencil_size']:
        result_dict[stencil_type] = {'baseline': {}, 'tensor': {}}
    for line in lines:
        compute_type, stencil_type, mesh_size, time_val = line.split(',')[:-1]
        mesh_size, time_val = int(mesh_size), float(time_val)
        update = mesh_size * mesh_size / time_val * 1e6
        result_dict[stencil_type][compute_type][mesh_size] = {'time_val': time_val, 'update': update}
    return result_dict


def draw_tile_1(A100_tile_1, V100_tile_1, is_box):
    fig = plt.figure()
    ax_list = []
    x_offset = 0.48
    y_offset = 0.48
    y_lim = 0
    for index in range(4):
        x = (index % 2) + 1
        y = (index // 2) + 1
        tile1 = A100_tile_1 if x == 1 else V100_tile_1
        ax = fig.add_axes([0.1 + x_offset * (x - 1), 0.11 + y_offset * (2 - y), 0.38, 0.33])
        ax_list.append(ax)
        stencil_type = f'box2d{y}r' if is_box else f'star2d{y}r'
        baseline = tile1[stencil_type]['baseline']
        tensor = tile1[stencil_type]['tensor']
        x_list = list(baseline.keys())
        baseline_update = [baseline[x]['update'] for x in x_list]
        tensor_update = [tensor[x]['update'] for x in x_list]
        y_lim = max(y_lim, tensor_update[-1])
        if index == 0:
            ax.plot(x_list, baseline_update, ':', label='TC-w/o-tc')
            ax.plot(x_list, tensor_update, '-', label='TC')
            ax.legend(prop={'family': font_family, 'size': 11})
        else:
            ax.plot(x_list, baseline_update, ':')
            ax.plot(x_list, tensor_update, '-')
        ax.set_xlabel('mesh size N', fontsize=14, fontfamily=font_family)
        ax.set_ylabel('update points/s', fontsize=14, fontfamily=font_family)
        plt.xticks([0, 1800, 3600, 5400, 7200], [0, 1800, 3600, 5400, 7200], fontsize=14,
                   fontfamily=font_family)
        plt.yticks(fontsize=12, fontfamily=font_family)
    y_lim *= 1.05
    for ax in ax_list:
        ax.set_ylim(top=y_lim)
    title = 'box' if is_box else 'star'
    r1 = 'r=1 9pt' if is_box else 'r=1 5pt'
    r2 = 'r=2 25pt' if is_box else 'r=2 9pt'
    plt.text(-2200, y_lim * 2.55, title, fontsize=24, fontfamily=font_family)
    plt.text(-6800, y_lim * 2.5, 'A100', fontsize=18, fontfamily=font_family)
    plt.text(2700, y_lim * 2.5, 'V100', fontsize=18, fontfamily=font_family)
    plt.text(-2200, y_lim * 1.18, r1, fontsize=18, fontfamily=font_family)
    plt.text(-2200, y_lim * -0.28, r2, fontsize=18, fontfamily=font_family)
    plt.savefig(fname=f'./figure/fig-{title}-tile1.pdf')
    plt.show()


def tile1_pass(data_dir):
    A100_tile_1 = load_tile_csv(data_dir.format('A100') + 'ncu_result_tile1.csv')
    V100_tile_1 = load_tile_csv(data_dir.format('V100') + 'ncu_result_tile1.csv')
    draw_tile_1(A100_tile_1, V100_tile_1, False)
    draw_tile_1(A100_tile_1, V100_tile_1, True)


def load_full_csv(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    result_dict = {}
    for stencil_type in FLAG_DICT['stencil_size']:
        result_dict[stencil_type] = {}
    for line in lines:
        compute_type, stencil_type, mesh_size, tile_size, time_val = line.split(',')[:-1]
        if compute_type == 'baseline':
            continue
        mesh_size, tile_size, time_val = int(mesh_size), int(tile_size), float(time_val)
        update = mesh_size * mesh_size / time_val * 1e6
        if mesh_size in result_dict[stencil_type].keys():
            result_dict[stencil_type][mesh_size][tile_size] = {'time_val': time_val, 'update': update}
        else:
            result_dict[stencil_type][mesh_size] = {
                tile_size: {'time_val': time_val, 'update': update}
            }
    return result_dict


def draw_mult_tile(A100_tile, V100_tile, is_box):
    fig = plt.figure()
    ax_list = []
    x_offset = 0.48
    y_offset = 0.48
    y_max = 0
    y_min = float('inf')
    mesh_list = [2880, 3840, 5760, 7200]
    for index in range(4):
        x = (index % 2) + 1
        y = (index // 2) + 1
        tile = A100_tile if x == 1 else V100_tile
        ax = fig.add_axes([0.1 + x_offset * (x - 1), 0.1 + y_offset * (2 - y), 0.38, 0.33])
        ax_list.append(ax)
        stencil_type = f'box2d{y}r' if is_box else f'star2d{y}r'
        res = tile[stencil_type]
        x_list = []  # diff tile
        y_list = []
        for mesh_size in mesh_list:
            x_list.append(list(res[mesh_size].keys()))
            y_list.append([res[mesh_size][x]['update'] for x in x_list[-1]])
            y_max = max(y_max, max(y_list[-1]))
            y_min = min(y_min, min(y_list[-1]))
        for i in range(len(mesh_list)):
            ax.plot(x_list[i], y_list[i], '-', label=f'N={mesh_list[i]}')
        if index == 1:
            ax.legend(prop={'family': font_family, 'size': 9})
        ax.set_xlabel('tileX', fontsize=14, fontfamily=font_family)
        ax.set_ylabel('update points/s', fontsize=14, fontfamily=font_family)
        plt.xticks([1, 5, 10, 15], [1, 5, 10, 15], fontsize=14, fontfamily=font_family)
        plt.yticks(fontsize=12, fontfamily=font_family)
    y_max *= 1.05
    y_min *= 0.98
    for ax in ax_list:
        ax.set_ylim(bottom=y_min, top=y_max)
    title = 'box' if is_box else 'star'
    r1 = 'r=1 9pt' if is_box else 'r=1 5pt'
    r2 = 'r=2 25pt' if is_box else 'r=2 25pt'
    plt.text(-4, y_max * 2, title, fontsize=24, fontfamily=font_family)
    plt.text(-13, y_max * 1.98, 'A100', fontsize=18, fontfamily=font_family)
    plt.text(6, y_max * 1.98, 'V100', fontsize=18, fontfamily=font_family)
    plt.text(-3.8, y_max * 1.1, r1, fontsize=18, fontfamily=font_family)
    plt.text(-3.8, y_max * 0.18, r2, fontsize=18, fontfamily=font_family)
    plt.savefig(fname=f'./figure/fig-{title}-mult_tile.pdf')
    plt.show()


def mult_tile_pass(data_dir):
    A100_tile = load_full_csv(data_dir.format('A100') + 'ncu_result_full.csv')
    V100_tile = load_full_csv(data_dir.format('V100') + 'ncu_result_full.csv')
    draw_mult_tile(A100_tile, V100_tile, False)
    draw_mult_tile(A100_tile, V100_tile, True)


if __name__ == '__main__':
    data_dir = './data/{}/layout16/'
    tile1_pass(data_dir)
    mult_tile_pass(data_dir)
