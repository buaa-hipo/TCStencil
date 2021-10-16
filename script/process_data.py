from param import FLAG_DICT


def load_result(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    csv_line = []
    for line in lines:
        csv_line.append(line.replace('_', ','))
    return csv_line


def store_csv(csv_line, filename):
    csv_line = [','.join(['type', 'stencil', 'mesh size', 'tile size', 'time', 'unit']) + '\n'] + csv_line
    with open(filename, 'w') as f:
        f.writelines(csv_line)


def pick_tile_csv(lines, tile_size, filename):
    csv_line = [','.join(['type', 'stencil', 'mesh size', 'time', 'unit']) + '\n']
    for line in lines:
        curr_tile = int(line.split(',')[3])
        if curr_tile == tile_size:
            csv_line.append(','.join(line.split(',')[0:3] + line.split(',')[4:]))
    with open(filename, 'w') as f:
        f.writelines(csv_line)


def pick_best_csv(lines, filename):
    res_dict = {}
    for line in lines:
        compute_type, stencil, mesh_size = line.split(',')[0:3]
        time_val = float(line.split(',')[4])
        key = f'{compute_type}-{stencil}-{mesh_size}'
        if key not in res_dict.keys() or float(time_val) < res_dict[key]['time_val']:
            res_dict[key] = {}
            res_dict[key]['time_val'] = float(time_val)
            res_dict[key]['csv'] = ','.join([compute_type, stencil, mesh_size] + line.split(',')[4:])
    csv_line = [','.join(['type', 'stencil', 'mesh size', 'time', 'unit']) + '\n']
    for value in res_dict.values():
        csv_line.append(value['csv'])
    with open(filename, 'w') as f:
        f.writelines(csv_line)


if __name__ == '__main__':
    data_dir = './data/A100/layout16/'
    config_csv = load_result(data_dir + 'ncu_result.txt')
    store_csv(config_csv, data_dir + 'ncu_result_full.csv')
    pick_tile_csv(config_csv, 1, data_dir + 'ncu_result_tile1.csv')
    pick_best_csv(config_csv, data_dir + 'ncu_result_best.csv')

    data_dir = './data/V100/layout16/'
    config_csv = load_result(data_dir + 'ncu_result.txt')
    store_csv(config_csv, data_dir + 'ncu_result_full.csv')
    pick_tile_csv(config_csv, 1, data_dir + 'ncu_result_tile1.csv')
    pick_best_csv(config_csv, data_dir + 'ncu_result_best.csv')
