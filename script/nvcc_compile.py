from multiprocessing import Pool
import itertools, subprocess, argparse
from param import FLAG_DICT, COMMON_SOURCE, BASELINE_SOURCE, TENSOR_SOURCE, BINARY_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='A100')
args = parser.parse_args()

BASE_FLAGS = '-O3 -arch sm_80 -DRUN_TIMES=1' if args.arch == 'A100' else '-O3 -arch sm_70 -DRUN_TIMES=1'


def compile_command_gen(cuda_compute, stencil_size, mesh_size, tile_size):
    cmd = ' '.join(['nvcc', BASE_FLAGS, f'-DMESH_SIZE={mesh_size},TILE_SIZE={tile_size}'])
    output_file_name = f'{cuda_compute}_{stencil_size}_{mesh_size}_{tile_size}'
    if cuda_compute == 'baseline':
        cmd = ' '.join([cmd, '-o', BINARY_DIR + output_file_name,
                        COMMON_SOURCE[stencil_size],
                        BASELINE_SOURCE[stencil_size],
                        ])
    else:
        cmd = ' '.join([cmd, '-o', BINARY_DIR + output_file_name,
                        COMMON_SOURCE[stencil_size],
                        TENSOR_SOURCE[stencil_size],
                        ])
    

    return cmd, output_file_name

def exec_cmd(cmd, wait=False):
    p = subprocess.Popen(cmd.split(' '))
    if wait:
        p.wait()

def exec_cmd_list_async(cmd_list):
    pool = Pool(24)
    for cmd in cmd_list:
        pool.apply_async(exec_cmd, args=(cmd, True))
    pool.close()
    pool.join()


def run_file_serial(file_list):
    example_cmd = 'bash script/run_file_with_ncu.sh {file_name} {output_file} {BIN_DIR}'

    for index, file_name in enumerate(file_list):
        print('{index}/{total} {file_name}'.format(index=index,
                                                   total=len(file_list), file_name=file_name))
        cmd = example_cmd.format(
            file_name=file_name, output_file='./data/{}/brick/ncu_result.txt'.format(args.arch),
            BIN_DIR=BINARY_DIR)
        proc = subprocess.Popen(cmd.split(' '))
        proc.wait()


if __name__ == '__main__':
    compile_cmd_list = []
    exec_files_list = []
    for cuda_compute, stencil_size, mesh_size, tile_size in itertools.product(FLAG_DICT['cuda_compute'], FLAG_DICT['stencil_size'], 
                                                                            FLAG_DICT['mesh_size'], FLAG_DICT['tile_size']):
        if (mesh_size // 16) % tile_size != 0:
            continue
        cmd, output_file_name = compile_command_gen(cuda_compute, stencil_size, mesh_size, tile_size)
        compile_cmd_list.append(cmd)
        exec_files_list.append(output_file_name)

    exec_cmd_list_async(compile_cmd_list)
    run_file_serial(exec_files_list)
