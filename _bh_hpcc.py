import os


if __name__ == '__main__':
    code = [
        '_av_CTE.py',
        '',
        '',
    ]
    run = [
        '/Share/apps/singularity/bin/singularity exec ~/tf.sif python3 -u',
        '~/anaconda3/envs/tf/bin/python -u',
        '',
    ]
    srun = [
        '-p GPU8 --gres=gpu:1',
        '-p GPU8',
        '-p GPU8 -w gpu02',
    ]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    command = 'srun -J hi -u ' + srun[0] + ' ' + run[0] + ' ' + code[0]
    print(command)
    os.system(command)
