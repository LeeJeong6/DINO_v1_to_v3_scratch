import torch
import torch.distributed as dist
import os

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

def run_ddp_example(rank, world_size):
    # 1. 초기화: 분산 환경을 설정합니다.
    # 'env://'는 환경 변수에서 통신 설정을 읽어온다는 의미입니다.
    dist.init_process_group("nccl", rank=rank, world_size=world_size) 
    
    # 2. 정보 출력
    print(f"👋 Hello from Global Rank: {rank} / World Size: {world_size}")
    
    # os.environ["LOCAL_RANK"]는 torchrun/torch.distributed.launch가 설정해주는 환경 변수입니다.
    local_rank = os.environ.get("LOCAL_RANK")
    print(f"🌍 Running on Node/Local Rank: {local_rank}")
    
    # 3. GPU 할당: 각 프로세스는 자신의 local rank에 해당하는 GPU를 사용합니다.
    device = torch.device(f"cuda:{local_rank}")
    
    # 4. 텐서 연산
    # 각 프로세스는 rank에 해당하는 값을 가진 텐서를 생성합니다.
    tensor = torch.tensor([rank * 10.0]).to(device)
    print(f"Rank {rank} initial tensor: {tensor}")
    
    # 5. Allreduce: 모든 프로세스의 텐서를 합산합니다.
    # 모든 프로세스가 동기적으로 참여하여 결과를 공유합니다.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Rank {rank} after Allreduce (SUM): {tensor}")
    
    # 6. 분산 환경 종료
    dist.destroy_process_group()

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # gpu간 통일성 
    np.random.seed(seed)

def setup_for_distributed(is_master):
    """
    log가 혼동되지 않게 0번 GPU에서만 프린트되게끔 하는 역할
    분산 학습에서는 필수적인 전처리 과정임
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])

    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(backend = "nccl", 
                            init_method="env://",
                            rank=rank, 
                            world_size=world_size) 
    torch.cuda.set_device(gpu)                        
    print('| distributed init (rank {}): {}'.format(
        rank, "env://"), flush=True)
    dist.barrier()
    setup_for_distributed(rank == 0) 

def get_sha():
    # 깃 저장소가 아니면 그냥 pass임 
    cwd = os.path.dirname(os.path.abspath(__file__)) #abspath로 절대경로 읽어서 현재 디렉토리 설정

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message
def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')


    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()

    init_distributed_mode()
    fix_random_seeds(31)
    print("git:\n  {}\n".format(get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True # 이걸 True로 하면 현재 하드웨어에서 가장 빠른 것을 선택한다고 함
    
     