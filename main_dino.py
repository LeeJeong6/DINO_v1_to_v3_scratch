import argparse
import os
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import torch.nn.functional as F
def get_args_parser():

    parser = argparse.ArgumentParser(description = "DINO", add_help = False)

    parser.add_argument("--arch", default = "vit_small", type = str, 
        choices = ["vit_tiny", "vit_small", "vit_base"])
    parser.add_argument("--patch_size", default = 16, type = int)
    parser.add_argument("--out_dim", default = 65536, type = int)
    parser.add_argument("--output_dir", default = ".", type = str)
    return parser

def train_dino(args):
    return 

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp = 0.1, center_momentum = 0.9):
        """
        하다가 중간에 그만둠 왜냐면 데이터aug랑 불러오는걸 먼저 해야할거같기 때문임 
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            # num만큼 초반 스케쥴러를 도입하는 방식
            np.linspace(start = warmup_teacher_temp,
                        stop = teacher_temp, 
                        num = warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self,student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops) # chunk : ncrops개로 리스트를 분할

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim = -1)
        teacher_out = teacher_out.detach().chunk(2) #이거 왜 2지?

        total_loss = 0
        n_loss_terms = 0
        for iq,q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v ==iq :
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim= -1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
                
        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents = True, exist_ok = True) #parents는 상위 경로가 없으면 생성해줌, exist_ok는 이미 존재해도 패스
    train_dino(args)
    # dino_loss = DINOLoss(
    #     65536,
    #     8 + 2,  # total number of crops = 2 global crops + local_crops_number
    #     0.04,
    #     0.04,
    #     0,
    #     100,
    # )

