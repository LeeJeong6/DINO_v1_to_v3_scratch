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

import utils
import vision_transformer as vits
import matplotlib.pyplot as plt
def get_args_parser():

    parser = argparse.ArgumentParser(description = "DINO", add_help = False)

    parser.add_argument("--arch", default = "vit_small", type = str, 
        choices = ["vit_tiny", "vit_small", "vit_base"])
    parser.add_argument("--patch_size", default = 16, type = int)
    parser.add_argument("--out_dim", default = 65536, type = int)
    parser.add_argument("--output_dir", default = ".", type = str)
    parser.add_argument("--batch_size_per_gpu", default = 16, type = int)
    parser.add_argument("--num_workers", default = 1, type = int)
    parser.add_argument("--global_crops_scale", default = (0.4, 1.), type = float)
    parser.add_argument("--local_crops_number", default = 7, type = int)
    parser.add_argument("--crops_scale", default = (0.05, 0.4), type = float)

    return parser

def train_dino(args):
    # utils.init_distributed_mode(args)
    transform = DataAugmentationDINO(
        global_crops_scale = args.global_crops_scale, 
        local_crops_number = args.local_crops_number, 
        local_crops_scale = args.crops_scale)

    dataset = datasets.ImageFolder(root = "/mnt/hdd_6tb/ImageNet/ILSVRC2012_img_val/", transform = transform)
    
    # sampler = torch.utils.data.DistributedSampler(datasets, shuffle = True) # gpu분산학습을 위해서 이걸 도입함
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size_per_gpu,
        
        # sampler = sampler,
        num_workers = args.num_workers,
        pin_memory = True, # cpu->gpu로 이동이 빠르다. 참고 : https://velog.io/@smuhyeon/Pytorch-Dataloader-pinmemory%EC%84%A4%EC%A0%95
        drop_last = True
    )
    print(f"Data loaded : there are {len(dataset)} images.")

    # image, label = next(iter(data_loader))
    # img = image[0][5]
    # tensor_imshow(img)

    
    return 
def tensor_imshow(img):
    
    image = img.permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max()-image.min())
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.savefig("aug_ex1.png")
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
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self,teacher_output):
        batch_center= torch.sum(teacher_output, dim = 0, keepdim = True)
        # dist.all_reduce(batch_center) # 분산학습에서 gpu간 데이터 통신에 개선할 수 있는 방법 / 참고 : https://algopoolja.tistory.com/95
        batch_center = batch_center * self.center_momentum + batch_center * (1 - self.center_momentum)    
                
        
class DataAugmentationDINO():
    # 원본 코드에서는 object를 상속받는데, 검색해보니 파이썬 3부터는 아무런 의미가 없음
    def __init__(self, global_crops_scale, local_crops_number, local_crops_scale):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5), # p확률로 좌우반전
            transforms.RandomApply(
                [transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.2, hue = 0.1)], #밝기, 대비, 채도, 색조
                p = 0.8
            ),
            transforms.RandomGrayscale(p = 0.2) # p확률로 흑백 변환 
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_trans1 = transforms.Compose([
            transforms.RandomResizedCrop(224,scale = global_crops_scale, interpolation = Image.BICUBIC), #scale*100%로 면적의 비율을 조정, interpolation은 크기를 어떻게 보간할건지 정하는 듯함
            flip_and_color_jitter,
            utils.GaussianBlur(p = 1.0),
            normalize,
        ])
        self.global_trans2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p = 0.1),
            utils.Solarization(p = 0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_trans = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p = 0.5),
            normalize,
        ])

        
    def __call__(self, image):
        # __call__은 클래스를 함수처럼 사용하는 용도로 클래스 내 함수를 호출할 때 객체에 바로 할당하면 바로 실행됨
        # 원본 크기 이미지가 2개, 크롭 이미지 args만큼 return 함
        crops = []
        crops.append(self.global_trans1(image))
        crops.append(self.global_trans2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_trans(image)) 
        return crops 

        
    
    
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

