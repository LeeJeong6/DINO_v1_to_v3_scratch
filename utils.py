import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps

class GaussianBlur():
    """ 
    참고 : https://www.geeksforgeeks.org/python/python-pil-gaussianblur-method/
    가우시안 필터로 이미지를 흐리게 함
    이때 radius값이 클수록 강하게 블러됨
    """
    def __init__(self, p = 0.5, radius_min = 0.1, radius_max = 2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob    
        if not do_it:
            return img
        return img.filter(

            ImageFilter.GaussianBlur(
                # random.uniform은 a보다 크고 b보다 작은 소수를 반환함
                radius = random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization():
    """
    참고 : https://www.geeksforgeeks.org/python/python-pil-imageops-solarize-method/
    threshold 이상의 모든 픽셀 값을 반전하는 역할 
    디폴트는 128임
    """
    def __init__(self, p):
        self.prob = p

    def __call__(self, img):
        if random.random() < self.prob :
            return ImageOps.solarize(img)
        else :
            return img       
class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        # backbone은 student와 teacher
        # head는 DINOHEAD
         
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        # teacher는 [global view 1 , global view 2]
        # -> 각각 B,3,224,224임
        # student는 [global view 1 , global view 2, local view 1, local view2 ,,,,,]
        # -> 각각 B,3,224,224 후에 B,3,96,96
        if not isinstance(x, list):
            x = [x]
        each_img_h = torch.tensor([inp.shape[-1] for inp in x])   
        output, counts = torch.unique_consecutive(each_img_h, return_counts = True)
        # unique_consecutive는 입력에 (고유한 스칼라 요소를 , 발생횟수)를 return함
        # student의 경우, (tensor([224,  96]), tensor([2, n_crops]))
        # 참고 : https://runebook.dev/ko/docs/pytorch/generated/torch.unique_consecutive
        idx_crops = torch.cumsum(counts, dim = 0)
        
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # teacher의 경우 global view에 대해 보는거고
            # student의 경우 x[0:2] -> global view , x[2:9] -> local view
            # (B*2,3,224,224) 또는 (B*ncrops,3,96,96)
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

def get_params_groups(model):
    """
    참고 : https://soundprovider.tistory.com/entry/pytorch-torch%EC%97%90%EC%84%9C-parameter-%EC%A0%91%EA%B7%BC%ED%95%98%EA%B8%B0
    named_paramters()는 튜플로 weight : tensor ~~~, bias : tensor~~~ 이런식으로 name,param을 준다
    AdamW (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.001
            maximize: False
            weight_decay: 0.01

        Parameter Group 1
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.001
            maximize: False
            weight_decay: 0.0
        )
    이렇게 파라미터를 나눠서 group0, group1 에 서로 다른 weight decay를 준다 왜그럴까?
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        # cls token, pos_emb, proj 등 param이 name으로, 해당 weight는 param으로
        if not param.requires_grad:
            # DINOHead의 last_norm_layer만 필요없는 듯 
            continue
        if name.endswith('.bias') or len(param.shape) == 1:
            # print("bias나 길이가 1이야")
            # print(name, param.shape)
            not_regularized.append(param)
        else : 
            # print("정상")
            # print(name, param.shape)        
            regularized.append(param)
    return [{"params":regularized},{"params":not_regularized, "weight_decay":0.}] #55,102

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
    