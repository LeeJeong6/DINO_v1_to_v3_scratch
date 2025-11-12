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
