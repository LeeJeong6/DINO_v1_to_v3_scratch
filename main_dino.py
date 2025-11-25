import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from tqdm import tqdm,trange

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
from vision_transformer import DINOHead
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
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--norm_last_layer', default=True, type=bool)
    parser.add_argument('--use_bn_in_head', default=False, type=bool)

    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--use_fp16', type = bool, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")    
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")        
  
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    transform = DataAugmentationDINO(
        global_crops_scale = args.global_crops_scale, 
        local_crops_number = args.local_crops_number, 
        local_crops_scale = args.crops_scale)

    dataset = datasets.ImageFolder(root = "/mnt/hdd_6tb/ImageNet/ILSVRC2012_img_train/", transform = transform)
    
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle = True) # gpu분산학습을 위해서 이걸 도입함
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.batch_size_per_gpu,
        
        sampler = sampler,
        num_workers = args.num_workers,
        pin_memory = True, # cpu->gpu로 이동이 빠르다. 참고 : https://velog.io/@smuhyeon/Pytorch-Dataloader-pinmemory%EC%84%A4%EC%A0%95
        drop_last = True
    )
    print(f"Data loaded : there are {len(dataset)} images.")

    # image, label = next(iter(data_loader))
    # img = image[0][5]
    # tensor_imshow(img)
    
    student = vits.__dict__[args.arch](patch_size = args.patch_size, drop_path_rate = args.drop_path_rate) # 파일.__dict__.keys()하면 그 파일에 존재하는 클래스,함수를 모두 보여줌, 그 중 내가 원하는 아키텍쳐를 선택해서 이런식으로 객체를 생성한다
    teacher = vits.__dict__[args.arch](patch_size = args.patch_size) #drop_path_rate는 teacher는 디폴트인 0을 선택, student는 0.1로 하는 듯 하다
    embed_dim = student.embed_dim


    # MultiCropWrapper가 backbone과 head를 이어주는 역할을 함
    # student와 teacher는 일반적인 backbone이었는데, 이제 head를 붙여서
    # student는 MultiCropWrapper의 객체가 됐고 그 안에는 backbone : vit, head : Dinohead가 됨
    # 그렇기 때문에 당연히 DINOHead의 in_dim은 vit의 출력 dim인 embed_dim이 되겠다 

    student = utils.MultiCropWrapper(student, DINOHead(in_dim = embed_dim, out_dim = args.out_dim, use_bn = args.use_bn_in_head, norm_last_layer = args.norm_last_layer)) #, nlayer = , hidden_dim = ,bottleneck_dim = ))
    teacher = utils.MultiCropWrapper(teacher, DINOHead(in_dim = embed_dim, out_dim = args.out_dim, use_bn = args.use_bn_in_head)) #teacher를 다르게 하려는건가,,, 둘 다 default가 True임..뭐지?

    # student, teacher = student.to(args.device), teacher.to(args.device)
    student, teacher = student.cuda(), teacher.cuda()


    print("student : ", sum(p.numel() for p in student.parameters() if p.requires_grad))
    print("teacher : ", sum(p.numel() for p in teacher.parameters() if p.requires_grad))
    
    student = nn.parallel.DistributedDataParallel(student, device_ids = [args.gpu])
    teacher_without_ddp = teacher
    teacher_without_ddp.load_state_dict(student.module.state_dict()) #dataparallel을 사용할거면 module.state_dict()
    # 참고 : https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    dino_loss = DINOLoss(
        out_dim = args.out_dim,
        ncrops = args.local_crops_number + 2,
        warmup_teacher_temp = args.warmup_teacher_temp,
        teacher_temp = args.teacher_temp,
        warmup_teacher_temp_epochs = args.warmup_teacher_temp_epochs,
        nepochs = args.epochs
    ).cuda()

    fp16_scaler = None #mixed precision으로 fp32 -> fp16으로 변경해서 메모리 효율 등의 장점이 있다.
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    param_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(param_groups) 

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in trange(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()    
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth')) 
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # running_loss = 0.0
    # for it,(images, labels) in enumerate(tqdm(data_loader,desc = "BATCH")):
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it
        for i,param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0 :
                param_group["weight_decay"] =wd_schedule[it]

        images = [im.cuda(non_blocking = True) for im in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping traning".format(loss.item()), force = True)    
            sys.exit(1)
        optimizer.zero_grad()
        
        fp16_scaler.scale(loss).backward()
        utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
        fp16_scaler.step(optimizer)
        fp16_scaler.update()

        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])        
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()    
    print("Averaged stats:", metric_logger)
    # torch.save(student.state_dict(), "./ImageNet/student.pt")
    # torch.save(teacher.state_dict(), "./ImageNet/teacher.pt")
    # return running_loss / len(data_loader)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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
        dist.all_reduce(batch_center) # 분산학습에서 gpu간 데이터 통신에 개선할 수 있는 방법 / 참고 : https://algopoolja.tistory.com/95
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        # batch_center = batch_center * self.center_momentum + batch_center * (1 - self.center_momentum)    
                
        
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
