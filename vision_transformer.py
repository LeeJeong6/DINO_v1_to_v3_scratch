"""
기본 ViT

"""

import torch
import torch.nn as nn
import math
from functools import partial
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
import torchvision
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size)**2
        self.img_size = img_size
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self,x):
        B,C,H,W = x.shape
        out = self.proj(x).flatten(2).transpose(1,2) # (B, embed_dim, num_patch, num_patch)
        return out    

class MLP(nn.Module):
    def __init__(self, in_features, out_features = None, hidden_featrues = None, act_layers = nn.GELU, drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_featrues or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layers()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4) #3,B,num_heads, num_patch+1, head_dim        
        q,k,v = qkv[0], qkv[1], qkv[2] 

        attn = (q@k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim= - 1)
        attn = self.attn_drop(attn)

        x = (attn@v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn    


class Block(nn.Module):
    def __init__(self, dim, qkv_bias, qk_scale, attn_drop, mlp_ratio, num_heads, drop = 0., drop_path = 0.,norm_layer = nn.LayerNorm) :
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim = dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale, attn_drop = attn_drop, proj_drop = drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_featrues = mlp_hidden_dim, act_layers = nn.GELU)

    def forward(self,x,return_attention = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))    
        return x    

class VisionTransformer(nn.Module):
    def  __init__(self, img_size = 224, patch_size = 16, in_chans = 3, embed_dim = 768, drop_rate = 0., num_heads = 12, depth = 12, mlp_ratio = 4., qkv_bias = False, qk_scale = None,
                attn_drop_rate = 0.,drop_path_rate = 0., norm_layer = nn.LayerNorm, num_classes = 0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p = drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, qk_scale = qk_scale,
                  drop = drop_rate, attn_drop = attn_drop_rate, drop_path = dpr[i], norm_layer = norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0  else nn.Identity()

    def prepare_tokens(self,x):
        B,C,w,h = x.shape
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B,-1,-1) # B,1,E 
        x = torch.cat([cls_token, x], dim = 1) # B,n+1,E

        x = x + self.interpolate_pos_encoding(x,w,h)
        return self.pos_drop(x)

    def interpolate_pos_encoding(self,x,w,h):
        """
        만약 w!=h면 보간법을 사용한다

        """
        N = self.pos_embed.shape[1] - 1 # N
        if w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:,0] # 1,E
   
        patch_pos_embed = self.pos_embed[:, 1:] # 1,N,E
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size #w // patch_sizes
        h0 = h // self.patch_embed.patch_size #h // patch_size

        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate( 
            
            # w,h가 다르니까 interpolate함수로 img_size // patch_size 로 크기를 맞춰주기
            # scale_factor가 이전에 (patch_size x patch_size)로 고정되어있던 pos_embed를 그 비율만큼 늘려주는 역할
            # interpolate의 출력은 B,E,p,p이기 때문에 출력 결과를 다시 B,p,p,E로 맞추는 과정이 필요함
            # B,patch_size,patch_size,E -> #B,E,p,p -> B,p,p,E
            input = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0,3,1,2),
            scale_factor = (w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode = "bicubic"
        )
        
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0,2,3,1).view(1,-1,dim) # 1,num_patch, E

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)



    def forward(self,x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) 
        return x[:, 0]    

def vit_tiny(patch_size = 16, **kwargs):
    print("tiny model")
    model = VisionTransformer(
        patch_size = patch_size, embed_dim = 192, depth = 12, num_heads = 3, mlp_ratio = 4, 
        qkv_bias = True, norm_layer = partial(nn.LayerNorm, eps = 1e-6), **kwargs
    )
    return model

def vit_small(patch_size=16, **kwargs):
    print("small model")
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    print("base model")
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn = False, norm_last_layer = True, nlayer = 3, hidden_dim = 2048, bottleneck_dim = 256):
        super().__init__()
        nlayers = max(nlayer,1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else : 
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn : 
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(n_layers-2):
                layers.append(nn.Linear(hidden_dim,hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU)
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        # self.mlp는 이제 위 layer들을 쌓은 탑이 됨
        # 처음 가중치는 랜덤일텐데 apply함수로 _init_weights에서 정의한 변수들로 초기화가 됨    
        self.apply(self.__init__weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias = False))            
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer :
            self.last_layer.weight_g.requires_grad = False

    def __init__weights(self,m):
        if isinstance(m,nn.Linear):
            # 이 실험은 prac에 있음
            # nn.Linear에 존재하는 파라미터를 apply함수를 통해 파라미터를 내가 원하는 값으로 설정이 가능함
            trunc_normal_(m.weight, std = .02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p = 2)
        x = self.last_layer(x)
        return x
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # x = torch.randn(16,3,224,224).to(device)
    vb = vit_base().to(device)
    # vb(x)
    # vt = vit_tiny().to(device)
    # vt(x)
    # vs = vit_small().to(device)
    # vs(x)
    train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(root = "/mnt/hdd_6tb/ImageNet/ILSVRC2012_img_val/", transform = train_transform)
    data_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

    for data in data_loader : 
        image, label = data
        image = image.to(device)
        label = label.to(device)
        vb(image)





