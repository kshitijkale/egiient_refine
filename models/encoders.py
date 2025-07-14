import numpy as np
from torch import einsum
from models.layers.subsample import furthest_point_sample, random_sample
from models.layers.group import KNNGroup, QueryAndGroup, get_aggregation_feautres
from models.layers.conv import  create_convblock2d,create_linearblock
from models.layers.local_aggregation import CHANNEL_MAP
from models.layers.norm import create_norm
import math
import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
    
def gram_matrix(tensor):
    gram = einsum('b i n, b n j  -> b i j', tensor.transpose(1,2),tensor)
    return gram

class img_tokenizer(nn.Module):
    def __init__(self, img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768,
                 norm_layer=None, 
                 flatten=True, 
                 bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # print(f"Image Tokenizer - Input shape: {x.shape}")
        x = self.proj(x)
        # print(f"Image Tokenizer - After projection: {x.shape}")
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # print(f"Image Tokenizer - After flatten and transpose: {x.shape}")
        x = self.norm(x)
        # print(f"Image Tokenizer - Final output: {x.shape}")
        return x, H, W

class pc_tokenizer(nn.Module):
    def __init__(self,
                 sample_ratio=0.0625,
                 scale=4,
                 group_size=32,
                 in_channels=3,
                 layers=4,
                 embed_dim=384,
                 subsample='fps',  # random, FPS
                 group='ballquery',
                 normalize_dp=False,
                 radius=0.1,
                 feature_type='dp_df',
                 relative_xyz=True,
                 norm_args={'norm': 'in2d'},
                 act_args={'act': 'relu'},
                 conv_args={'order': 'conv-norm-act'},
                 reduction='max',
                 return_group_points=False,
                 **kwargs
                 ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.group_size = group_size
        self.scale=scale
        self.feature_type = feature_type
        # subsample layer and group layer
        if subsample.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif 'random' in subsample.lower():
            self.sample_fn = random_sample

        self.group = group.lower()
        if 'ball' in self.group or 'query' in self.group:
            self.grouper = QueryAndGroup(nsample=self.group_size,
                                         relative_xyz=relative_xyz, normalize_dp=normalize_dp,
                                         radius=radius)
        elif 'knn' in self.group.lower():
            self.grouper = KNNGroup(self.group_size, relative_xyz=relative_xyz, normalize_dp=normalize_dp)
        else:
            raise NotImplementedError(f'{self.group.lower()} is not implemented. Only support ballquery, knn')

        # stages
        stages = int(math.log(1/sample_ratio, scale))
        embed_dim = int(embed_dim // 2 ** (stages-1))
        self.convs = nn.ModuleList()
        self.channel_list = [in_channels]
        for _ in range(int(stages)):
            # convolutions
            channels = [CHANNEL_MAP[feature_type](in_channels)] + [embed_dim] * (layers // 2) + [embed_dim * 2] * (
                    layers // 2 - 1) + [embed_dim]
            conv1 = []
            for i in range(layers // 2):
                conv1.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args if i!=(layers//2-1) else None,
                                                act_args=act_args if i!=(layers//2-1) else None,
                                                **conv_args))
            conv1 = nn.Sequential(*conv1)

            channels[layers // 2] *= 2
            conv2 = []
            for i in range(layers // 2, layers):
                conv2.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=act_args,
                                                **conv_args
                                                ))
            conv2 = nn.Sequential(*conv2)
            self.convs.append(nn.ModuleList([conv1, conv2]))

            self.channel_list.append(embed_dim)
            in_channels = embed_dim
            embed_dim *= 2

        # reduction layer
        if reduction in ['mean', 'avg', 'meanpool', 'avgpool']:
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=True)
        else:
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        self.out_channels = channels[-1]
        self.re_gr_p=return_group_points

    def forward(self, p, f=None):
        B, N, _ = p.shape[:3]
        # print(f"PC Tokenizer - Input shapes: p {p.shape}, f {f.shape if f is not None else 'None'}")
        out_p, out_f = [p], [f]
        
        for stage_idx, convs in enumerate(self.convs):
            # Progressive downsampling
            cur_p, cur_f = out_p[-1], out_f[-1]
            # print(f"PC Tokenizer - Stage {stage_idx} input: cur_p {cur_p.shape}, cur_f {cur_f.shape}")
            
            idx = self.sample_fn(cur_p, int(N //self.scale)).long()
            N = N // self.scale
            # print(f"PC Tokenizer - Stage {stage_idx} sampling: N reduced to {N}")
            
            center_p = torch.gather(cur_p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            center_f = torch.gather(cur_f, 2, idx.unsqueeze(1).expand(-1, cur_f.shape[1], -1))
            # print(f"PC Tokenizer - Stage {stage_idx} centers: center_p {center_p.shape}, center_f {center_f.shape}")

            # query neighbors.
            dp, fj = self.grouper(center_p, cur_p, cur_f)
            fj = get_aggregation_feautres(center_p, dp, center_f, fj, self.feature_type)
            # print(f"PC Tokenizer - Stage {stage_idx} after grouping: fj {fj.shape}")

            # graph convolutions
            fj = convs[0](fj)
            # print(f"PC Tokenizer - Stage {stage_idx} after conv1: fj {fj.shape}")
            
            fj = torch.cat(
                [self.pool(fj).expand(-1, -1, -1, self.group_size),
                fj],
                dim=1)
            # print(f"PC Tokenizer - Stage {stage_idx} after pooling and concat: fj {fj.shape}")

            # output
            stage_output = self.pool(convs[1](fj)).squeeze(-1)
            # print(f"PC Tokenizer - Stage {stage_idx} final output: {stage_output.shape}")
            out_f.append(stage_output)
            out_p.append(center_p)
            
        if self.re_gr_p:
            return out_p, out_f,dp
        else:
            return out_p, out_f

class transfer_loss_shared_encoder(nn.Module):
    def __init__(self,
                 embed_dim=192, 
                 block_head=12, 
                 depth=3,
                 img_patch_size=14,
                 sample_ratio=0.125, 
                 scale=2,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 pc_h_hidden_dim=384,
                 fuse_layer_num=1,
                 **kwargs
                 ):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.im_to_token=img_tokenizer(patch_size=img_patch_size,embed_dim=embed_dim)
        self.pc_to_token=pc_tokenizer(sample_ratio=sample_ratio,scale=scale,embed_dim=pc_h_hidden_dim)
        self.embed_dim=embed_dim
        self.fuse_layer_num=fuse_layer_num
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=block_head,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(depth)])
        
        self.cross_layer=nn.MultiheadAttention(embed_dim, block_head, batch_first=True)
        self.cross_norm=nn.LayerNorm(embed_dim)

        self.depth=depth
        self.pc_norm = create_norm(norm_args, self.embed_dim)

        self.im_norm = create_norm(norm_args, self.embed_dim)
        self.proj = nn.Linear(self.pc_to_token.out_channels, self.embed_dim)
        self.pc_pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=None, act_args=act_args),
            nn.Linear(128, self.embed_dim)
        )

    def forward(self,pc,im):
        B=pc.size(0)
        # print(f"Encoder - Input shapes: pc {pc.shape}, im {im.shape}")
        
        p_list,x_list=self.pc_to_token(pc,pc.transpose(1,2).contiguous())
        # print(f"Encoder - PC tokenizer output: {len(p_list)} point lists, {len(x_list)} feature lists")
        # print(f"Encoder - Final PC points shape: {p_list[-1].shape}")
        # print(f"Encoder - Final PC features before projection: {x_list[-1].shape}")

        cent_p,pc_f=p_list[-1],self.proj(x_list[-1].transpose(1, 2))
        # print(f"Encoder - After projection: pc_f {pc_f.shape}")
        
        im_f,_,_=self.im_to_token(im)
        # print(f"Encoder - Image tokenizer output: im_f {im_f.shape}")

        pc_pos_emd = self.pc_pos_embed(cent_p)
        # print(f"Encoder - Position embeddings: {pc_pos_emd.shape}")

        pc_f_list=[]
        im_f_list=[]

        for i in range(self.depth):
            pc_f = self.blocks[i](pc_f + pc_pos_emd)
            im_f = self.blocks[i](im_f)
            # print(f"Encoder - After transformer block {i}: pc_f {pc_f.shape}, im_f {im_f.shape}")
            pc_f_list.append(pc_f)
            im_f_list.append(im_f)
            
        pc_f=self.pc_norm(pc_f)
        im_f=self.im_norm(im_f)
        # print(f"Encoder - After normalization: pc_f {pc_f.shape}, im_f {im_f.shape}")
        
        x, _ = self.cross_layer(pc_f, im_f, pc_f)
        pc_f=self.cross_norm(pc_f+x)
        # print(f"Encoder - After cross attention: pc_f {pc_f.shape}")

        if self.fuse_layer_num==0:
            im_target_gram = gram_matrix(im_f_list[-1])
            pc_target_gram = gram_matrix(pc_f_list[-1])
            style_transfer_loss = torch.mean((im_target_gram - pc_target_gram) ** 2) / (self.embed_dim*256)
        else:
            im_target_gram = gram_matrix(im_f_list[-1])
            im_style_gram = gram_matrix(im_f_list[-(self.fuse_layer_num+1)])
            pc_target_gram = gram_matrix(pc_f_list[-1])
            pc_style_gram = gram_matrix(pc_f_list[-(self.fuse_layer_num+1)])
            im_style_loss = torch.mean((im_target_gram - pc_style_gram) ** 2)
            pc_style_loss = torch.mean((pc_target_gram - im_style_gram) ** 2)
            style_loss = (im_style_loss+pc_style_loss) / (self.embed_dim*256)#info
            pc_content_loss=torch.mean((pc_f_list[-1]-pc_f_list[-(self.fuse_layer_num+1)])**2)#str
            style_transfer_loss=pc_content_loss+style_loss
            
        # print(f"Encoder - Final output shapes: pc_f {pc_f.shape}, im_f {im_f.shape}, cent_p {cent_p.shape}")
        # print(f"Encoder - Style transfer loss: {style_transfer_loss}")
        
        return pc_f,im_f,cent_p,style_transfer_loss
           
if __name__ == '__main__':
    import time
    pc = torch.rand([4, 2048, 3]).cuda()
    img = torch.rand([4, 3, 224, 224]).cuda()
    model = transfer_loss_shared_encoder().cuda()
    s = time.time()
    x1,x2, p,s_l= model(pc,img)
    e = time.time()
    print(e-s)
    print(x1.shape)
    print(x2.shape)
    print(s_l)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")

