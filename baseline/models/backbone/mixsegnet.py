'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
* thanks to https://github.com/lucidrains/mlp-mixer-pytorch
'''
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops.layers.torch import Rearrange

from baseline.models.registry import BACKBONE

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

@BACKBONE.register_module
class MixSegNet(nn.Module):
    def __init__(self,
                image_size=144,
                channels=64,
                patch_size=8,
                dim=512,
                depth=5,
                output_channels=1024,
                expansion_factor=4,
                dropout=0.,
                cfg=None,
                is_using_convolution_pooling = False):
        super(MixSegNet, self).__init__()
        self.cfg=cfg
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        out_img_size = int(image_size/patch_size)
        out_in_channels = int(dim/(patch_size*patch_size))
        
        if isinstance( image_size , list ) :
            image_height = image_size[0]
            image_width = image_size[1]
            
        else :
            image_height, image_width = image_size , image_size #pair(image_size)

            self.image_height = image_height
            self.image_width = image_width

        if isinstance( patch_size , list ) :
            patch_height = patch_size[0]
            patch_width = patch_size[1]
        else :
            patch_height, patch_width = patch_size , patch_size #pair(patch_size)

        self.mixsegnet = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = out_img_size, p1 = patch_size, p2 = patch_size),
            nn.Conv2d(in_channels=out_in_channels, out_channels=output_channels, kernel_size=1),
        )
        
        self.is_using_convolution_pooling = is_using_convolution_pooling

        self.convolution_pooling = None

    def forward(self, x, is_get_features=False):
    
        if self.is_using_convolution_pooling == True :
        
        

            
            
            img = x
            
            if not self.convolution_pooling :

                
                
                self.number_of_strides_height = int( img.shape[2]/ self.image_height)
                self.number_of_strides_width = int( img.shape[3]/ self.image_width )

                #print( "Stride of Convolutional Pooling is : " + str( self.number_of_strides ))

                self.convolution_pooling = nn.Conv2d( img.shape[1], img.shape[1], kernel_size=[self.number_of_strides_height , self.number_of_strides_width] ,     stride=[self.number_of_strides_height , self.number_of_strides_width]).cuda( int( self.cfg.gpus_ids.split("'")[0]))

            img = self.convolution_pooling( img.cuda( int( self.cfg.gpus_ids.split("'")[0])) )
            
            x = img
            
        out = self.mixsegnet(x)

        if is_get_features:
            list_feature = []
            list_feature.append(x)
            for i in range(len(self.mixsegnet)):
                x = self.mixsegnet[i](x)
                list_feature.append(x)
            return x, list_feature
        
        return out
        
