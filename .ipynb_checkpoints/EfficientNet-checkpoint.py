import os
import numpy as np
import torch
from torch import nn
import utils
import torch.nn.functional as F
from math import sqrt
import math
from layers import conv_bn_act
from layers import SamePadConv2d
from layers import Flatten
from layers import SEModule
from layers import DropConnect
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# EfficientNet
########################################################################################################################
class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            SamePadConv2d(mid_, out_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_, 1e-3, 0.01)
        )

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        # self.dropconnect = DropConnect(dc_ratio) if dc_ratio > 0 else nn.Identity()
        # Original TF Repo not using drop_rate
        # https://github.com/tensorflow/tpu/blob/05f7b15cdf0ae36bac84beb4aef0a09983ce8f66/models/official/efficientnet/efficientnet_model.py#L408
        self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 n_classes=1000):
        super().__init__()
        min_depth = min_depth or depth_div
        
        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.stem = conv_bn_act(3, renew_ch(32), kernel_size=3, stride=2, bias=False)
        
        self.blocks = nn.Sequential(
            #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        )
        
        self.stage0 = MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        self.stage1 = MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate)
        self.stage2 = MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, 0.25, drop_connect_rate)
        self.stage3 = MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate)
        self.stage4 = MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate)
        self.stage5 = MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate)
        self.stage6 = MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)

        self.head = nn.Sequential(
            *conv_bn_act(renew_ch(320), renew_ch(1280), kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
            Flatten(),
            nn.Linear(renew_ch(1280), n_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, SamePadConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def forward(self, inputs):
        stem = self.stem(inputs)
        #x = self.blocks(stem)
        #head = self.head(x)
        #return head
        x0 = self.stage0(stem) # 8, 16, 176, 176
        x1 = self.stage1(x0)   # 8, 24, 88, 88
        x2 = self.stage2(x1)   # 8, 40, 44, 44
        x3 = self.stage3(x2)   # 8, 80, 22, 22
        x4 = self.stage4(x3)   # 8, 112, 22, 22
        x5 = self.stage5(x4)   # 8, 192, 11, 11
        x6 = self.stage6(x5)   # 8, 320, 11, 11
        return x2,x4,x6
########################################################################################################################


# predefined the model components for connecting with SSD detector and segmentation
########################################################################################################################
class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and ReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, padding=None):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        if padding == None:
            padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and ReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class R(nn.Module):
    '''
        This class for ReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.act(input)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        #combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output

class C3BlockB(nn.Module):
    '''
    '''
    def __init__(self, nIn, nOut):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        '''
        super().__init__()
        self.c3 = CBR(nIn, nOut, 3, 1)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.c3(input)
        return output

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input
########################################################################################################################


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the base model
        
        #self.extra_conv1a = CBR(512, 256, 1, 1)
        # self.extra_conv1b = DepthWiseBlock(256, 256, 2, 0)  # dim. reduction because stride > 1
        #self.extra_conv1b = CBR(256, 256, 3, 2, 0)  # dim. reduction because stride > 1

        # self.extra_conv2b = DepthWiseBlock(256, 256, 1, 0)  # dim. reduction because padding = 0
        #self.extra_conv2b = CBR(256, 256, 3, 1, 0)  # dim. reduction because padding = 0

        # self.extra_conv3b = DepthWiseBlock(256, 256, 1, 0)  # dim. reduction because padding = 0
        #self.extra_conv3b = CBR(256, 256, 3, 1, 0)  # dim. reduction because padding = 0

        self.extra_conv1a = CBR(320, 160, 1, 1)
        self.extra_conv1b = CBR(160, 160, 3, 2, 0)  # dim. reduction because stride > 1
        self.extra_conv2b = CBR(160, 160, 3, 1, 0)  # dim. reduction because padding = 0
        self.extra_conv3b = CBR(160, 160, 3, 1, 0)  # dim. reduction because padding = 0
        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)

    def forward(self, feats_32s):
        """
        Forward propagation.

        :param feats_32s: lower-level 32s (1/32) feature map, a tensor of dimensions (N, C, 11, 11)
        :return: higher-level feature maps extra_conv1b_feats, extra_conv2b_feats, extra_conv3b_feats
        """

        out = self.extra_conv1a(feats_32s)  # (N, C, 11, 11)
        out = self.extra_conv1b(out)  # (N, C, 5, 5)
        extra_conv1b_feats = out  # (N, C, 5, 5)

        out = self.extra_conv2b(out)  # (N, C, 3, 3)
        extra_conv2b_feats = out  # (N, C, 3, 3)

        extra_conv3b_feats = self.extra_conv3b(out)  # (N, C, 1, 1)

        # Higher-level feature maps
        return extra_conv1b_feats, extra_conv2b_feats, extra_conv3b_feats

class PredictionConvolutions_Upsample_NN(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the pre-calculated prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the pre-calculated bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions_Upsample_NN, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'base_8s': 4,
                   'base_16s': 6,
                   'base_32s': 6,
                   'extra_conv1b': 6,
                   'extra_conv2b': 4,
                   'extra_conv3b': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # for 8s detection
        #self.base_8s_CBR = CBR(512, 256//2, 1)
        self.base_8s_CBR = CBR(96, 48//2, 1)
        # self.base_8s_C3BlockB = C3BlockB(256, 256 // 2)

        # for 16s detection
        #self.base_16s_CBR = CBR(512, 256, 1)
        self.base_16s_CBR = CBR(112, 56, 1)
        
        # self.base_16s_DepDown = DepthWiseBlock(256//2, 256//2, 2, 1)
        #self.base_16s_DepDown = CBR(256//2, 256//2, 3, 2, 1)
        self.base_16s_DepDown = CBR(24, 24, 3, 2, 1)
        # self.base_16s_C3BlockB = C3BlockB(256 // 2, 256 // 2)

        # for 32s detection
        #self.base_32s_CBR = CBR(512, 256//2, 1)
        self.base_32s_CBR = CBR(320, 160//2, 1)
        # self.base_32s_C3BlockB = C3BlockB(256, 256 // 2)

        # for extra detection
        # self.extra_conv1b_C3BlockB = C3BlockB(256, 256 // 2)
        # self.extra_conv2b_C3BlockB = C3BlockB(256, 256 // 2)
        # self.extra_conv3b_C3BlockB = C3BlockB(256, 256 // 2)
        #self.extra_conv1b_CBR = CBR(256, 256 // 2, 1)
        #self.extra_conv2b_CBR = CBR(256, 256 // 2, 1)
        #self.extra_conv3b_CBR = CBR(256, 256 // 2, 1)
        self.extra_conv1b_CBR = CBR(160, 160 // 2, 1)
        self.extra_conv2b_CBR = CBR(160, 160 // 2, 1)
        self.extra_conv3b_CBR = CBR(160, 160 // 2, 1)

        # for segmentation
        # self.sample = InputProjectionA(3)
        # self.base_8s_CBR_seg = CBR(256//2+3, 128, 1)
        # self.base_8s_C3BlockB_seg = C3BlockB(128, 128 // 2)
        # self.base_8s_seg_pred = nn.Conv2d(128//2, 2, kernel_size=1, padding=0)
        # self.upsample_seg_pred = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=False)

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        #self.loc_base_8s = nn.Conv2d(256//2, n_boxes['base_8s'] * 4, kernel_size=3, padding=1)
        #self.loc_base_16s = nn.Conv2d(256//2, n_boxes['base_16s'] * 4, kernel_size=3, padding=1)
        #self.loc_base_32s = nn.Conv2d(256//2, n_boxes['base_32s'] * 4, kernel_size=3, padding=1)
        #self.loc_extra_conv1b = nn.Conv2d(256//2, n_boxes['extra_conv1b'] * 4, kernel_size=3, padding=1)
        #self.loc_extra_conv2b = nn.Conv2d(256//2, n_boxes['extra_conv2b'] * 4, kernel_size=3, padding=1)
        #self.loc_extra_conv3b = nn.Conv2d(256//2, n_boxes['extra_conv3b'] * 4, kernel_size=3, padding=1)
        
        self.loc_base_8s = nn.Conv2d(24, n_boxes['base_8s'] * 4, kernel_size=3, padding=1)
        self.loc_base_16s = nn.Conv2d(24, n_boxes['base_16s'] * 4, kernel_size=3, padding=1)
        self.loc_base_32s = nn.Conv2d(80, n_boxes['base_32s'] * 4, kernel_size=3, padding=1)
        self.loc_extra_conv1b = nn.Conv2d(80, n_boxes['extra_conv1b'] * 4, kernel_size=3, padding=1)
        self.loc_extra_conv2b = nn.Conv2d(80, n_boxes['extra_conv2b'] * 4, kernel_size=3, padding=1)
        self.loc_extra_conv3b = nn.Conv2d(80, n_boxes['extra_conv3b'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        #self.cl_base_8s_ = nn.Conv2d(256//2, n_boxes['base_8s'] * n_classes, kernel_size=3, padding=1)
        #self.cl_base_16s_ = nn.Conv2d(256//2, n_boxes['base_16s'] * n_classes, kernel_size=3, padding=1)
        #self.cl_base_32s_ = nn.Conv2d(256//2, n_boxes['base_32s'] * n_classes, kernel_size=3, padding=1)
        #self.cl_extra_conv1b_ = nn.Conv2d(256//2, n_boxes['extra_conv1b'] * n_classes, kernel_size=3, padding=1)
        #self.cl_extra_conv2b_ = nn.Conv2d(256//2, n_boxes['extra_conv2b'] * n_classes, kernel_size=3, padding=1)
        #self.cl_extra_conv3b_ = nn.Conv2d(256//2, n_boxes['extra_conv3b'] * n_classes, kernel_size=3, padding=1)

        
        self.cl_base_8s_ = nn.Conv2d(24, n_boxes['base_8s'] * n_classes, kernel_size=3, padding=1)
        self.cl_base_16s_ = nn.Conv2d(24, n_boxes['base_16s'] * n_classes, kernel_size=3, padding=1)
        self.cl_base_32s_ = nn.Conv2d(80, n_boxes['base_32s'] * n_classes, kernel_size=3, padding=1)
        self.cl_extra_conv1b_ = nn.Conv2d(80, n_boxes['extra_conv1b'] * n_classes, kernel_size=3, padding=1)
        self.cl_extra_conv2b_ = nn.Conv2d(80, n_boxes['extra_conv2b'] * n_classes, kernel_size=3, padding=1)
        self.cl_extra_conv3b_ = nn.Conv2d(80, n_boxes['extra_conv3b'] * n_classes, kernel_size=3, padding=1)
        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.modules():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, image, base_8s_feats, base_16s_feats, base_32s_feats,
                extra_conv1b_feats, extra_conv2b_feats, extra_conv3b_feats):
        """
        Forward propagation.

        :param base_8s_feats: base_8s feature map, a tensor of dimensions (N, C, 44, 44)
        :param base_16s_feats: base_16s feature map, a tensor of dimensions (N, C, 22, 22)
        :param base_32s_feats: base_32s feature map, a tensor of dimensions (N, C, 11, 11)
        :param extra_conv1b_feats: extra_conv1b feature map, a tensor of dimensions (N, C, 5, 5)
        :param extra_conv2b_feats: extra_conv2b feature map, a tensor of dimensions (N, C, 3, 3)
        :param extra_conv3b_feats: extra_conv3b feature map, a tensor of dimensions (N, C, 1, 1)
        :return: pre-calculated locations and class scores (i.e. w.r.t each prior box) for each image
        """


        # for 16s detection
        base_16s_feats_ = self.base_16s_CBR(base_16s_feats)

        # for 8s detection
        # base_8s_feats_ = F.interpolate(base_16s_feats_, scale_factor=2, mode='bilinear', align_corners=None)
        base_8s_feats_ = F.interpolate(base_16s_feats_, scale_factor=2, mode='nearest', align_corners=None)
        base_8s_feats = torch.cat([base_8s_feats, base_8s_feats_], 1)
        base_8s_feats = self.base_8s_CBR(base_8s_feats)
        # base_8s_feats = self.base_8s_C3BlockB(base_8s_feats)

        # for 16s detection
        base_16s_feats = self.base_16s_DepDown(base_8s_feats)
        # base_16s_feats = self.base_16s_C3BlockB(base_16s_feats)

        # for Segmentation
        # base_8s_feats_seg = self.base_8s_CBR_seg(torch.cat([self.sample(image), base_8s_feats], 1))
        # base_8s_feats_seg = self.base_8s_C3BlockB_seg(base_8s_feats_seg)
        # base_8s_feats_seg = self.base_8s_seg_pred(base_8s_feats_seg)
        # base_8s_feats_seg = self.upsample_seg_pred(base_8s_feats_seg)

        # for 32s detection
        base_32s_feats = self.base_32s_CBR(base_32s_feats)
        # base_32s_feats = self.base_32s_C3BlockB(base_32s_feats)

        extra_conv1b_feats = self.extra_conv1b_CBR(extra_conv1b_feats)
        extra_conv2b_feats = self.extra_conv2b_CBR(extra_conv2b_feats)
        extra_conv3b_feats = self.extra_conv3b_CBR(extra_conv3b_feats)

        batch_size = base_8s_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_base_8s = self.loc_base_8s(base_8s_feats)  # (N, C, 44, 44)
        l_base_8s = l_base_8s.permute(0, 2, 3,
                                      1).contiguous()  # (N, 44, 44, C), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_base_8s = l_base_8s.view(batch_size, -1, 4)  # (N, 44x44x4, 4), there are a total 44x44x4 boxes on this feature map

        l_base_16s = self.loc_base_16s(base_16s_feats)  # (N, C, 22, 22)
        l_base_16s = l_base_16s.permute(0, 2, 3, 1).contiguous()  # (N, 22, 22, C)
        l_base_16s = l_base_16s.view(batch_size, -1, 4)  # (N, 22x22x6, 4), there are a total 22x22x6 boxes on this feature map

        l_base_32s = self.loc_base_32s(base_32s_feats)  # (N, C, 11, 11)
        l_base_32s = l_base_32s.permute(0, 2, 3, 1).contiguous()  # (N, 11, 11, C)
        l_base_32s = l_base_32s.view(batch_size, -1, 4)  # (N, 11x11x6, 4)

        l_extra_conv1b = self.loc_extra_conv1b(extra_conv1b_feats)  # (N, C, 5, 5)
        l_extra_conv1b = l_extra_conv1b.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, C)
        l_extra_conv1b = l_extra_conv1b.view(batch_size, -1, 4)  # (N, 5x5x6, 4)

        l_extra_conv2b = self.loc_extra_conv2b(extra_conv2b_feats)  # (N, C, 3, 3)
        l_extra_conv2b = l_extra_conv2b.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, C)
        l_extra_conv2b = l_extra_conv2b.view(batch_size, -1, 4)  # (N, 3x3x4, 4)

        l_extra_conv3b = self.loc_extra_conv3b(extra_conv3b_feats)  # (N, C, 1, 1)
        l_extra_conv3b = l_extra_conv3b.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, C)
        l_extra_conv3b = l_extra_conv3b.view(batch_size, -1, 4)  # (N, 1x1x4, 4)

        # Predict classes in localization boxes
        c_base_8s = self.cl_base_8s_(base_8s_feats)  # (N, 4 * n_classes, 44, 44)
        c_base_8s = c_base_8s.permute(0, 2, 3,
                                      1).contiguous()  # (N, 44, 44, 4 * n_classes), to match prior-box order (after .view())
        c_base_8s = c_base_8s.view(batch_size, -1,
                                   self.n_classes)  # (N, 44x44x4, n_classes), there are a total 44x44x4 boxes on this feature map

        c_base_16s = self.cl_base_16s_(base_16s_feats)  # (N, 6 * n_classes, 22, 22)
        c_base_16s = c_base_16s.permute(0, 2, 3, 1).contiguous()  # (N, 22, 22, 6 * n_classes)
        c_base_16s = c_base_16s.view(batch_size, -1,
                               self.n_classes)  # (N, 22x22x6, n_classes), there are a total 22x22x6 boxes on this feature map

        c_base_32s = self.cl_base_32s_(base_32s_feats)  # (N, 6 * n_classes, 11, 11)
        c_base_32s = c_base_32s.permute(0, 2, 3, 1).contiguous()  # (N, 11, 11, 6 * n_classes)
        c_base_32s = c_base_32s.view(batch_size, -1, self.n_classes)  # (N, 11x11x6, n_classes)

        c_extra_conv1b = self.cl_extra_conv1b_(extra_conv1b_feats)  # (N, 6 * n_classes, 5, 5)
        c_extra_conv1b = c_extra_conv1b.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_extra_conv1b = c_extra_conv1b.view(batch_size, -1, self.n_classes)  # (N, 5x5x6, n_classes)

        c_extra_conv2b = self.cl_extra_conv2b_(extra_conv2b_feats)  # (N, 4 * n_classes, 3, 3)
        c_extra_conv2b = c_extra_conv2b.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_extra_conv2b = c_extra_conv2b.view(batch_size, -1, self.n_classes)  # (N, 3x3x4, n_classes)

        c_extra_conv3b = self.cl_extra_conv3b_(extra_conv3b_feats)  # (N, 4 * n_classes, 1, 1)
        c_extra_conv3b = c_extra_conv3b.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_extra_conv3b = c_extra_conv3b.view(batch_size, -1, self.n_classes)  # (N, 1x1x4, n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_base_8s, l_base_16s, l_base_32s, l_extra_conv1b, l_extra_conv2b, l_extra_conv3b], dim=1)  # (N, pre-calculated, 4)
        classes_scores = torch.cat([c_base_8s, c_base_16s, c_base_32s, c_extra_conv1b, c_extra_conv2b, c_extra_conv3b],
                                   dim=1)  # (N, pre-calculated, n_classes)

        # return locs, classes_scores, base_8s_feats_seg
        return locs, classes_scores

class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the pre-calculated prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the pre-calculated bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'base_8s': 4,
                   'base_16s': 6,
                   'base_32s': 6,
                   'extra_conv1b': 6,
                   'extra_conv2b': 4,
                   'extra_conv3b': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # for 8s detection
        self.base_8s_CBR = CBR(512, 256//2, 1)
        # self.base_8s_C3BlockB = C3BlockB(256, 256 // 2)

        # for 16s detection
        self.base_16s_CBR = CBR(512, 256, 1)
        # self.base_16s_DepDown = DepthWiseBlock(256//2, 256//2, 2, 1)
        self.base_16s_DepDown = CBR(256//2, 256//2, 3, 2, 1)
        # self.base_16s_C3BlockB = C3BlockB(256 // 2, 256 // 2)

        # for 32s detection
        self.base_32s_CBR = CBR(512, 256//2, 1)
        # self.base_32s_C3BlockB = C3BlockB(256, 256 // 2)

        # for extra detection
        # self.extra_conv1b_C3BlockB = C3BlockB(256, 256 // 2)
        # self.extra_conv2b_C3BlockB = C3BlockB(256, 256 // 2)
        # self.extra_conv3b_C3BlockB = C3BlockB(256, 256 // 2)
        self.extra_conv1b_CBR = CBR(256, 256 // 2, 1)
        self.extra_conv2b_CBR = CBR(256, 256 // 2, 1)
        self.extra_conv3b_CBR = CBR(256, 256 // 2, 1)

        # for segmentation
        # self.sample = InputProjectionA(3)
        # self.base_8s_CBR_seg = CBR(256//2+3, 128, 1)
        # self.base_8s_C3BlockB_seg = C3BlockB(128, 128 // 2)
        # self.base_8s_seg_pred = nn.Conv2d(128//2, 2, kernel_size=1, padding=0)
        # self.upsample_seg_pred = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=False)

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_base_8s = nn.Conv2d(256//2, n_boxes['base_8s'] * 4, kernel_size=3, padding=1)
        self.loc_base_16s = nn.Conv2d(256//2, n_boxes['base_16s'] * 4, kernel_size=3, padding=1)
        self.loc_base_32s = nn.Conv2d(256//2, n_boxes['base_32s'] * 4, kernel_size=3, padding=1)
        self.loc_extra_conv1b = nn.Conv2d(256//2, n_boxes['extra_conv1b'] * 4, kernel_size=3, padding=1)
        self.loc_extra_conv2b = nn.Conv2d(256//2, n_boxes['extra_conv2b'] * 4, kernel_size=3, padding=1)
        self.loc_extra_conv3b = nn.Conv2d(256//2, n_boxes['extra_conv3b'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_base_8s = nn.Conv2d(256//2, n_boxes['base_8s'] * n_classes, kernel_size=3, padding=1)
        self.cl_base_16s = nn.Conv2d(256//2, n_boxes['base_16s'] * n_classes, kernel_size=3, padding=1)
        self.cl_base_32s = nn.Conv2d(256//2, n_boxes['base_32s'] * n_classes, kernel_size=3, padding=1)
        self.cl_extra_conv1b = nn.Conv2d(256//2, n_boxes['extra_conv1b'] * n_classes, kernel_size=3, padding=1)
        self.cl_extra_conv2b = nn.Conv2d(256//2, n_boxes['extra_conv2b'] * n_classes, kernel_size=3, padding=1)
        self.cl_extra_conv3b = nn.Conv2d(256//2, n_boxes['extra_conv3b'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.modules():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                if c.bias is not None:
                    nn.init.constant_(c.bias, 0.)

    def forward(self, image, base_8s_feats, base_16s_feats, base_32s_feats,
                extra_conv1b_feats, extra_conv2b_feats, extra_conv3b_feats):
        """
        Forward propagation.

        :param base_8s_feats: base_8s feature map, a tensor of dimensions (N, C, 44, 44)
        :param base_16s_feats: base_16s feature map, a tensor of dimensions (N, C, 22, 22)
        :param base_32s_feats: base_32s feature map, a tensor of dimensions (N, C, 11, 11)
        :param extra_conv1b_feats: extra_conv1b feature map, a tensor of dimensions (N, C, 5, 5)
        :param extra_conv2b_feats: extra_conv2b feature map, a tensor of dimensions (N, C, 3, 3)
        :param extra_conv3b_feats: extra_conv3b feature map, a tensor of dimensions (N, C, 1, 1)
        :return: pre-calculated locations and class scores (i.e. w.r.t each prior box) for each image
        """


        # for 16s detection
        base_16s_feats_ = self.base_16s_CBR(base_16s_feats)

        # for 8s detection
        base_8s_feats_ = F.interpolate(base_16s_feats_, scale_factor=2, mode='bilinear', align_corners=False)
        base_8s_feats = torch.cat([base_8s_feats, base_8s_feats_], 1)
        base_8s_feats = self.base_8s_CBR(base_8s_feats)
        # base_8s_feats = self.base_8s_C3BlockB(base_8s_feats)

        # for 16s detection
        base_16s_feats = self.base_16s_DepDown(base_8s_feats)
        # base_16s_feats = self.base_16s_C3BlockB(base_16s_feats)

        # for Segmentation
        # base_8s_feats_seg = self.base_8s_CBR_seg(torch.cat([self.sample(image), base_8s_feats], 1))
        # base_8s_feats_seg = self.base_8s_C3BlockB_seg(base_8s_feats_seg)
        # base_8s_feats_seg = self.base_8s_seg_pred(base_8s_feats_seg)
        # base_8s_feats_seg = self.upsample_seg_pred(base_8s_feats_seg)

        # for 32s detection
        base_32s_feats = self.base_32s_CBR(base_32s_feats)
        # base_32s_feats = self.base_32s_C3BlockB(base_32s_feats)

        extra_conv1b_feats = self.extra_conv1b_CBR(extra_conv1b_feats)
        extra_conv2b_feats = self.extra_conv2b_CBR(extra_conv2b_feats)
        extra_conv3b_feats = self.extra_conv3b_CBR(extra_conv3b_feats)

        batch_size = base_8s_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_base_8s = self.loc_base_8s(base_8s_feats)  # (N, C, 44, 44)
        l_base_8s = l_base_8s.permute(0, 2, 3,
                                      1).contiguous()  # (N, 44, 44, C), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_base_8s = l_base_8s.view(batch_size, -1, 4)  # (N, 44x44x4, 4), there are a total 44x44x4 boxes on this feature map

        l_base_16s = self.loc_base_16s(base_16s_feats)  # (N, C, 22, 22)
        l_base_16s = l_base_16s.permute(0, 2, 3, 1).contiguous()  # (N, 22, 22, C)
        l_base_16s = l_base_16s.view(batch_size, -1, 4)  # (N, 22x22x6, 4), there are a total 22x22x6 boxes on this feature map

        l_base_32s = self.loc_base_32s(base_32s_feats)  # (N, C, 11, 11)
        l_base_32s = l_base_32s.permute(0, 2, 3, 1).contiguous()  # (N, 11, 11, C)
        l_base_32s = l_base_32s.view(batch_size, -1, 4)  # (N, 11x11x6, 4)

        l_extra_conv1b = self.loc_extra_conv1b(extra_conv1b_feats)  # (N, C, 5, 5)
        l_extra_conv1b = l_extra_conv1b.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, C)
        l_extra_conv1b = l_extra_conv1b.view(batch_size, -1, 4)  # (N, 5x5x6, 4)

        l_extra_conv2b = self.loc_extra_conv2b(extra_conv2b_feats)  # (N, C, 3, 3)
        l_extra_conv2b = l_extra_conv2b.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, C)
        l_extra_conv2b = l_extra_conv2b.view(batch_size, -1, 4)  # (N, 3x3x4, 4)

        l_extra_conv3b = self.loc_extra_conv3b(extra_conv3b_feats)  # (N, C, 1, 1)
        l_extra_conv3b = l_extra_conv3b.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, C)
        l_extra_conv3b = l_extra_conv3b.view(batch_size, -1, 4)  # (N, 1x1x4, 4)

        # Predict classes in localization boxes
        c_base_8s = self.cl_base_8s(base_8s_feats)  # (N, 4 * n_classes, 44, 44)
        c_base_8s = c_base_8s.permute(0, 2, 3,
                                      1).contiguous()  # (N, 44, 44, 4 * n_classes), to match prior-box order (after .view())
        c_base_8s = c_base_8s.view(batch_size, -1,
                                   self.n_classes)  # (N, 44x44x4, n_classes), there are a total 44x44x4 boxes on this feature map

        c_base_16s = self.cl_base_16s(base_16s_feats)  # (N, 6 * n_classes, 22, 22)
        c_base_16s = c_base_16s.permute(0, 2, 3, 1).contiguous()  # (N, 22, 22, 6 * n_classes)
        c_base_16s = c_base_16s.view(batch_size, -1,
                               self.n_classes)  # (N, 22x22x6, n_classes), there are a total 22x22x6 boxes on this feature map

        c_base_32s = self.cl_base_32s(base_32s_feats)  # (N, 6 * n_classes, 11, 11)
        c_base_32s = c_base_32s.permute(0, 2, 3, 1).contiguous()  # (N, 11, 11, 6 * n_classes)
        c_base_32s = c_base_32s.view(batch_size, -1, self.n_classes)  # (N, 11x11x6, n_classes)

        c_extra_conv1b = self.cl_extra_conv1b(extra_conv1b_feats)  # (N, 6 * n_classes, 5, 5)
        c_extra_conv1b = c_extra_conv1b.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_extra_conv1b = c_extra_conv1b.view(batch_size, -1, self.n_classes)  # (N, 5x5x6, n_classes)

        c_extra_conv2b = self.cl_extra_conv2b(extra_conv2b_feats)  # (N, 4 * n_classes, 3, 3)
        c_extra_conv2b = c_extra_conv2b.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_extra_conv2b = c_extra_conv2b.view(batch_size, -1, self.n_classes)  # (N, 3x3x4, n_classes)

        c_extra_conv3b = self.cl_extra_conv3b(extra_conv3b_feats)  # (N, 4 * n_classes, 1, 1)
        c_extra_conv3b = c_extra_conv3b.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_extra_conv3b = c_extra_conv3b.view(batch_size, -1, self.n_classes)  # (N, 1x1x4, n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_base_8s, l_base_16s, l_base_32s, l_extra_conv1b, l_extra_conv2b, l_extra_conv3b], dim=1)  # (N, pre-calculated, 4)
        classes_scores = torch.cat([c_base_8s, c_base_16s, c_base_32s, c_extra_conv1b, c_extra_conv2b, c_extra_conv3b],
                                   dim=1)  # (N, pre-calculated, n_classes)

        # return locs, classes_scores, base_8s_feats_seg
        return locs, classes_scores


class SSD352EFFB0(nn.Module):
    """
    The SSD352 network - encapsulates the base network, auxiliary, and prediction convolutions.
    """

    def __init__(self, width_coeff, depth_coeff,n_classes):
        super(SSD352EFFB0, self).__init__()

        self.n_classes = n_classes
        self.width_coeff = width_coeff
        self.depth_coeff = depth_coeff
        self.base = EfficientNet(width_coeff=self.width_coeff,depth_coeff=self.depth_coeff)
        self.aux_convs = AuxiliaryConvolutions()
        # self.pred_convs = PredictionConvolutions(n_classes)
        self.pred_convs = PredictionConvolutions_Upsample_NN(n_classes)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, H, W)
        :return: pre-calculated locations and class scores (i.e. w.r.t each prior box) for each image
        """

        # Run base network convolutions (lower level feature map generators)
        base_8s_feats, base_16s_feats, base_32s_feats = self.base(image)


        # Run auxiliary convolutions (higher level feature map generators)
        extra_conv1b_feats, extra_conv2b_feats, extra_conv3b_feats = self.aux_convs(base_32s_feats)


        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        # locs, classes_scores, peleenet8_feats_sg = self.pred_convs(image, base_8s_feats, base_16s_feats, base_32s_feats,
        locs, classes_scores = self.pred_convs(image, base_8s_feats, base_16s_feats, base_32s_feats,
                                               extra_conv1b_feats, extra_conv2b_feats, extra_conv3b_feats)  # (N, pre-calculated, 4), (N, pre-calculated, n_classes)

        # return locs, classes_scores, peleenet8_feats_sg
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the pre-calculated prior (default) boxes for the SSD, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (pre-calculated, 4)
        """

        fmap_dims = {'base_8s': 88,
                     'base_16s': 44,
                     'base_32s': 22,
                     'extra_conv1b': 5,
                     'extra_conv2b': 3,
                     'extra_conv3b': 1}

        obj_scales = {'base_8s': 0.1,
                      'base_16s': 0.2,
                      'base_32s': 0.375,
                      'extra_conv1b': 0.55,
                      'extra_conv2b': 0.725,
                      'extra_conv3b': 0.9}

        # there are two for 1
        aspect_ratios = {'base_8s': [1., 2., 0.5],
                         'base_16s': [1., 2., 3., 0.5, .333],
                         'base_32s': [1., 2., 3., 0.5, .333],
                         'extra_conv1b': [1., 2., 3., 0.5, .333],
                         'extra_conv2b': [1., 2., 0.5],
                         'extra_conv3b': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (pre-calculated, 4)
        prior_boxes.clamp_(0, 1)  # (pre-calculated, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the pre-calculated locations and class scores (output of ths SSD) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the pre-calculated prior boxes, a tensor of dimensions (N, pre-calculated, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, pre-calculated, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, pre-calculated, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = utils.cxcy_to_xy(
                utils.gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (pre-calculated, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (pre-calculated)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (pre-calculated)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= pre-calculated
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = utils.find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    #suppress = torch.max(suppress, overlap[box] > max_overlap)
                    condition = overlap[box] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.uint8).to(device)
                    suppress = torch.max(suppress, condition)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = utils.cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the pre-calculated prior boxes, a tensor of dimensions (N, pre-calculated, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, pre-calculated, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        print(n_priors)
        print(predicted_locs.size(1))
        print(predicted_scores.size(1))
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, pre-calculated, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, pre-calculated)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = utils.find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, pre-calculated)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (pre-calculated)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (pre-calculated)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (pre-calculated)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = utils.cxcy_to_gcxgcy(utils.xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (pre-calculated, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, pre-calculated)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * pre-calculated)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, pre-calculated)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, pre-calculated)
        conf_loss_neg[positive_priors] = 0.  # (N, pre-calculated), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, pre-calculated), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, pre-calculated)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, pre-calculated)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss


def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])


if __name__ == "__main__":
    from thop import profile
    model = SSD352(n_classes=11)
    #print(model)
    # print("params: {}".format(netParams(model)))
    # flops, params = profile(model, input_size=(1, 3, 352, 352))
    # print("flops: {:.2f}G, params: {:.2f}M".format(flops/1e9, params/1e6))
    #
    # dummy_input = torch.randn(1, 3, 352, 352)
    # torch.onnx.export(model, dummy_input, os.path.join("./", "ELANetV3_modified.onnx"))
    #
    # mobilenet_base = mobilenet()
    # torch.onnx.export(mobilenet_base, dummy_input, os.path.join("./", "mobilenet.onnx"))