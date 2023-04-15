# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import sys

def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class FCN(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=13):
        super(FCN, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.blend4 = bn_relu_conv(32, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 128, kernel_size=3)
        self.blend2 = bn_relu_conv(128, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        out1 = self.conv4(out1)
        out1 = self.relu(out1)
        #out1 = self.pool4(out1)
        out1 = self.dropout(out1)
        skip5 = out1
        # decode process
        up4 = upsample(skip5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense


class FCNA(nn.Module):
    """FCNA for motif mining"""
    def __init__(self, motiflen=13):
        super(FCNA, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=128, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # decode process
        self.blend5 = bn_relu_conv(32, 32, kernel_size=3)
        self.blend4 = bn_relu_conv(32, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 128, kernel_size=3)
        self.blend2 = bn_relu_conv(128, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        out1 = self.conv4(out1)
        out1 = self.relu(out1)
       #out1 = self.pool4(out1)
        out1 = self.dropout(out1)
        skip5 = out1
        up6 = self.aap(out1)
        # decode process
        up5 = upsample(up6, skip5.size()[-1])
        up5 = up5 + skip5
        up5 = self.blend5(up5)
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense

