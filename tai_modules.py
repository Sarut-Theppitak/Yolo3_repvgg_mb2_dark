import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np

def where(cond, x1, x2):
    cond = cond.float()
    return (cond * x1) + ((1-cond) * x2)

def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
        #print('vec',tensor)
        #return tensor.sign()
        #zeros = torch.zeros_like(tensor)
        #plus  = where(tensor != 0.0, tensor.sign(), zeros) #torch.where(tensor>0, 1, 0)
        #minus = #torch.where(tensor<0,-1, 0)
        #return plus# + minus
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


def Ternarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        #print('vec',tensor)

        #return tensor.sign()
        zeros = torch.zeros_like(tensor)
        plus  = where(tensor != 0.0, tensor.sign(), zeros) #torch.where(tensor>0, 1, 0)
        #minus = #torch.where(tensor<0,-1, 0)
        return plus# + minus
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def StaticQuantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        #tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))

        #print('org tensor',tensor)
        #print('org tensor',tensor.mul(2**(numBits-1)))
        #print('org tensor',tensor.mul(2**(numBits-1)).round())

        #int_tensor = tensor.mul(2**(numBits-1)).round().int()
        #print("int tensor", int_tensor)

        #float_tensor = tensor.mul(2**(numBits-1)).round().float()
        #print("float tensor", float_tensor)

        #tensor = tensor.div(2**(numBits-1))

        ##tensor=tensor.mul(2**(numBits-1)).trunc().div(2**(numBits-1))

        #print('quant tensor',tensor)
        #exit()

        integer_bit = int(tensor.__abs__().max().log2().floor() + 1)
        #print(integer_bit)
        decimal_bit = 8 - integer_bit - 1
        tensor = (tensor * 2 ** decimal_bit).round() / 2 ** decimal_bit

    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

class StaticQuantizeConv2d(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', wBits=-1, aBits=-1):
        super(StaticQuantizeConv2d, self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.wBits = wBits
        self.aBits = -1 ###aBits
        #print('QuantConv2d w=%d a=%d' % (wBits,aBits))

    def forward(self, input):
        #if input.size(1) != 3:
        #    #input.data = input.data #Binarize(input.data)
        #    if self.aBits != 1:
        #        input.data = Quantize(input.data, numBits=self.aBits)
        #    else:
        #        input.data = Binarize(input.data)

        if self.aBits != -1:
            if self.aBits != 1:
                input.data = StaticQuantize(input.data, numBits=self.aBits)
            else:
                input.data = Binarize(input.data)

        if self.wBits != -1:
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
                self.weight.numBits=self.wBits

            if self.wBits == 1:
                self.weight.data=Ternarize(self.weight.org)
            else:
                #print("weight quant")
                self.weight.data=StaticQuantize(self.weight.org,numBits=self.wBits)

        #print('quant x',input.data)
        #och, ich, ky, kx = self.weight.shape
        #print(och,ich,ky,kx)
        #if ky != 1:
        #print('quant w',self.weight.data)
        #print('quant w',self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        # convert to original floating point value (Norm and Activation)
        #print('quant conv', out)

        '''
        if self.aBits == -1 and self.wBits == -1:
            out = out
        elif self.aBits == -1 and self.wBits != -1:
            out = out.div(2**(self.wBits-1)).mul(abs_max_w)
        elif self.aBits != -1 and self.wBits == -1:
            out = out.div(2**(self.aBits-1)).mul(abs_max_x)
        else:        
            out = out.div(2**(self.aBits-1)).mul(abs_max_x).div(2**(self.wBits-1)).mul(abs_max_w)
        '''

        #out = out.div(2**(self.aBits-1)).mul(abs_max_x).div(2**(self.wBits-1)).mul(abs_max_w)
        #print('float conv', out)
        #exit()

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def DynamicQuantize(tensor,quant_mode='det',  params=None, numBits=8):
    ##tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        #tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))

        #print('tensor',tensor)
        maxval = torch.max(torch.abs(tensor))
        #print('abs_max value', maxval)
        if maxval == 0.0:
            maxval = 1.0
        ##tensor=tensor.mul(2**(numBits-1)).trunc().div(2**(numBits-1))

        tensor=tensor.div(maxval).mul(2**(numBits-1)).trunc()
        #print('quant tensor',tensor)
        #exit()

    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

#import torch.nn._functions as tnnf

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class DynamicQuantizeConv2d(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', wBits=-1, aBits=-1):
        super(DynamicQuantizeConv2d, self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.wBits = wBits
        self.aBits = aBits
        #print('QuantConv2d w=%d a=%d' % (wBits,aBits))

    def forward(self, input):
        #if input.size(1) != 3:
        #    #input.data = input.data #Binarize(input.data)
        #    if self.aBits != 1:
        #        input.data = Quantize(input.data, numBits=self.aBits)
        #    else:
        #        input.data = Binarize(input.data)

        abs_max_x = torch.max(torch.abs(input.data))

        if self.aBits != -1:
            if self.aBits != 1:
                input.data = DynamicQuantize(input.data, numBits=self.aBits)
            else:
                input.data = Binarize(input.data)

        if self.wBits != -1:
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
                self.weight.numBits=self.wBits

            abs_max_w = torch.max(torch.abs(self.weight.org))

            if self.wBits == 1:
                self.weight.data=Ternarize(self.weight.org)
            else:
                self.weight.data=DynamicQuantize(self.weight.org,numBits=self.wBits)
        else:
            abs_max_w = 1.0

        #print('quant x',input.data)
        #och, ich, ky, kx = self.weight.shape
        #print(och,ich,ky,kx)
        #if ky != 1:
        #print('quant w',self.weight.data)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        # convert to original floating point value (Norm and Activation)
        #print('quant conv', out)

        if abs_max_x == 0.0:
            abs_max_x = 1.0
        
        if abs_max_w == 0.0:
            abs_max_w = 1.0

        if self.aBits == -1 and self.wBits == -1:
            out = out
        elif self.aBits == -1 and self.wBits != -1:
            out = out.div(2**(self.wBits-1)).mul(abs_max_w)
        elif self.aBits != -1 and self.wBits == -1:
            out = out.div(2**(self.aBits-1)).mul(abs_max_x)
        else:        
            out = out.div(2**(self.aBits-1)).mul(abs_max_x).div(2**(self.wBits-1)).mul(abs_max_w)

        #out = out.div(2**(self.aBits-1)).mul(abs_max_x).div(2**(self.wBits-1)).mul(abs_max_w)
        #print('float conv', out)
        #exit()

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
