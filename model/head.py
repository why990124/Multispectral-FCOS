
import torch.nn as nn
import torch
import math

class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior   # useless
        self.class_num=class_num  # class number
        self.cnt_on_reg=cnt_on_reg  # fcos v1: False, fcos v2: True
        
        cls_branch=[]
        reg_branch=[]

        for i in range(4):   # 4× conv 3×3
            cls_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True)) # add con 3×3 into cls_branch
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel)) # group normalization
            cls_branch.append(nn.ReLU(True)) # add activation function

            reg_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))  # add con 3×3 into reg_branch
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel)) # group normalization
            reg_branch.append(nn.ReLU(True)) # add activation function

        self.cls_conv=nn.Sequential(*cls_branch) # form sequential network
        self.reg_conv=nn.Sequential(*reg_branch) # form sequential network

        self.cls_logits=nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1) # H*W*256 -> H*W*C
        self.cnt_logits=nn.Conv2d(in_channel,1,kernel_size=3,padding=1)  # H*W*256 -> H*W*1
        self.reg_pred=nn.Conv2d(in_channel,4,kernel_size=3,padding=1)  # H*W*256 -> H*W*1
        
        self.apply(self.init_conv_RandomNormal) # initialize model
        
        nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior)) # initialize cls_logits.bias with (-log((1 - prior) / prior)))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])  # set learnable scale factor
    
    def init_conv_RandomNormal(self,module,std=0.01):  # newwork initialization
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,inputs):
        '''inputs:[P3~P7]'''
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(inputs):
            cls_conv_out=self.cls_conv(P) # cls branch
            reg_conv_out=self.reg_conv(P) # reg branch

            cls_logits.append(self.cls_logits(cls_conv_out)) # output cls H*W*C
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out)) # output cls H*W*1  on reg_branch
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out)) # output cls H*W*1  on cls_branch

            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out))) # output reg H*W*1  on reg_branch
        return cls_logits,cnt_logits,reg_preds



        
