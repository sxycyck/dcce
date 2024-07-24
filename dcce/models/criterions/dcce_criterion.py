import torch.nn as nn
import torch
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from .debias_loss import LearnedMixin as LMH
import math

TOTAL_EPOCH = 22

class DCCECriterion(nn.Module):

    def __init__(self,question_loss_weight,entropy_loss_weight,engine=None):
        super().__init__()

        Logger()(f'DCCECriterion, with question_loss_weight = ({question_loss_weight})')
   

        self.fusion_loss_qk = nn.CrossEntropyLoss()
        self.fusion_loss_tk = nn.CrossEntropyLoss()
        self.question_loss = nn.CrossEntropyLoss()
        self.entropy_loss = LMH()                   #entropy as penalty loss
        
        self.question_loss_weight = question_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.engine = engine
     
    def forward(self, net_out, batch):
        
        out = {}
        class_id = batch['class_id'].squeeze(1)
        loss = 0
        
        if 'logits_q' in net_out.keys():
            logits_q = net_out['logits_q']
            logits_qk = net_out['logits_qk']
            qk_loss = self.fusion_loss_qk(logits_qk,class_id)
            question_loss = self.question_loss(logits_q,class_id)
            loss+=qk_loss
            loss = loss + self.question_loss_weight * question_loss
            
        if 'logits_tk' in net_out.keys():
            
            logits_tk = net_out['logits_tk']
            tk_loss = self.fusion_loss_tk(logits_tk,class_id)
            entropy_loss = self.entropy(net_out,batch)
            loss+=tk_loss
            way = self.entropy_loss_weight['way']
            weight = self.entropy_loss_weight['weight']
            if way=='dynamic':
                current_epoch = self.engine.epoch
                n = math.ceil(2/3*TOTAL_EPOCH)
                if current_epoch<n:
                    weight = weight*math.cos((math.pi/2)*(current_epoch/n))
                else:
                    weight = 0
            loss = loss + weight * entropy_loss
        
        if loss == 0:              #no debias
            logits_vq = net_out['logits_vq']
            loss = self.fusion_loss_qk(logits_vq,class_id)
            
        logits_dcce = net_out['logits_dcce']
        
        te = net_out['z_te']
        de = net_out['z_de']
    
        p_te = torch.nn.functional.softmax(te, -1)
        p_de = torch.nn.functional.softmax(de, -1)
        
        kl_loss = - p_te * p_de.log()    
        kl_loss = kl_loss.sum(1).mean() 
        
        loss += kl_loss
        
        out['loss'] = loss
        
        return out
    
    def entropy(self,net_out,batch):
        out = {}  
        joint_repr = net_out['joint_repr']
        smooth_param = net_out['smooth_param']
        entropy_loss = 0
        if 'target' in batch.keys():
            bias = batch['bias']
            bias = torch.tensor(bias,dtype=torch.float32).cuda()             
            entropy_loss = self.entropy_loss(joint_repr,bias,smooth_param)

        return entropy_loss
    
