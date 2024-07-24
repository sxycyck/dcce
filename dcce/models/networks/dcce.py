import torch
import torch.nn as nn
from block.models.networks.mlp import MLP
from .utils import grad_mul_const # mask_softmax, grad_reverse, grad_reverse_mask, 
from torch.nn import functional as F
eps = 1e-12

class DCCE(nn.Module):
    
    def __init__(self, model, output_size, classif_q, s_bias_way, d_bias_way, end_classif=True):
        super().__init__()
        self.net = model
        self.end_classif = end_classif
        
        # Q->A branch
        self.q_1 = MLP(**classif_q)
        if self.end_classif: # default: True (following RUBi)
            self.q_2 = nn.Linear(output_size, output_size)

        self.constant = nn.Parameter(torch.tensor(0.0))
        self.s_bias_way = s_bias_way
        self.d_bias_way = d_bias_way

    def forward(self, batch):
        
        out = {}
        # model prediction
        net_out = self.net(batch)
        logits = net_out['logits']
       
        out['joint_repr'] = net_out['joint_repr']
        out['smooth_param'] = net_out['smooth_param']
        
        #initalize q_pred,t_pred
        q_pred = torch.zeros_like(logits)
        t_pred = torch.zeros_like(logits)
        
        if self.s_bias_way is not None:
            
            # shortcut bias branch
            q_embedding = net_out['q_emb']  # N * q_emb
            q_embedding = grad_mul_const(q_embedding, 0.0) # don't backpropagate
            q_pred = self.q_1(q_embedding)
            z_qk = torch.log(torch.sigmoid(logits+q_pred)+eps)
            out['logits_qk'] = z_qk
            if self.end_classif:
                q_out = self.q_2(q_pred)
            else:
                q_out = q_pred
            out['logits_q'] = q_out 
        
        if self.d_bias_way is not None:
           
            # distribution bias branch
            t_pred = self.getBias(out,batch)
            z_tk = torch.log(torch.sigmoid(logits+t_pred)+eps)
            out['logits_tk'] = z_tk
        
        
        out['logits_vq']  = logits # predictions of the original UpDn
        
        
        out['z_te'] = self.fusion(logits.clone().detach(),q_pred.clone().detach(),t_pred.clone().detach(),is_k = True)
        out['z_de'] = self.fusion(logits.clone().detach(),q_pred.clone().detach(),t_pred.clone().detach(),is_k = False)

        out['logits_dcce'] =  out['z_te'] - out['z_de']  #final decisions as results
            
        return out   

    def process_answers(self, out, key=''):
        
        out = self.net.process_answers(out, key='_vq')
        out = self.net.process_answers(out, key='_dcce')
        
        return out

    def fusion(self, z_k, z_q, z_t, is_k):
        
        if not is_k:
            z_k = self.constant * torch.ones_like(z_k).cuda()
            
        z = torch.zeros_like(z_k)
        z+=z_k
        
        if self.s_bias_way == 'NIE':
            z_q = self.constant * torch.ones_like(z_q).cuda()
            z+=z_q
        elif self.s_bias_way == 'TIE':
            z+=z_q
        
        if self.d_bias_way == 'NIE':
            z_t = self.constant * torch.ones_like(z_t).cuda()
            z+=z_t
        elif self.d_bias_way =='TIE':
            z+=z_t
           
        z = torch.log(torch.sigmoid(z) + eps)

        return z
    
    def getBias(self,out,batch):
        
        factor = out['joint_repr']
        smooth_param = out['smooth_param']
        factor = F.softplus(factor)
        bias = batch['bias']
        
        bias = torch.tensor(bias,dtype=torch.float32).cuda().unsqueeze(1)
        sofen_factor = F.sigmoid(smooth_param)
        bias = bias + sofen_factor.unsqueeze(1)
        bias = torch.log(bias)
        bias = bias * factor.unsqueeze(1)
        return bias.squeeze(1)