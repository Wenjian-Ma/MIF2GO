import torch.nn as nn
import torch
from torch.nn import functional as F
from GAE_model import double_NoiseGAE
class Transformer_scale_aggr(nn.Module):

    def __init__(self,dropout,device=None,heads=None):
        super(Transformer_scale_aggr,self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(400, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.BatchNorm1d(400)
        self.norm2 = nn.BatchNorm1d(400)
        self.FFN = nn.Sequential(
            nn.Linear(400, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,400)
        )

        self.MLP_scale1 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.MLP_scale2 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.MLP_scale3 = nn.Sequential(
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
    def forward(self,scale1, scale2,scale3):#scale3



        combined_feat = torch.cat([self.MLP_scale1(scale1), self.MLP_scale2(scale2),self.MLP_scale3(scale3)], dim=-1)#self.MLP_scale3(scale3)
        output_ori = combined_feat.reshape(combined_feat.shape[0], 3, -1)#3
        output = self.self_attn(output_ori, output_ori, output_ori,
                           attn_mask=None,
                           key_padding_mask=None,
                           need_weights=False)[0]

        output = self.norm1(output_ori.reshape(-1,400) + self.dropout(output.reshape(-1,400)))

        dh = self.FFN(output)

        output = self.norm2(output + self.dropout(dh))

        return output.reshape(-1,3,400).sum(1)#3

class nnModel(nn.Module):

    def __init__(self,num_labels,dropout,device,args):
        super(nnModel,self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.device = device

        self.classifier = nn.Sequential(
            nn.Linear(400, 1024),#1200 for ablation
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_labels),
        )

        self.MLP_lm = nn.Sequential(#*3
            nn.Linear(1280*3, 1280),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1280, 400),
        )


        self.scale_aggr = Transformer_scale_aggr(dropout,heads=args.heads)

        self.weight = torch.nn.init.constant_(nn.Parameter(torch.Tensor(3)), 1.0)
    def forward(self,emb,lm_33,lm_28,lm_23):
        weight = F.softmax(self.weight)

        lm = torch.cat([weight[0]*lm_33,weight[1]*lm_28,weight[2]*lm_23],dim=-1)

        #lm = lm_33

        lm = self.MLP_lm(lm)
        PPI_emb = emb[:, :400]
        SSN_emb = emb[:, 400:]
        lm = lm

        combined_feat = self.scale_aggr(PPI_emb,SSN_emb,lm)


        # combined_feat = torch.cat([PPI_emb,SSN_emb,lm],dim=-1)
        output = self.classifier(combined_feat)
        output = self.sigmoid(output)
        return output,weight



