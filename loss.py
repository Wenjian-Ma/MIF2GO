from torch import nn
import torch
from torch.nn import functional as F
    
#SCE loss
def sce_loss(x, y, alpha=None):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


# def InfoNCE(view1, view2, temperature: float = 0.2, b_cos: bool = True):
#     """
#     Args:
#         view1: (torch.Tensor - N x D)
#         view2: (torch.Tensor - N x D)
#         temperature: float
#         b_cos (bool)
#
#     Return: Average InfoNCE Loss
#     """
#     if b_cos:
#         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
#
#     pos_score = (view1 @ view2.T) / temperature
#     score = torch.diag(F.log_softmax(pos_score, dim=1))
#     return -score.mean()

def InfoNCE(z_ppi, z_ssn,one_hop_ppi,one_hop_ssn,mark, temperature: float=0.2, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if mark == 'ppi':
        if b_cos:
            z_ppi, z_ssn = F.normalize(z_ppi, dim=1), F.normalize(z_ssn, dim=1)

        pos_score1 = (z_ppi @ z_ssn.T) / temperature
        score1 = torch.sum(F.log_softmax(pos_score1, dim=1)*one_hop_ssn,dim=1)
        # score = torch.diag(F.log_softmax(pos_score, dim=1))

        pos_score2 = (z_ppi @ z_ppi.T)/ temperature
        score2 = torch.sum(F.log_softmax(pos_score2,dim=1)*one_hop_ppi,dim=1)
        score = score1+score2



    elif mark == 'ssn':
        if b_cos:
            z_ppi, z_ssn = F.normalize(z_ppi, dim=1), F.normalize(z_ssn, dim=1)

        pos_score1 = (z_ssn @ z_ppi.T) / temperature
        score1 = torch.sum(F.log_softmax(pos_score1, dim=1) * one_hop_ppi,dim=1)

        pos_score2 = (z_ssn@z_ssn.T) / temperature
        score2 = torch.sum(F.log_softmax(pos_score2,dim=1)*one_hop_ssn,dim=1)
        score = score1 + score2



    return -score.mean()