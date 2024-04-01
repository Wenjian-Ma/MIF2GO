import torch.nn as nn
import torch
from torch.nn import functional as F

from base_layers.gcn import GraphConvolution,GraphConvolutionSparse

class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim, dropout=0, act=nn.Sigmoid(), **kwargs):
        super(InnerProductDecoder,self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        x = torch.transpose(inputs,0,1)
        x = torch.matmul(inputs, x)
        x = torch.reshape(x, [1,-1])
        outputs = self.act(x)
        return outputs



class GAE(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0, **kwargs):
        super(GAE, self).__init__()

        self.n_samples = num_nodes
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.input_dim = num_features
        # self.adj = adj
        # self.inputs = features
        self.device = device
        self.dropout = dropout
        self.features_nonzero = features_nonzero


        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              features_nonzero=self.features_nonzero,
                                              act=nn.ReLU(),
                                              dropout=self.dropout)
        self.hidden2 = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       act=lambda x: x,
                                       dropout=self.dropout)

        self.hidden3 = GraphConvolution(input_dim=hidden2,
                                       output_dim=num_features,
                                       act=lambda x: x,
                                       dropout=dropout)

        self.InnerProductDecoder = InnerProductDecoder(input_dim=self.hidden2_dim,act = lambda x: x)##decoder
    def forward(self,adj,x):
        x1 = self.hidden1(adj, x)

        z = self.hidden2(adj,x1)

        reconstructions = self.InnerProductDecoder(z)
        # reconstructions = self.hidden3(adj,z)
        return z,reconstructions

class double_GAE(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0, **kwargs):
        super(double_GAE, self).__init__()
        self.GAE1 = GAE(num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout)
        self.GAE2 = GAE(num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout)
    def forward(self,ppi_adj,ppi_fea,ssn_adj,ssn_fea):
        z_ppi,recon_ppi = self.GAE1(ppi_adj,ppi_fea)
        z_ssn,recon_ssn = self.GAE2(ssn_adj,ssn_fea)
        return z_ppi,z_ssn,recon_ppi,recon_ssn


class GCN_encoder(nn.Module):
    def __init__(self,num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0, **kwargs):
        super(GCN_encoder, self).__init__()
        self.hidden1 = GraphConvolutionSparse(input_dim=num_features,
                                              output_dim=hidden1,
                                              features_nonzero=features_nonzero,
                                              act=nn.ReLU(),
                                              dropout=dropout)
        self.hidden2 = GraphConvolution(input_dim=hidden1,
                                        output_dim=hidden2,
                                        act=lambda x: x,
                                        dropout=dropout)
    def forward(self,adj,x):
        x = self.hidden1(adj,x)
        x = self.hidden2(adj,x)
        return x


class GCN_decoder(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout=0.0, **kwargs):
        super(GCN_decoder, self).__init__()
        self.hidden = GraphConvolution(input_dim=hidden2,
                                        output_dim=num_features,
                                        act=lambda x: x,
                                        dropout=dropout)
    def forward(self,adj,x):
        x = self.hidden(adj,x)
        return x
class NoiseGAE(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2,device, eps,noise_rate=None, **kwargs):
        super(NoiseGAE, self).__init__()
        self.InnerProductDecoder = InnerProductDecoder(input_dim=hidden2, act=lambda x: x)

        self.noise_rate = noise_rate

        self.eps = eps
        #encoder-decoder
        self.encoder = GCN_encoder(num_features, num_nodes, features_nonzero, hidden1, hidden2, device, dropout=0.0)

        self.encoder_to_decoder = nn.Linear(hidden2, hidden2, bias=False)

        self.decoder = GCN_decoder(num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0.0)
        ############################




    def forward(self, adj, x,noise_nodes):
        fea = x
        #
        if len(noise_nodes)!=0:
            random_noise = torch.rand_like(x[noise_nodes]).to(adj.device)
            x[noise_nodes] += torch.sign(x[noise_nodes]) * F.normalize(random_noise, dim=-1) * self.eps
        #
        z = self.encoder(adj,x)

        rep = self.encoder_to_decoder(z)

        recon = self.decoder(adj,rep)

        if len(noise_nodes)!=0:
            x_init = fea[noise_nodes]

            x_rec = recon[noise_nodes]
        else:
            x_init = fea

            x_rec = recon
        emb = self.encoder(adj,fea)

        # adj_rec = self.InnerProductDecoder(z)
        return x_init,x_rec,emb,rep,z##############rep or z



class double_NoiseGAE(nn.Module):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2, device, eps, noise_rate=None, **kwargs):
        super(double_NoiseGAE, self).__init__()
        self.NoiseGAE = NoiseGAE(num_features=num_features, num_nodes=num_nodes, features_nonzero=features_nonzero,
                 hidden1=hidden1, hidden2=hidden2, device=device, noise_rate=noise_rate,eps=eps)

        # self.NoiseGAE2 = NoiseGAE(num_features=num_features, num_nodes=num_nodes, features_nonzero=features_nonzero,
        #                           hidden1=hidden1, hidden2=hidden2, device=device, noise_rate=noise_rate, eps=eps)

        self.n_samples = num_nodes
        self.noise_rate = noise_rate

    def random_noise_node(self, adj, noise_rate):

        num_nodes = self.n_samples
        perm = torch.randperm(num_nodes, device=adj.device)


        num_mask_nodes = int(noise_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        return (mask_nodes, keep_nodes)


    def forward(self,ppi_adj,ppi_fea,ssn_adj,ssn_fea):
        (noise_nodes, keep_nodes) = self.random_noise_node(ppi_adj,self.noise_rate)

        x_init_ppi, x_rec_ppi, emb_ppi, z_ppi_rep, z_ppi = self.NoiseGAE(ppi_adj,ppi_fea,noise_nodes)
        x_init_ssn, x_rec_ssn, emb_ssn, z_ssn_rep, z_ssn = self.NoiseGAE(ssn_adj, ssn_fea,noise_nodes)
        return x_init_ppi, x_rec_ppi, emb_ppi,z_ppi,x_init_ssn, x_rec_ssn, emb_ssn,z_ssn