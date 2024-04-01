from GAE_model import double_GAE,double_NoiseGAE
import numpy as np
import torch
from utils import preprocess_graph,sparse_to_tuple
import scipy.sparse as sp
from loss import sce_loss,InfoNCE

def sparse_to_dense(sparse):
    count = 0
    metrics = np.zeros(sparse[2])
    for index in sparse[0]:
        row = int(index[0])
        col = int(index[1])
        metrics[row][col] = sparse[1][count]
        count = count + 1
    return metrics

def process_adj_fea(adj_train,features,device):
    adj = adj_train
    adj_norm = preprocess_graph(adj)
    adj_norm_dense = torch.Tensor(sparse_to_dense(adj_norm)).cuda(device)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label_dense = torch.Tensor(sparse_to_dense(adj_label)).cuda(device)
    feature = torch.Tensor(features.toarray()).cuda(device)
    return adj_norm_dense,adj_label_dense,feature

def train_NoiseGAE(ppi_features, ppi_adj,ssn_features,ssn_adj, args,device):
    epoch = args.epochs

    adj_norm_dense_ppi, adj_label_dense_ppi, feature_ppi = process_adj_fea(ppi_adj,ppi_features,device)
    adj_norm_dense_ssn, adj_label_dense_ssn, feature_ssn = process_adj_fea(ssn_adj, ssn_features, device)

    #np.save('/home/sgzhang/perl5/CFAGO-code/Dataset/human/feat.npy',feature_ppi.cpu().numpy())

    num_nodes = ppi_adj.shape[0]
    features = sparse_to_tuple(ppi_features.tocoo())

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    model = double_NoiseGAE(num_features=num_features, num_nodes=num_nodes, features_nonzero=features_nonzero, hidden1=args.hidden1, hidden2=args.hidden2,device=device,noise_rate=args.noise_rate,eps=args.eps)

    model = model.cuda(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for e in range(epoch):
        model.train()
        model_optimizer.zero_grad()

        x_init_ppi, x_rec_ppi, emb_ppi,z_ppi, x_init_ssn, x_rec_ssn, emb_ssn,z_ssn = model(adj_norm_dense_ppi,feature_ppi,adj_norm_dense_ssn,feature_ssn)
        loss_ppi = sce_loss(x_rec_ppi,x_init_ppi,alpha=args.alpha)
        loss_ssn = sce_loss(x_rec_ssn, x_init_ssn, alpha=args.alpha)
        # loss_cl = (InfoNCE(emb_ppi,emb_ssn,adj_norm_dense_ssn,mark='ppi')+InfoNCE(emb_ppi,emb_ssn,adj_norm_dense_ppi,mark='ssn'))*0.1
        loss_cl = (InfoNCE(z_ppi=z_ppi,z_ssn=z_ssn,one_hop_ppi=adj_norm_dense_ppi,one_hop_ssn=adj_norm_dense_ssn,mark='ppi')+InfoNCE(z_ppi=z_ppi,z_ssn=z_ssn,one_hop_ppi=adj_norm_dense_ppi,one_hop_ssn=adj_norm_dense_ssn,mark='ssn'))*args.lambda_

        #loss_cl = (InfoNCE(z_ppi, z_ssn) + InfoNCE(z_ssn,z_ppi))*0.1

        loss = loss_ppi+loss_ssn+loss_cl
        loss.backward()
        model_optimizer.step()
        print('Epoch '+str(e)+':\ttotal loss\t'+str(loss.item()),'\tppi_loss:\t',str(loss_ppi.item()),'\tssn_loss:\t',str(loss_ssn.item()),'\tcl_loss:\t',str(loss_cl.item()))

    emb = torch.cat((emb_ppi,emb_ssn),1).cpu().detach().numpy()
    return emb



def train_GAE(ppi_features, ppi_adj,ssn_features,ssn_adj, args,device):
    epoch = args.epochs

    adj_norm_dense_ppi, adj_label_dense_ppi, feature_ppi = process_adj_fea(ppi_adj,ppi_features,device)
    adj_norm_dense_ssn, adj_label_dense_ssn, feature_ssn = process_adj_fea(ssn_adj, ssn_features, device)

    num_nodes = ppi_adj.shape[0]
    features = sparse_to_tuple(ppi_features.tocoo())

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    model = double_GAE(num_features=num_features, num_nodes=num_nodes, features_nonzero=features_nonzero, hidden1=args.hidden1, hidden2=args.hidden2,device=device)

    model = model.cuda(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    BCEloss = torch.nn.BCEWithLogitsLoss()
    for e in range(epoch):
        model.train()
        model_optimizer.zero_grad()

        emb_ppi,emb_ssn,recon_ppi,recon_ssn = model(adj_norm_dense_ppi,feature_ppi,adj_norm_dense_ssn,feature_ssn)
        loss_ppi = BCEloss(recon_ppi.reshape(1,-1),adj_label_dense_ppi.reshape(1,-1))
        loss_ssn = BCEloss(recon_ssn.reshape(1,-1),adj_label_dense_ssn.reshape(1,-1))

        # loss_ppi = sce_loss(recon_ppi,feature_ppi,alpha=args.alpha)
        # loss_ssn = sce_loss(recon_ssn, feature_ssn, alpha=args.alpha)

        loss = loss_ppi+loss_ssn
        loss.backward()
        model_optimizer.step()
        print('Epoch '+str(e)+':\ttotal loss\t'+str(loss.item()),'\tppi_loss:\t',str(loss_ppi.item()),'\tssn_loss:\t',str(loss_ssn.item()))

    emb = torch.cat((emb_ppi,emb_ssn),1).cpu().detach().numpy()
    return emb