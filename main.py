import argparse
from input_data import load_data,load_labels
from trainAE import train_NoiseGAE,train_GAE
from trainNN import train_nn
import numpy as np
import pandas as pd
import os
import torch
from preprocessing import PFPDataset#,collate
from torch.utils.data import DataLoader
import warnings
from scipy import sparse
def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))


def train(args):
    # load feature dataframe
    print("loading features...")
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))

    device = torch.device('cuda:'+args.device)
    if 'embeddings.npy' not in os.listdir('./data/'+args.species+'/trained_emb_files/'):

        for graph in args.graphs:
            print("#############################")
            print(graph," data...")
            if graph == 'ppi':
                ppi_adj, ppi_features = load_data(graph, uniprot, args)
            else:
                ssn_adj, ssn_features = load_data(graph, uniprot, args)
        # sparse.save_npz('/home/sgzhang/perl5/CFAGO-code/Dataset/human/adj.npz', ppi_adj)
        embeddings = train_NoiseGAE(ppi_features, ppi_adj,ssn_features,ssn_adj, args,device)
        #embeddings = train_GAE(ppi_features, ppi_adj, ssn_features, ssn_adj, args, device)
        #np.save('./data/'+args.species+'/trained_emb_files/embeddings.npy',embeddings)
    else:
        embeddings = np.load('./data/'+args.species+'/trained_emb_files/embeddings.npy')

    np.random.seed(5959)

    cc, mf, bp = load_labels(uniprot)

    # split data into train and test
    num_test = int(np.floor(cc.shape[0] / 5.))
    num_train = cc.shape[0] - num_test

    if 'data_idx.txt' not in os.listdir('./data/'+args.species):
        all_idx = list(range(cc.shape[0]))
        np.random.shuffle(all_idx)

        with open('./data/'+args.species+'/data_idx.txt','a') as f:
            for idx in all_idx:
                f.write(str(idx)+'\n')
    else:
        all_idx = []
        with open('./data/'+args.species+'/data_idx.txt') as f:
            for line in f:
                all_idx.append(int(line.strip()))


    train_idx = all_idx[:num_train]
    test_idx = all_idx[num_train:(num_train + num_test)]


    ESM_33 = np.load('./data/'+args.species+'/ESM-2_33.npy')#ESM-embeddings.npy
    ESM_28 = np.load('./data/' + args.species + '/ESM-2_28.npy')
    ESM_23 = np.load('./data/' + args.species + '/ESM-2_23.npy')


    Y_train_cc = cc[train_idx]
    Y_train_bp = bp[train_idx]
    Y_train_mf = mf[train_idx]

    Y_test_cc = cc[test_idx]
    Y_test_bp = bp[test_idx]
    Y_test_mf = mf[test_idx]



    X_train = embeddings[train_idx]  # 12107，800
    X_test = embeddings[test_idx]  # 3026，800


    LM_train = [ESM_33[train_idx],ESM_28[train_idx],ESM_23[train_idx]]
    LM_test = [ESM_33[test_idx],ESM_28[test_idx],ESM_23[test_idx]]
    ##########################

    train_data_cc = PFPDataset(emb_X=X_train, data_Y=Y_train_cc,args=args,global_lm = LM_train)
    train_data_bp = PFPDataset(emb_X=X_train, data_Y=Y_train_bp,args=args,global_lm = LM_train)
    train_data_mf = PFPDataset(emb_X=X_train, data_Y=Y_train_mf,args=args,global_lm = LM_train)

    test_data_cc = PFPDataset(emb_X=X_test, data_Y=Y_test_cc,args=args,global_lm = LM_test)
    test_data_bp = PFPDataset(emb_X=X_test, data_Y=Y_test_bp,args=args,global_lm = LM_test)
    test_data_mf = PFPDataset(emb_X=X_test, data_Y=Y_test_mf,args=args,global_lm = LM_test)

    dataset_train_cc = DataLoader(train_data_cc, batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=args.num_workers)
    dataset_train_bp = DataLoader(train_data_bp, batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=args.num_workers)
    dataset_train_mf = DataLoader(train_data_mf, batch_size=args.batch_size, shuffle=True, drop_last=False,num_workers=args.num_workers)

    dataset_test_cc = DataLoader(test_data_cc, batch_size=args.batch_size, shuffle=False,  drop_last=False,num_workers=args.num_workers)
    dataset_test_bp = DataLoader(test_data_bp, batch_size=args.batch_size, shuffle=False,  drop_last=False,num_workers=args.num_workers)
    dataset_test_mf = DataLoader(test_data_mf, batch_size=args.batch_size, shuffle=False,  drop_last=False,num_workers=args.num_workers)

    print("Start running supervised model...")

    print("###################################")
    print('----------------------------------')
    print('MF')

    train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_mf.shape[1],
             train_loader=dataset_train_mf,go=mf, test_loader=dataset_test_mf,term='mf')

    print("###################################")
    print('----------------------------------')
    print('BP')

    train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_bp.shape[1],
             train_loader=dataset_train_bp, go=bp, test_loader=dataset_test_bp,term='bp')

    print("###################################")
    print('----------------------------------')
    print('CC')

    train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_cc.shape[1],
             train_loader=dataset_train_cc, go=cc, test_loader=dataset_test_cc,term='cc')


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--ppi_attributes', type=int, default=5, help="types of attributes used by ppi.(1 for pathway ablation, 2 for location ablation, 3 for domain ablation, 5 for using all feature)")
    parser.add_argument('--simi_attributes', type=int, default=5, help="types of attributes used by simi.(1 for pathway ablation, 2 for location ablation, 3 for domain ablation, 5 for using all feature)")
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")],
                        default=['ppi','sequence_similarity'], help="lists of graphs to use.")#'ppi',
    parser.add_argument('--species', type=str, default="Human", help="which species to use (Human/scerevisiae/rat/mouse/fly/ecoli).")
    parser.add_argument('--data_path', type=str, default="./data/", help="path storing data.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=160, help="Number of epochs to train ppi.")
    parser.add_argument('--device', type=str, default='1', help="cuda device.")
    parser.add_argument('--thr_combined', type=float, default=0.4, help="threshold for combiend ppi network.")#0.4
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")# 1e-4
    parser.add_argument('--noise_rate', type=float, default=0.6, help="noise rate.")
    parser.add_argument('--alpha', type=int, default=2, help="alpha for sce_loss.")
    parser.add_argument('--eps', type=float, default=2.0, help="Eps for Noise.")
    parser.add_argument('--heads', type=int, default=4, help="Attention heads.")
    parser.add_argument('--lambda_', type=float, default=0.4, help="Coefficient for CL loss.")

    parser.add_argument('--num_workers', type=int, default=8, help="num_workers.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size.")
    parser.add_argument('--save_model', type=bool, default=False, help="save the trained model or not.")
    ################################################################

    ################################################################

    args = parser.parse_args()
    print(args)
    train(args)