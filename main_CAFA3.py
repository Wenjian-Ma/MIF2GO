import argparse
from input_data import load_data_CAFA,load_labels_CAFA
from trainAE import train_NoiseGAE
from trainNN import train_nn
import numpy as np
import pandas as pd
import os,time
import torch
from preprocessing import PFPDataset#,collate
from torch.utils.data import DataLoader
import warnings



def reshape(features):
    matrix = np.hstack(features)
    return matrix.reshape((len(features),len(features[0][0])))


def train(args):
    # load feature dataframe
    device = torch.device('cuda:' + args.device) if args.device != 'cpu' else torch.device('cpu')
    for aspect,(num_labels,num_tests) in {'mf':[677,1137],'bp':[3992,2392],'cc':[551,1265]}.items():

        print("loading features for "+aspect+" ...")
        uniprot = pd.read_pickle(os.path.join(args.data_path, aspect+"_features.pkl"))
        go_term = uniprot.columns.values.tolist()[-num_labels:]

        if aspect+'_embeddings_one_hot_loc_inter_path.npy' not in os.listdir(args.data_path+'/trained_emb_files/'):
            print('#'*20,'Training for '+aspect+' ','#'*20)
            for graph in args.graphs:
                print("#############################")
                print(graph," data...")
                if graph == 'ppi':
                    ppi_adj, ppi_features = load_data_CAFA(graph, uniprot, args,aspect)
                else:
                    ssn_adj, ssn_features = load_data_CAFA(graph, uniprot, args,aspect)

            start_time = time.time()
            embeddings = train_NoiseGAE(ppi_features, ppi_adj,ssn_features,ssn_adj, args,torch.device('cpu'))#out of gpu memory, so using cpu
            EMB_time = (time.time() - start_time) / 3600
            print('Running EMB_time: %f hours' % EMB_time)
            torch.cuda.empty_cache()
            # np.save(args.data_path+'/trained_emb_files/'+aspect+'_embeddings_one_hot_loc_inter_path.npy',embeddings)
        else:
            embeddings = np.load(args.data_path+'/trained_emb_files/'+aspect+'_embeddings_one_hot_loc_inter_path.npy')


        labels = load_labels_CAFA(uniprot,num_labels)
        Y_train = labels[:-num_tests,:]
        Y_test = labels[-num_tests:,:]

        ESM_33 = reshape(uniprot['esm_33'].values)
        ESM_28 = reshape(uniprot['esm_28'].values)
        ESM_23 = reshape(uniprot['esm_23'].values)
        LM_train = [ESM_33[:-num_tests,:], ESM_28[:-num_tests,:], ESM_23[:-num_tests,:]]
        LM_test = [ESM_33[-num_tests:,:], ESM_28[-num_tests:,:], ESM_23[-num_tests:,:]]

        X_train = embeddings[:-num_tests,:]  # 12107，800
        X_test = embeddings[-num_tests:,:]

        train_data = PFPDataset(emb_X=X_train, data_Y=Y_train, args=args, global_lm=LM_train)
        test_data = PFPDataset(emb_X=X_test, data_Y=Y_test, args=args, global_lm=LM_test)

        dataset_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                      num_workers=args.num_workers)
        dataset_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     num_workers=args.num_workers)

        print("Start running supervised model...")

        print("###################################")
        print('----------------------------------')
        print(aspect.upper())
        start_time = time.time()
        train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=num_labels,
                 train_loader=dataset_train, go=go_term, test_loader=dataset_test, term=aspect)
        cost_time = (time.time() - start_time) / 3600
        print('Running '+aspect.upper()+' time: %f hours' % cost_time)



if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--ppi_attributes', type=int, default=1, help="types of attributes used by ppi.(0 for interpro and node2vec ablation, 1 for node2vec ablation, 2 for using all feats)")
    parser.add_argument('--simi_attributes', type=int, default=1, help="types of attributes used by simi.(0 for interpro and node2vec ablation, 1 for node2vec ablation, 2 for using all feats)")
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")],
                        default=['ppi','sequence_similarity'], help="lists of graphs to use.")#'ppi',
    parser.add_argument('--species', type=str, default="CAFA3",
                        help="only CAFA3.")
    parser.add_argument('--data_path', type=str, default="./data/CAFA3/", help="path storing data.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=120, help="Number of epochs to train ppi.")
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