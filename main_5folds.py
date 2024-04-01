import argparse
from input_data import load_data,load_labels
from trainAE import train_NoiseGAE
import numpy as np
import pandas as pd
import os,sys
from preprocessing import PFPDataset
from torch.utils.data import DataLoader
import warnings
# from evaluation import get_results
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from nn_Model import nnModel

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score





def get_label_frequency(ontology):
    col_sums = ontology.sum(0)
    index_11_30 = np.where((col_sums >= 11) & (col_sums <= 30))[0]
    index_31_100 = np.where((col_sums >= 31) & (col_sums <= 100))[0]
    index_101_300 = np.where((col_sums >= 101) & (col_sums <= 300))[0]
    index_larger_300 = np.where(col_sums >= 301)[0]
    return index_11_30, index_31_100, index_101_300, index_larger_300


def calculate_accuracy(y_test, y_score):
    y_score_max = np.argmax(y_score, axis=1)
    cnt = 0
    for i in range(y_score.shape[0]):
        if y_test[i, y_score_max[i]] == 1:
            cnt += 1

    return float(cnt) / y_score.shape[0]


def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max

def evaluate_performance(y_test, y_score):
    """Evaluate performance"""

    n_classes = y_test.shape[1]

    perf = dict()

    perf["M-aupr"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            aupr_list.append(ap)
            num_pos_list.append(str(num_pos))
    perf["M-aupr"] /= n
    perf['aupr_list'] = aupr_list
    perf['num_pos_list'] = num_pos_list

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())

    # Computes accuracy
    perf['accuracy'] = calculate_accuracy(y_test, y_score)

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha * [1])
    perf["F1-score"] = f1_score(y_test, y_new_pred, average='micro')

    perf['F-max'] = calculate_fmax(y_score, y_test)

    return perf


def get_results(ontology, Y_test, y_score):
    #print(Y_test.shape)
    #print(y_score.shape)
    perf = defaultdict(dict)
    # index_11_30, index_31_100, index_101_300, index_301 = get_label_frequency(ontology)

    # perf['11-30'] = evaluate_performance(Y_test[:, index_11_30], y_score[:, index_11_30])
    # perf['31-100'] = evaluate_performance(Y_test[:, index_31_100], y_score[:, index_31_100])
    # perf['101-300'] = evaluate_performance(Y_test[:, index_101_300], y_score[:, index_101_300])
    # perf['301-'] = evaluate_performance(Y_test[:, index_301], y_score[:, index_301])
    perf['all'] = evaluate_performance(Y_test, y_score)

    return perf




def train_nn(args,train_loader,device,input_dim,output_dim,go,test_loader,term):


    Epoch = 50

    model = nnModel(output_dim,dropout=0.2,device=device,args=args)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr/2)  #  0.0005

    bceloss = nn.BCELoss()

    weight_dict = {'lm_33':{},'lm_28':{},'lm_23':{}}

    M_aupr = []
    m_aupr = []
    F_max = []

    for idx,e in enumerate(range(Epoch)):
        model.train()
        weight_dict['lm_33'][idx] = []
        weight_dict['lm_28'][idx] = []
        weight_dict['lm_23'][idx] = []


        for batch_idx,batch in enumerate(tqdm(train_loader,mininterval=0.5,desc='Training',leave=False,ncols=50)):
            optimizer.zero_grad()
            emb = batch[0].to(device)
            Y_label = batch[1].to(device)
            lm_33 = batch[2].to(device)
            lm_28 = batch[3].to(device)
            lm_23 = batch[4].to(device)

            # Y_label = batch[1].to(device)
            # emb = batch[0].to(device)
            # lm = batch[2].to(device)

            output,weight = model(emb.squeeze(),lm_33.squeeze(),lm_28.squeeze(),lm_23.squeeze())
            loss_out = bceloss(output, Y_label.squeeze())
            loss_out.backward()
            optimizer.step()

            weight_dict['lm_33'][idx].append(float(weight[0].cpu()))
            weight_dict['lm_28'][idx].append(float(weight[1].cpu()))
            weight_dict['lm_23'][idx].append(float(weight[2].cpu()))

        model.eval()
        total_preds = torch.Tensor().to(device)
        total_labels = torch.Tensor().to(device)
        with torch.no_grad():
            for batch_test_idx,batch_test in enumerate(tqdm(test_loader,mininterval=0.5,desc='Testing',leave=False,ncols=50)):


                label_test = batch_test[1].to(device)
                emb_test = batch_test[0].to(device)
                lm_33_test = batch_test[2].to(device)
                lm_28_test = batch_test[3].to(device)
                lm_23_test = batch_test[4].to(device)


                output_test,_ = model(emb_test.squeeze(),lm_33_test.squeeze(),lm_28_test.squeeze(),lm_23_test.squeeze())
                total_preds = torch.cat((total_preds, output_test), 0)
                total_labels = torch.cat((total_labels, label_test.squeeze()), 0)


            loss_test = bceloss(total_preds,total_labels)


        perf = get_results(go, total_labels.cpu().numpy(), total_preds.cpu().numpy())


        print('Epoch ' + str(e + 1) + '\tTrain loss:\t', loss_out.item(), '\tTest loss:\t',loss_test.item(), '\n\tM-AUPR:\t', perf['all']['M-aupr'], '\tm-AUPR:\t', perf['all']['m-aupr'],'\tF-max:\t', perf['all']['F-max'])

        M_aupr.append(perf['all']['M-aupr'])
        m_aupr.append(perf['all']['m-aupr'])
        F_max.append(perf['all']['F-max'])

    best_epoch = M_aupr.index(max(M_aupr))

    return np.array([M_aupr[best_epoch],m_aupr[best_epoch],F_max[best_epoch]])








def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))


def train(args):

    print("loading features...")
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))

    device = torch.device('cuda:'+args.device)

    if 'embeddings_5folds.npy' not in os.listdir('./data/' + args.species + '/trained_emb_files/'):

        for graph in args.graphs:
            print("#############################")
            print(graph," data...")
            if graph == 'ppi':
                ppi_adj, ppi_features = load_data(graph, uniprot, args)
            else:
                ssn_adj, ssn_features = load_data(graph, uniprot, args)
        embeddings = train_NoiseGAE(ppi_features, ppi_adj,ssn_features,ssn_adj, args,device)
        np.save('./data/'+args.species+'/trained_emb_files/embeddings_5folds.npy',embeddings)
    else:
        embeddings = np.load('./data/'+args.species+'/trained_emb_files/embeddings_5folds.npy')


    np.random.seed(5959)

    cc, mf, bp = load_labels(uniprot)

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

    fold_num = int(np.floor(num_train / 5.))

    MF_recorder = []
    BP_recorder = []
    CC_recorder = []

    five_folds = [train_idx[:fold_num],train_idx[fold_num:fold_num*2],train_idx[fold_num*2:fold_num*3],train_idx[fold_num*3:fold_num*4],train_idx[fold_num*4:]]
    for fold in [0,1,2,3,4]:
        train_folds_idx = train_idx.copy()
        print('\n#######Fold\t'+str(fold+1)+'...\t#######')

        valid_folds_idx = five_folds[fold]
        for i in valid_folds_idx:
            train_folds_idx.remove(i)

        ESM_33 = np.load('./data/' + args.species + '/ESM-2_33.npy')  # ESM-embeddings.npy
        ESM_28 = np.load('./data/' + args.species + '/ESM-2_28.npy')
        ESM_23 = np.load('./data/' + args.species + '/ESM-2_23.npy')

        Y_train_cc = cc[train_folds_idx]
        Y_train_bp = bp[train_folds_idx]
        Y_train_mf = mf[train_folds_idx]

        Y_test_cc = cc[valid_folds_idx]
        Y_test_bp = bp[valid_folds_idx]
        Y_test_mf = mf[valid_folds_idx]

        X_train = embeddings[train_folds_idx]
        X_test = embeddings[valid_folds_idx]

        LM_train = [ESM_33[train_folds_idx], ESM_28[train_folds_idx], ESM_23[train_folds_idx]]
        LM_test = [ESM_33[valid_folds_idx], ESM_28[valid_folds_idx], ESM_23[valid_folds_idx]]
        ##########################

        train_data_cc = PFPDataset(emb_X=X_train, data_Y=Y_train_cc, args=args, global_lm=LM_train)
        train_data_bp = PFPDataset(emb_X=X_train, data_Y=Y_train_bp, args=args, global_lm=LM_train)
        train_data_mf = PFPDataset(emb_X=X_train, data_Y=Y_train_mf, args=args, global_lm=LM_train)

        test_data_cc = PFPDataset(emb_X=X_test, data_Y=Y_test_cc, args=args, global_lm=LM_test)
        test_data_bp = PFPDataset(emb_X=X_test, data_Y=Y_test_bp, args=args, global_lm=LM_test)
        test_data_mf = PFPDataset(emb_X=X_test, data_Y=Y_test_mf, args=args, global_lm=LM_test)

        dataset_train_cc = DataLoader(train_data_cc, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                      num_workers=args.num_workers)
        dataset_train_bp = DataLoader(train_data_bp, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                      num_workers=args.num_workers)
        dataset_train_mf = DataLoader(train_data_mf, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                      num_workers=args.num_workers)

        dataset_test_cc = DataLoader(test_data_cc, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     num_workers=args.num_workers)
        dataset_test_bp = DataLoader(test_data_bp, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     num_workers=args.num_workers)
        dataset_test_mf = DataLoader(test_data_mf, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                     num_workers=args.num_workers)

        print("Start running supervised model...")


        print("###################################")
        print('----------------------------------')
        print('MF')

        MF_performance = train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_mf.shape[1],
                 train_loader=dataset_train_mf, go=mf, test_loader=dataset_test_mf, term='MF')

        MF_recorder.append(MF_performance)

        print("###################################")
        print('----------------------------------')
        print('BP')

        BP_performance = train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_bp.shape[1],
                 train_loader=dataset_train_bp, go=bp, test_loader=dataset_test_bp, term='BP')

        BP_recorder.append(BP_performance)

        print("###################################")
        print('----------------------------------')
        print('CC')

        CC_performance = train_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_train_cc.shape[1],
                 train_loader=dataset_train_cc, go=cc, test_loader=dataset_test_cc, term='CC')

        CC_recorder.append(CC_performance)

    MF = np.vstack(MF_recorder)
    BP = np.vstack(BP_recorder)
    CC = np.vstack(CC_recorder)

    MF_mean = np.mean(MF,axis=0)
    MF_std = np.std(MF,axis=0)

    BP_mean = np.mean(BP,axis=0)
    BP_std = np.std(BP,axis=0)

    CC_mean = np.mean(CC,axis=0)
    CC_std = np.std(CC,axis=0)

    with open('./data/Human/5folds_result/'+args.result_file+'.txt','a') as f:
        f.write('MF:'+'\n\n')
        f.write('M-AUPR:\n')
        f.write(str(MF_mean[0])+'\n'+str(MF_std[0])+'\n')

        f.write('m-AUPR:\n')
        f.write(str(MF_mean[1]) + '\n' + str(MF_std[1])+'\n')

        f.write('F-max:\n')
        f.write(str(MF_mean[2]) + '\n' + str(MF_std[2])+'\n')

        f.write('\nBP:' + '\n\n')
        f.write('M-AUPR:\n')
        f.write(str(BP_mean[0]) + '\n' + str(BP_std[0])+'\n')

        f.write('m-AUPR:\n')
        f.write(str(BP_mean[1]) + '\n' + str(BP_std[1])+'\n')

        f.write('F-max:\n')
        f.write(str(BP_mean[2]) + '\n' + str(BP_std[2])+'\n')

        f.write('\nCC:' + '\n\n')
        f.write('M-AUPR:\n')
        f.write(str(CC_mean[0]) + '\n' + str(CC_std[0])+'\n')

        f.write('m-AUPR:\n')
        f.write(str(CC_mean[1]) + '\n' + str(CC_std[1])+'\n')

        f.write('F-max:\n')
        f.write(str(CC_mean[2]) + '\n' + str(CC_std[2])+'\n')




if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--ppi_attributes', type=int, default=5,
                        help="types of attributes used by ppi.(1 for pathway ablation, 2 for location ablation, 3 for domain ablation, 5 for using all feature)")
    parser.add_argument('--simi_attributes', type=int, default=5,
                        help="types of attributes used by simi.(1 for pathway ablation, 2 for location ablation, 3 for domain ablation, 5 for using all feature)")
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")],
                        default=['ppi', 'sequence_similarity'], help="lists of graphs to use.")  # 'ppi',
    parser.add_argument('--species', type=str, default="Human",
                        help="which species to use (Human/scerevisiae/rat/mouse/fly/ecoli).")
    parser.add_argument('--data_path', type=str, default="./data/", help="path storing data.")
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=160, help="Number of epochs to train ppi.")
    parser.add_argument('--device', type=str, default='6', help="cuda device.")
    parser.add_argument('--thr_combined', type=float, default=0.3, help="threshold for combiend ppi network.")  # 0.3
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")  # 1e-4
    parser.add_argument('--noise_rate', type=float, default=0.3, help="noise rate.")
    parser.add_argument('--alpha', type=int, default=2, help="alpha for sce_loss.")
    parser.add_argument('--eps', type=float, default=0.01, help="Eps for Noise.")
    parser.add_argument('--heads', type=int, default=4, help="Attention heads.")
    parser.add_argument('--lambda_', type=float, default=0.4, help="Coefficient for CL loss.")

    parser.add_argument('--num_workers', type=int, default=8, help="num_workers.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size.")
    parser.add_argument('--result_file', type=str, required=True, help="file name to restore 5folds result .")


    args = parser.parse_args()
    print(args)
    train(args)