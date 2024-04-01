import warnings
import argparse
import pandas as pd
import torch,os
import numpy as np
from input_data import load_labels
from preprocessing import PFPDataset
from torch.utils.data import DataLoader
from nn_Model import nnModel
import torch.nn as nn
from evaluation import get_results
from tqdm import tqdm

def test_nn(args,device,input_dim,output_dim,go,test_loader,term):

    model = nnModel(output_dim, dropout=0.2, device=device, args=args)

    model_path = './data/'+args.species+'/trained_model/'+term+'/model.pkl'

    model = model.to(device)

    model.load_state_dict(torch.load(model_path,map_location=device))

    bceloss = nn.BCELoss()

    model.eval()

    total_preds = torch.Tensor().to(device)
    total_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for batch_test_idx, batch_test in enumerate(tqdm(test_loader, mininterval=0.5, desc='Testing', leave=False, ncols=50)):
            label_test = batch_test[1].to(device)
            emb_test = batch_test[0].to(device)
            lm_33_test = batch_test[2].to(device)
            lm_28_test = batch_test[3].to(device)
            lm_23_test = batch_test[4].to(device)

            output_test, _ = model(emb_test.squeeze(), lm_33_test.squeeze(), lm_28_test.squeeze(), lm_23_test.squeeze())
            total_preds = torch.cat((total_preds, output_test), 0)
            total_labels = torch.cat((total_labels, label_test.squeeze()), 0)

        loss_test = bceloss(total_preds, total_labels)

    perf = get_results(go, total_labels.cpu().numpy(), total_preds.cpu().numpy())

    print('\tTest loss:\t', loss_test.item(),'\n\tM-AUPR:\t', perf['all']['M-aupr'], '\tm-AUPR:\t', perf['all']['m-aupr'], '\tF-max:\t',perf['all']['F-max'])


def test(args):
    print("loading features...")
    uniprot = pd.read_pickle(os.path.join(args.data_path, args.species, "features.pkl"))

    device = torch.device('cuda:' + args.device)
    if 'embeddings.npy' not in os.listdir('./data/' + args.species + '/trained_emb_files/'):
        raise Exception('Please run main.py for generating embeddings.npy...')
    else:
        embeddings = np.load('./data/' + args.species + '/trained_emb_files/embeddings.npy')

    np.random.seed(5959)

    cc, mf, bp = load_labels(uniprot)

    # split data into train and test
    num_test = int(np.floor(cc.shape[0] / 5.))
    num_train = cc.shape[0] - num_test

    if 'data_idx.txt' not in os.listdir('./data/' + args.species):
        all_idx = list(range(cc.shape[0]))
        np.random.shuffle(all_idx)

        with open('./data/' + args.species + '/data_idx.txt', 'a') as f:
            for idx in all_idx:
                f.write(str(idx) + '\n')
    else:
        all_idx = []
        with open('./data/' + args.species + '/data_idx.txt') as f:
            for line in f:
                all_idx.append(int(line.strip()))
    test_idx = all_idx[num_train:(num_train + num_test)]

    ESM_33 = np.load('./data/' + args.species + '/ESM-2_33.npy')  # ESM-embeddings.npy
    ESM_28 = np.load('./data/' + args.species + '/ESM-2_28.npy')
    ESM_23 = np.load('./data/' + args.species + '/ESM-2_23.npy')

    Y_test_cc = cc[test_idx]
    Y_test_bp = bp[test_idx]
    Y_test_mf = mf[test_idx]

    X_test = embeddings[test_idx]

    LM_test = [ESM_33[test_idx], ESM_28[test_idx], ESM_23[test_idx]]

    test_data_cc = PFPDataset(emb_X=X_test, data_Y=Y_test_cc, args=args, global_lm=LM_test)
    test_data_bp = PFPDataset(emb_X=X_test, data_Y=Y_test_bp, args=args, global_lm=LM_test)
    test_data_mf = PFPDataset(emb_X=X_test, data_Y=Y_test_mf, args=args, global_lm=LM_test)

    dataset_test_cc = DataLoader(test_data_cc, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    dataset_test_bp = DataLoader(test_data_bp, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    dataset_test_mf = DataLoader(test_data_mf, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)

    print("Start running trained model...")

    print("###################################")
    print('----------------------------------')
    print('MF')

    test_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_test_mf.shape[1],
              go=mf, test_loader=dataset_test_mf, term='mf')

    print("###################################")
    print('----------------------------------')
    print('BP')

    test_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_test_bp.shape[1],
              go=bp, test_loader=dataset_test_bp, term='bp')

    print("###################################")
    print('----------------------------------')
    print('CC')

    test_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_test_cc.shape[1],
              go=cc, test_loader=dataset_test_cc, term='cc')








if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--species', type=str, default="Human",help="which species to use (Human/scerevisiae/rat/mouse/fly/ecoli).")
    parser.add_argument('--data_path', type=str, default="./data/", help="path storing data.")
    parser.add_argument('--device', type=str, default='0', help="cuda device.")
    # parser.add_argument('--thr_combined', type=float, default=0.4, help="threshold for combiend ppi network.")  # 0.4
    # parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")  # 1e-4
    parser.add_argument('--heads', type=int, default=4, help="Attention heads.")

    parser.add_argument('--num_workers', type=int, default=8, help="num_workers.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size.")

    ################################################################

    ################################################################

    args = parser.parse_args()
    print(args)
    test(args)