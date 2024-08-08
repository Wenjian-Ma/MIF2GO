import warnings
import argparse,time
import pandas as pd
import torch,os
import numpy as np
from input_data import load_labels_CAFA
from preprocessing import PFPDataset
from torch.utils.data import DataLoader
from nn_Model import nnModel
import torch.nn as nn
from evaluation import get_results
from tqdm import tqdm

def reshape(features):
    matrix = np.hstack(features)
    return matrix.reshape((len(features),len(features[0][0])))

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
    return total_preds.cpu().numpy(),total_labels.cpu().numpy()

def test(args):
    device = torch.device('cuda:' + args.device) if args.device != 'cpu' else torch.device('cpu')

    for aspect, (num_labels, num_tests) in {'mf': [677, 1137], 'bp': [3992, 2392], 'cc': [551, 1265]}.items():

        print("loading features...")
        uniprot = pd.read_pickle(os.path.join(args.data_path, aspect+"_features.pkl"))
        go_term = uniprot.columns.values.tolist()[-num_labels:]

        if aspect+'_embeddings_one_hot_loc_inter_path.npy' not in os.listdir(args.data_path+'/trained_emb_files/'):
            raise Exception('Please run main.py for generating embeddings.npy...')
        else:
            embeddings = np.load(args.data_path+'/trained_emb_files/'+aspect+'_embeddings_one_hot_loc_inter_path.npy')

        labels = load_labels_CAFA(uniprot, num_labels)

        Y_test = labels[-num_tests:, :]

        ESM_33 = reshape(uniprot['esm_33'].values)
        ESM_28 = reshape(uniprot['esm_28'].values)
        ESM_23 = reshape(uniprot['esm_23'].values)

        LM_test = [ESM_33[-num_tests:, :], ESM_28[-num_tests:, :], ESM_23[-num_tests:, :]]


        X_test = embeddings[-num_tests:, :]


        test_data = PFPDataset(emb_X=X_test, data_Y=Y_test, args=args, global_lm=LM_test)


        dataset_test = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                  num_workers=args.num_workers)

        print("Start running supervised model...")

        print("###################################")
        print('----------------------------------')
        print(aspect.upper())
        start_time = time.time()

        print("Start evaluating model...")

        pred,label  = test_nn(args=args, device=device, input_dim=embeddings.shape[1], output_dim=Y_test.shape[1],
                  go=go_term, test_loader=dataset_test, term=aspect)







if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--species', type=str, default="CAFA3",
                        help="only CAFA3.")
    parser.add_argument('--data_path', type=str, default="./data/CAFA3/", help="path storing data.")
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