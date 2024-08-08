import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import scale
import os
from tqdm import tqdm


def load_ppi_network(filename, gene_num, thr):
    count = 0
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        if float(temp[2]) >= thr:
            count = count + 1
            #adj[int(temp[0]), int(temp[1])] = 1
            adj[int(temp[0]), int(temp[1])] = float(temp[2])
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T
        adj[adj > 1] = 1
    # print(count)
    # exit()
    return adj

def load_simi_network(filename, gene_num, thr):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        # check whether evalue smaller than the threshold
        if float(temp[2]) <= thr:
            adj[int(temp[0]), int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T
        adj[adj>1] = 1
    # count1 = np.sum(adj)
    # print(count1)
    # exit()
    return adj


def load_labels(uniprot):
    print('loading labels...')
    # load labels (GO)
    cc = uniprot['cc_label'].values
    cc = np.hstack(cc).reshape((len(cc), len(cc[0])))

    bp = uniprot['bp_label'].values
    bp = np.hstack(bp).reshape((len(bp), len(bp[0])))

    mf = uniprot['mf_label'].values
    mf = np.hstack(mf).reshape((len(mf), len(mf[0])))

    return cc, mf, bp


def load_data(graph_type, uniprot, args):
    print('loading data...')

    def reshape(features):
        return np.hstack(features).reshape((len(features), len(features[0])))

    # get feature representations
    features_loc = reshape(uniprot['Sub_cell_loc_encoding'].values)
    features_domain = reshape(uniprot['Pro_domain_encoding'].values)
    features_pathway = reshape(uniprot['Pathway'].values)
    # features_pathway = np.load('./data/'+args.species+'/pathway.npy')
    print('generating features...')

    if graph_type == "ppi":
        attribute = args.ppi_attributes
    elif graph_type == "sequence_similarity":
        attribute = args.simi_attributes

    if attribute == 0:
        features = np.identity(uniprot.shape[0])
        print("Without features")
    elif attribute == 1:
        features = np.concatenate((features_loc,features_domain),axis=1)
        print("Without pathway feature")
    elif attribute == 2:
        features = np.concatenate((features_domain,features_pathway),axis=1)
        print("Without location feature")
    elif attribute == 3:
        features = np.concatenate((features_loc,features_pathway),axis=1)
        print("Without domain feature")
    elif attribute == 5:
        features = np.concatenate((features_loc, features_domain,features_pathway), axis=1)
        print("use all features")
    features = sp.csr_matrix(features)

    print('loading graph...')
    if graph_type == "sequence_similarity":
        filename = os.path.join(args.data_path, args.species, "sequence_similarity.txt")
        adj = load_simi_network(filename, uniprot.shape[0], args.thr_evalue)
    elif graph_type == "ppi":
        filename = os.path.join(args.data_path, args.species, "ppi.txt")
        adj = load_ppi_network(filename, uniprot.shape[0], args.thr_combined)

    adj = sp.csr_matrix(adj)

    return adj, features


def load_data_CAFA(graph_type, uniprot, args,aspect):
    print('loading data...')

    def reshape(features):
        return np.hstack(features).reshape((len(features), len(features[0])))

    # get feature representations
    features_loc = reshape(uniprot['location'].values)
    features_domain = reshape(uniprot['domain'].values)
    features_pathway = reshape(uniprot['pathway'].values)
    features_interpro = reshape(uniprot['interpro'].values)
    features_ppi_node2vec = reshape(uniprot['ppi_node2vec'].values)
    features_ssn_node2vec = reshape(uniprot['ssn_node2vec'].values)

    print('generating features...')

    if graph_type == "ppi":
        attribute = args.ppi_attributes
    elif graph_type == "sequence_similarity":
        attribute = args.simi_attributes

    if attribute == 0:
        features = np.concatenate((features_loc,features_domain,features_pathway),axis=1)
        print("Using location, domain, and pathway features")
    elif attribute == 1:
        features = np.concatenate((features_loc,features_interpro,features_pathway),axis=1)
        print("Using location, domain, pathway, and interpro features")
    elif attribute == 2:
        features = features_loc#features_loc,features_domain,features_pathway,features_interpro,
        print("Using all features")
    # features = sp.csr_matrix(features)

    print('loading graph...')
    if graph_type == "sequence_similarity":
        adj = np.zeros((uniprot.shape[0],uniprot.shape[0]))
        filename = os.path.join(args.data_path,aspect+"_phn.npy")
        adj_ssn = np.load(filename)#
        adj[adj_ssn <= args.thr_evalue] = 1
        if (adj.T == adj).all():
            pass
        else:
            adj = adj + adj.T
            adj[adj > 1] = 1
    elif graph_type == "ppi":
        filename = os.path.join(args.data_path, aspect+ "_ppi.npy")
        adj = np.load(filename)/1000
        adj[adj<args.thr_combined] = 0

    np.fill_diagonal(adj,0)

    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)

    return adj, features


def load_labels_CAFA(uniprot,num_labels):
    print('loading labels...')
    # load labels (GO)
    labels = np.zeros((uniprot.shape[0],num_labels))
    for idx,i in enumerate(list(uniprot.columns)[-num_labels:]):
        labels[:,idx]  = uniprot[i]



    return labels
