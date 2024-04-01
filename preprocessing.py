from torch_geometric.data import InMemoryDataset
import torch


class PFPDataset(InMemoryDataset):
    def __init__(self, dir=None, emb_X=None,data_Y=None,transform=None,pre_transform=None,args=None,global_lm=None):

        super(PFPDataset, self).__init__( transform, pre_transform)
        self.dir=dir
        self.X_data_list = emb_X
        self.Y_data_list = data_Y
        self.args = args
        self.global_lm = global_lm
    def __len__(self):
        return int(self.X_data_list.shape[0])

    def __getitem__(self, idx):
        embedding = self.X_data_list[idx]
        embedding = torch.Tensor([embedding])
        # uid = self.uid[idx]
        label = self.Y_data_list[idx]
        label = torch.Tensor([label])
        lm_33 = self.global_lm[0][idx]
        lm_33 = torch.Tensor([lm_33])

        lm_28 = self.global_lm[1][idx]
        lm_28 = torch.Tensor([lm_28])

        lm_23 = self.global_lm[2][idx]
        lm_23 = torch.Tensor([lm_23])

        return embedding,label,lm_33,lm_28,lm_23,idx


