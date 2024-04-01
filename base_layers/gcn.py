import torch
import torch.nn as nn

def dropout_sparse(x, keep_prob, num_nonzero_elems):#keep_prob=1
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += torch.rand(noise_shape)
    return x

class GraphConvolutionSparse(nn.Module):
    def __init__(self,input_dim, output_dim,features_nonzero, dropout=0., act=nn.ReLU(), **kwargs):
        super(GraphConvolutionSparse,self).__init__()
        w = nn.Parameter(torch.Tensor(input_dim,output_dim))
        self.weights = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        self.dropout = dropout
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
    def forward(self,adj,inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = torch.matmul(x,self.weights)
        x = torch.matmul(adj,x)

        outputs = self.act(x)
        return outputs


class GraphConvolution(nn.Module):
    def __init__(self,input_dim, output_dim,dropout=0., act=nn.ReLU(), **kwargs):
        super(GraphConvolution, self).__init__()
        w = nn.Parameter(torch.Tensor(input_dim,output_dim))
        self.weights = nn.init.xavier_uniform_(w,gain=nn.init.calculate_gain('relu'))
        self.dropout = dropout
        self.act = act
    def forward(self,adj,inputs):
        x = inputs
        x = torch.matmul(x, self.weights)
        x = torch.matmul(adj,x)
        outputs = self.act(x)
        return outputs