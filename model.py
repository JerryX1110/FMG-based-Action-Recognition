import torch
import torch.nn as nn
from torch.nn import functional as F
from  utils import sparse_dropout



class LSTMnet(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(LSTMnet,self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out,_=self.lstm(x)
        out = out[:,-1,:]

        out=self.linear(out)
        out=self.sigmoid(out)

        return out


class ANNnet(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(ANNnet,self).__init__()
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        self.n_layer=n_layer
        self.n_class=n_class

        self.fc1=nn.Linear(in_features=in_dim,out_features=hidden_dim)
        self.relu=nn.ReLU()

        layers=[]
        for i in range(1,n_layer):
            # layers.append(nn.Dropout(p=0.5))
            layers.append(nn.Linear(in_features=hidden_dim,out_features=hidden_dim))
            layers.append(nn.ReLU())  
        self.layers=nn.Sequential(*layers)

        self.classifier=nn.Linear(in_features=hidden_dim,out_features=n_class)

    def forward(self,x):
        x = self.relu(self.fc1(x))#[B,T,F] -> [B,T,H_dim]
        x = self.layers(x) #[B,T,H_dim] -> [B,T,H_dim]
        x = torch.mean(x,dim=1) #[B,H_dim]
        x = self.classifier(x) #[B,H_dim] -> [B,C]
        out=torch.softmax(x,dim=1)
        return out
 


class CNNnet(nn.Module):
    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(CNNnet,self).__init__()

        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        self.n_layer=n_layer
        self.n_class=n_class

        self.conv1=nn.Conv1d(in_channels=self.in_dim,out_channels=self.hidden_dim,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        
        layers=[]
        for i in range(1,n_layer):
            layers.append(nn.Conv1d(in_channels=self.hidden_dim,out_channels=self.hidden_dim,kernel_size=3,stride=1,padding=1))
            layers.append(nn.ReLU())
        self.layers=nn.Sequential(*layers)

        self.classifier=nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.n_class, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        
        x = x.permute(0,2,1)#[B,T,F] -> [B,F,T]
        x = self.relu(self.conv1(x))#[B,F,T] -> [B,H_dim,T]
        x = self.layers(x)
        x = self.classifier(x)#[B,H_dim,T] -> [B,C,T]
        x = x.permute(0,2,1)#[B,C,T] -> [B,T,C]
        x = self.sigmoid(x)

        out = torch.mean(x, dim=1)

        return out

class GraphConvolution(nn.Module):

    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless


        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs
        if self.training and self.is_sparse_inputs:
            num_features_nonzero=x._nnz()
            x = sparse_dropout(x, self.dropout, num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

class GCNnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNnet, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim
        self.dropout=0

        print('input dim:', input_dim)
        print('output dim:', output_dim)


        self.layers = nn.Sequential(GraphConvolution(self.input_dim, hidden_dim,
                                                     activation=F.relu,
                                                     dropout=self.dropout,
                                                     is_sparse_inputs=True),

                                    GraphConvolution(hidden_dim, hidden_dim,
                                                     activation=F.relu,
                                                     dropout=self.dropout,
                                                     is_sparse_inputs=False),

                                    )
        self.classifier=nn.Linear(in_features=hidden_dim,out_features=output_dim)

    def forward(self, x, support):
        
        x = self.layers((x.to(torch.float32), support))#[C,T]
        # x=torch.sum(x[0],dim=0)
        x=torch.mean(x[0],dim=1)
        x=x.view(1,x.shape[0])
        x=self.classifier(x)
        x=torch.softmax(x,dim=1)

        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))
        loss = None
        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()
        return loss
