import os

from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv
from torch_geometric.nn import SAGEConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import itertools


class GNN(torch.nn.Module):
    def __init__(self, feats_dim, emb_dim, base_model=None, type="gcn", **kwargs):
        super(GNN, self).__init__()

        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        if type == "ppmi":
            model_cls = PPMIConv
        elif type == "gcn":
            model_cls = CachedGCNConv
        elif type == "sage":
            model_cls = SAGEConv
        # model_cls = PPMIConv if type == "ppmi" else CachedGCNConv
        if (type == "ppmi") or (type == "gcn"):
            self.conv_layers = nn.ModuleList([
                model_cls(feats_dim, 128,
                        weight=weights[0],
                        bias=biases[0],
                        **kwargs),
                model_cls(128, emb_dim,
                        weight=weights[1],
                        bias=biases[1],
                        **kwargs)
            ])
        else:
            self.conv_layers = nn.ModuleList([
                model_cls(feats_dim, 128,
                        **kwargs),
                model_cls(128, emb_dim,
                        **kwargs)
            ])

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            if (type == "ppmi") or (type == "gcn"):
                x = conv_layer(x, edge_index, cache_name)
                if i < len(self.conv_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout_layers[i](x)
            else:
                x = conv_layer(x, edge_index)
                if i < len(self.conv_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout_layers[i](x)
        return x


class Classifier_OSDA(nn.Module):
    # from OSDA-BP
    def __init__(self, encoder_dim, num_classes):
        super(Classifier_OSDA, self).__init__()
        self.fc = nn.Linear(encoder_dim, num_classes)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x.register_hook(grl_hook(coeff=1.0))
        x = self.fc(x)

        return x


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class Classifier(nn.Module):
    def __init__(self, encoder_dim, num_classes):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self._features_dim = encoder_dim
        self.head = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        predictions = self.head(x)
        return predictions


# for adversarial domain discriminator
class DomainDiscriminator(nn.Module):
    def __init__(self, encoder_dim):
        super(DomainDiscriminator, self).__init__()
        self.ad_layer1 = nn.Linear(encoder_dim, 32)
        self.ad_layer2 = nn.Linear(32, 1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x * 1.0
        x.register_hook(grl_hook(1.0))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        output = self.sigmoid(x)

        return output


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)


    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs






