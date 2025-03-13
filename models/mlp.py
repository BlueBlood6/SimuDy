import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim, use_bias=False):
        super().__init__()
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim_list[0])])
        for i in range(1, len(hidden_dim_list)):
            self.layers.append(nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i], bias=use_bias))
        self.layers.append(nn.Linear(hidden_dim_list[-1], output_dim, bias=False))  # output layer

    def forward(self, data):
        feats = Flatten()(data)
        for layer in self.layers[:-1]:
            feats = layer(feats)
            feats = self.activation(feats)
        feats = self.layers[-1](feats)
        return feats