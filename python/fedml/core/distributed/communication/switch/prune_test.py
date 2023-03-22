import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_model.prune_tool import *
from typing import List
from torchsummary import summary

class CNN_DropOut(torch.nn.Module):
    def __init__(self, only_digits=True):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x

def pruneCNN(model: CNN_DropOut, ratio):

    res: List[Patch] = []
    model.pic_size = 12*12
    model.conv1_channel_id = range(model.conv2d_1.out_channels)
    model.conv2_channel_id = range(model.conv2d_2.out_channels)
    model.fc1_channel_id = range(model.linear_1.out_features)

    tool = PruneTool(model.conv2d_1, model.conv2d_2, model.conv1_channel_id)
    prune_channel = tool.get_pruned_channel(ratio)
    (model.conv2d_1, model.conv2d_2, patch, model.conv1_channel_id) = tool.prune(prune_channel)
    res.append(patch)
    
    tool = PruneTool(model.conv2d_2, model.linear_1, model.conv2_channel_id)
    prune_channel = tool.get_pruned_channel(ratio)
    (model.conv2d_2, model.linear_1, patch, model.conv2_channel_id) = tool.prune(prune_channel, model.pic_size)
    res.append(patch)

    tool = PruneTool(model.linear_1, model.linear_2, model.fc1_channel_id)
    prune_channel = tool.get_pruned_channel(ratio)
    (model.linear_1, model.linear_2, patch, model.fc1_channel_id) = tool.prune(prune_channel)
    res.append(patch)

    return res

def recoverCNN(model: CNN_DropOut, patchList: List[Patch]):
    tool = PruneTool(model.linear_1, model.linear_2, range(model.linear_1.out_features))
    (model.linear_1, model.linear_2, model.fc1_channel_id) = tool.recovery(patchList[2])

    tool = PruneTool(model.conv2d_2, model.linear_1, range(model.conv2d_2.out_channels))
    (model.conv2d_2, model.linear_1, model.conv2_channel_id) = tool.recovery(patchList[1])

    tool = PruneTool(model.conv2d_1, model.conv2d_2, range(model.conv2d_1.out_channels))
    (model.conv2d_1, model.conv2d_2, model.conv1_channel_id) = tool.recovery(patchList[0])


myModel = CNN_DropOut(True)

patchList = pruneCNN(myModel, 0.2)

for name, parameter in myModel.named_parameters():
    print(f"{name}: {parameter.size()}")

(meta, tensor) = dump_patches(patchList)

print(tensor.size())

patchList = load_patches(meta, tensor)


recoverCNN(myModel, patchList)

for name, parameter in myModel.named_parameters():
    print(f"{name}: {parameter.size()}")
 