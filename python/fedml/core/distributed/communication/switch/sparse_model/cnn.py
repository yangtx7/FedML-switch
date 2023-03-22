import torch
from typing import List
from prune_tool import Patch, PruneTool
from sparse_model.utils import create_state_dict, apply_data

class SparseCNN(torch.nn.Module):
    def __init__(self, meta: dict, data: torch.tensor = None):
        super(SparseCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(*meta["conv1"])
        self.conv1_channel_id = meta["conv1_channel_id"]

        self.conv2 = torch.nn.Conv2d(*meta["conv2"])
        self.conv2_channel_id = meta["conv2_channel_id"]

        self.pool = torch.nn.MaxPool2d(2)  # 最大池化层，池化核大小为2x2

        # 全连接层，输入特征维度为64*5*5（根据前面的卷积和池化计算得到），输出特征维度为128
        self.pic_size = meta["pic_size"]
        self.fc1 = torch.nn.Linear(*meta["fc1"])
        self.fc1_channel_id = meta["fc1_channel_id"]
        
        self.fc2 = torch.nn.Linear(*meta["fc2"])
        self.softmax = torch.nn.Softmax(dim=1)  # softmax层，在第一个维度上进行归一化

        cursor = 0
        if not data is None:
            cursor += apply_data(self.conv1, data[cursor:])
            cursor += apply_data(self.conv2, data[cursor:])
            cursor += apply_data(self.fc1, data[cursor:])
            cursor += apply_data(self.fc2, data[cursor:])
            if cursor != data.numel():
                print("WARNING: data 未消费完毕")

    def dump(self):
        # 参数顺序等于构造函数的参数顺序
        meta = {
            "conv1": (self.conv1.in_channels, self.conv1.out_channels, self.conv1.kernel_size),
            "conv1_channel_id": self.conv1_channel_id,
            "conv2": (self.conv2.in_channels, self.conv2.out_channels, self.conv2.kernel_size),
            "conv2_channel_id": self.conv2_channel_id,
            "fc1": (self.fc1.in_features, self.fc1.out_features),
            "fc1_channel_id": self.fc1_channel_id,
            "fc2": (self.fc2.in_features, self.fc2.out_features),
            "pic_size": self.pic_size
        }
        data = torch.cat([
            self.conv1.weight.flatten(), self.conv1.bias.flatten(),
            self.conv2.weight.flatten(), self.conv2.bias.flatten(),
            self.fc1.weight.flatten(), self.fc1.bias.flatten(),
            self.fc2.weight.flatten(), self.fc2.bias.flatten(),
        ])

        return (meta, data)

    def apply_patches(self, patches: List[Patch]) -> None:
        (self.fc1, self.fc2, self.fc1_channel_id) = PruneTool(
            self.fc1, self.fc2, self.fc1_channel_id).recovery(patches[2])
        (self.conv2, self.fc1, self.conv2_channel_id) = PruneTool(
            self.conv2, self.fc1, self.conv2_channel_id).recovery(patches[1])
        (self.conv1, self.conv2, self.conv1_channel_id) = PruneTool(
            self.conv1, self.conv2, self.conv1_channel_id).recovery(patches[0])

    def prune(self, ratio: float) -> List[Patch]:
        """
        按照剪枝率 ratio 剪去不重要通道, 暂时实现为剪去0-ratio的通道

        返回 Patches
        """
        res: List[Patch] = []

        tool = PruneTool(self.conv1, self.conv2, self.conv1_channel_id)
        prune_channel = tool.get_pruned_channel(ratio)
        (self.conv1, self.conv2, patch, self.conv1_channel_id) = tool.prune(prune_channel)
        res.append(patch)

        tool = PruneTool(self.conv2, self.fc1, self.conv2_channel_id)
        prune_channel = tool.get_pruned_channel(ratio)
        (self.conv2, self.fc1, patch, self.conv2_channel_id) = tool.prune(prune_channel, self.pic_size)
        res.append(patch)

        tool = PruneTool(self.fc1, self.fc2, self.fc1_channel_id)
        prune_channel = tool.get_pruned_channel(ratio)
        (self.fc1, self.fc2, patch, self.fc1_channel_id) = tool.prune(prune_channel)
        res.append(patch)
        return res


    def redo_prune(self, example_patches: List[Patch]) -> List[Patch]:
        """
        根据 example_patches 的剪枝方式对当前模型进行剪枝
        """
        res: List[Patch] = []
        (self.conv1, self.conv2, patch, self.conv1_channel_id) = PruneTool(
            self.conv1, self.conv2, self.conv1_channel_id).prune(example_patches[0].prune_channel_id)
        res.append(patch)
        (self.conv2, self.fc1, patch, self.conv2_channel_id) = PruneTool(
            self.conv2, self.fc1, self.conv2_channel_id).prune(example_patches[1].prune_channel_id, self.pic_size)
        res.append(patch)
        (self.fc1, self.fc2, patch, self.fc1_channel_id) = PruneTool(
            self.fc1, self.fc2, self.fc1_channel_id).prune(example_patches[2].prune_channel_id)
        res.append(patch)
        return res

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))  # 第一个卷积-激活-池化操作
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))  # 第二个卷积-激活-池化操作
        x = x.view(-1, self.conv2.out_channels * self.pic_size)  # 将二维特征图展平成一维向量
        x = torch.nn.functional.relu(self.fc1(x))  # 第一个全连接-激活操作
        x = self.fc2(x)  # 第二个全连接操作
        x = self.softmax(x)  # softmax操作
        return x
