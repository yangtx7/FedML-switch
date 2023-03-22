import torch.nn as nn
import torch
from sparse_model.utils import create_state_dict, apply_data
from prune_tool import Patch, PruneTool
from typing import List


class SparseVGG(nn.Module):
    def __init__(self, meta: dict, data: torch.tensor = None) -> None:
        super(SparseVGG, self).__init__()
        self.conv1 = nn.Conv2d(*meta["conv1"])
        self.conv1_channel_id = meta["conv1_channel_id"]
        self.conv2 = nn.Conv2d(*meta["conv2"])
        self.conv2_channel_id = meta["conv2_channel_id"]
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(*meta["conv3"])
        self.conv3_channel_id = meta["conv3_channel_id"]
        self.conv4 = nn.Conv2d(*meta["conv4"])
        self.conv4_channel_id = meta["conv4_channel_id"]
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(*meta["conv5"])
        self.conv5_channel_id = meta["conv5_channel_id"]
        self.conv6 = nn.Conv2d(*meta["conv6"])
        self.conv6_channel_id = meta["conv6_channel_id"]
        self.conv7 = nn.Conv2d(*meta["conv7"])
        self.conv7_channel_id = meta["conv7_channel_id"]
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(*meta["conv8"])
        self.conv8_channel_id = meta["conv8_channel_id"]
        self.conv9 = nn.Conv2d(*meta["conv9"])
        self.conv9_channel_id = meta["conv9_channel_id"]
        self.conv10 = nn.Conv2d(*meta["conv10"])
        self.conv10_channel_id = meta["conv10_channel_id"]
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(*meta["conv11"])
        self.conv11_channel_id = meta["conv11_channel_id"]
        self.conv12 = nn.Conv2d(*meta["conv12"])
        self.conv12_channel_id = meta["conv12_channel_id"]
        self.conv13 = nn.Conv2d(*meta["conv13"])
        self.conv13_channel_id = meta["conv13_channel_id"]
        self.pool5 = nn.MaxPool2d(2, 2)

        self.pic_size = meta["pic_size"]
        self.fc14 = nn.Linear(*meta["fc14"])
        self.fc14_channel_id = meta["fc14_channel_id"]
        self.fc15 = nn.Linear(*meta["fc15"])
        self.fc15_channel_id = meta["fc15_channel_id"]
        self.fc16 = nn.Linear(*meta["fc16"])
        self.softmax = nn.Softmax(dim=-1)
        cursor = 0
        if not data is None:
            cursor += apply_data(self.conv1, data[cursor:])
            cursor += apply_data(self.conv2, data[cursor:])
            cursor += apply_data(self.conv3, data[cursor:])
            cursor += apply_data(self.conv4, data[cursor:])
            cursor += apply_data(self.conv5, data[cursor:])
            cursor += apply_data(self.conv6, data[cursor:])
            cursor += apply_data(self.conv7, data[cursor:])
            cursor += apply_data(self.conv8, data[cursor:])
            cursor += apply_data(self.conv9, data[cursor:])
            cursor += apply_data(self.conv10, data[cursor:])
            cursor += apply_data(self.conv11, data[cursor:])
            cursor += apply_data(self.conv12, data[cursor:])
            cursor += apply_data(self.conv13, data[cursor:])
            cursor += apply_data(self.fc14, data[cursor:])
            cursor += apply_data(self.fc15, data[cursor:])
            cursor += apply_data(self.fc16, data[cursor:])
            if cursor != data.numel():
                print("WARNING: data 未消费完毕")

    def dump(self):
        meta = {"pic_size": self.pic_size}
        data = []

        for i in range(1, 14):
            key = "conv%d" % i
            conv: nn.Conv2d = getattr(self, key)
            meta[key] = (conv.in_channels, conv.out_channels,
                         conv.kernel_size, conv.stride, conv.padding)
            meta["%s_channel_id" % key] = getattr(self, "%s_channel_id" % key)
            data += [conv.weight.flatten(), conv.bias.flatten()]

        for i in range(14, 17):
            key = "fc%d" % i
            fc: nn.Linear = getattr(self, key)
            meta[key] = (fc.in_features, fc.out_features)
            if i < 16: # 最后一层不包含 channel id
                meta["%s_channel_id" % key] = getattr(self, "%s_channel_id" % key)
            data += [fc.weight.flatten(), fc.bias.flatten()]

        data = torch.cat(data)

        return (meta, data)

    def __index_to_tool(self, i: int):
        if i < 12:
            curr_layer = "conv%d" % (i + 1)
            next_layer = "conv%d" % (i + 2)
        if i == 12:
            curr_layer = "conv%d" % (i + 1)
            next_layer = "fc%d" % (i + 2)
        if i > 12:
            curr_layer = "fc%d" % (i + 1)
            next_layer = "fc%d" % (i + 2)
        tool = PruneTool(getattr(self, curr_layer), getattr(
            self, next_layer), getattr(self, curr_layer + "_channel_id"))
        return (tool, curr_layer, next_layer)

    def apply_patches(self, patches: List[Patch]):
        layer_index = 14
        for patch in reversed(patches):
            tool, curr_layer, next_layer = self.__index_to_tool(layer_index)
            new_curr, new_next, new_channel_id = tool.recovery(patch)
            setattr(self, curr_layer, new_curr)
            setattr(self, next_layer, new_next)
            setattr(self, curr_layer + "_channel_id", new_channel_id)
            layer_index -= 1

    def prune(self, ratio: float) -> List[Patch]:
        res: List[Patch] = []
        for i in range(15):
            tool, curr_layer, next_layer = self.__index_to_tool(i)
            scale = self.pic_size if next_layer == "fc14" else -1
            prune_channel = tool.get_pruned_channel(ratio)
            if len(prune_channel) == 0:
                patch = Patch([], torch.tensor([]), torch.tensor([]), torch.tensor([]), scale)
            else:
                (new_curr, new_next, patch, new_channel_id) = tool.prune(prune_channel, scale)
                setattr(self, curr_layer, new_curr)
                setattr(self, next_layer, new_next)
                setattr(self, curr_layer + "_channel_id", new_channel_id)
            res.append(patch)

        return res

    def redo_prune(self, example_patches: List[Patch]) -> List[Patch]:
        res: List[Patch] = []
        for i, example_patch in enumerate(example_patches):
            tool, curr_layer, next_layer = self.__index_to_tool(i)
            (new_curr, new_next, patch, new_channel_id) = tool.prune(
                example_patch.prune_channel_id, self.pic_size)
            res.append(patch)
            setattr(self, curr_layer, new_curr)
            setattr(self, next_layer, new_next)
            setattr(self, curr_layer + "_channel_id", new_channel_id)
        return res

    def forward(self, x):
        x = self.pool1(torch.nn.functional.relu(self.conv2(self.conv1(x))))
        x = self.pool2(torch.nn.functional.relu(self.conv4(self.conv3(x))))
        x = self.pool3(torch.nn.functional.relu(
            self.conv7(self.conv6(self.conv5(x)))))
        x = self.pool4(torch.nn.functional.relu(
            self.conv10(self.conv9(self.conv8(x)))))
        x = self.pool5(torch.nn.functional.relu(
            self.conv13(self.conv12(self.conv11(x)))))
        x = x.view(-1, self.conv13.out_channels * self.pic_size)
        x = torch.nn.functional.relu(self.fc14(x))
        x = torch.nn.functional.relu(self.fc15(x))
        x = self.fc16(x)
        x = self.softmax(x)  # softmax操作
        return x


pic_size = 7 * 7
# example_vgg_meta = {
#     "conv1": (1, 64, 3, 1, 1),
#     "conv1_channel_id": list(range(64)),
#     "conv2": (64, 64, 3, 1, 1),
#     "conv2_channel_id": list(range(64)),

#     "conv3": (64, 128, 3, 1, 1),
#     "conv3_channel_id": list(range(128)),
#     "conv4": (128, 128, 3, 1, 1),
#     "conv4_channel_id": list(range(128)),

#     "conv5": (128, 256, 3, 1, 1),
#     "conv5_channel_id": list(range(256)),
#     "conv6": (256, 256, 3, 1, 1),
#     "conv6_channel_id": list(range(256)),
#     "conv7": (256, 256, 3, 1, 1),
#     "conv7_channel_id": list(range(256)),

#     "conv8": (256, 512, 3, 1, 1),
#     "conv8_channel_id": list(range(512)),
#     "conv9": (512, 512, 3, 1, 1),
#     "conv9_channel_id": list(range(512)),
#     "conv10": (512, 512, 3, 1, 1),
#     "conv10_channel_id": list(range(512)),

#     "conv11": (512, 512, 3, 1, 1),
#     "conv11_channel_id": list(range(512)),
#     "conv12": (512, 512, 3, 1, 1),
#     "conv12_channel_id": list(range(512)),
#     "conv13": (512, 512, 3, 1, 1),
#     "conv13_channel_id": list(range(512)),

#     "fc14": (pic_size * 512, 4096),
#     "fc14_channel_id": list(range(4096)),
#     "fc15": (4096, 4096),
#     "fc15_channel_id": list(range(4096)),
#     "fc16": (4096, 10),

#     "pic_size": pic_size
# }


example_vgg_meta = {
    "conv1": (1, 32, 3, 1, 1),
    "conv1_channel_id": list(range(32)),
    "conv2": (32, 32, 3, 1, 1),
    "conv2_channel_id": list(range(32)),

    "conv3": (32, 64, 3, 1, 1),
    "conv3_channel_id": list(range(64)),
    "conv4": (64, 64, 3, 1, 1),
    "conv4_channel_id": list(range(64)),

    "conv5": (64, 128, 3, 1, 1),
    "conv5_channel_id": list(range(128)),
    "conv6": (128, 128, 3, 1, 1),
    "conv6_channel_id": list(range(128)),
    "conv7": (128, 128, 3, 1, 1),
    "conv7_channel_id": list(range(128)),

    "conv8": (128, 256, 3, 1, 1),
    "conv8_channel_id": list(range(256)),
    "conv9": (256, 256, 3, 1, 1),
    "conv9_channel_id": list(range(256)),
    "conv10": (256, 256, 3, 1, 1),
    "conv10_channel_id": list(range(256)),

    "conv11": (256, 256, 3, 1, 1),
    "conv11_channel_id": list(range(256)),
    "conv12": (256, 256, 3, 1, 1),
    "conv12_channel_id": list(range(256)),
    "conv13": (256, 256, 3, 1, 1),
    "conv13_channel_id": list(range(256)),

    "fc14": (pic_size * 256, 1024),
    "fc14_channel_id": list(range(1024)),
    "fc15": (1024, 1024),
    "fc15_channel_id": list(range(1024)),
    "fc16": (1024, 10),

    "pic_size": pic_size
}
if __name__ == "__main__":
    vgg = SparseVGG(example_vgg_meta)
    print(vgg)
    patches = vgg.prune(0.5)
    print(vgg)
    vgg.apply_patches(patches)
    print(vgg)
