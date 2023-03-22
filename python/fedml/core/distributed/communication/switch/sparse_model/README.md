# Prune Tool
## 用法
关键代码在 `prune.py` 中

对于以下的一个 Sequential 模型

``` py
conv1 = torch.nn.Conv2d(1, 32, 3)  # 输入通道为1（灰度图），输出通道为32，卷积核大小为3x3
conv2 = torch.nn.Conv2d(32, 64, 3)  # 输入通道为32，输出通道为64，卷积核大小为3x3
pool = torch.nn.MaxPool2d(2)  # 最大池化层，池化核大小为2x2
# 全连接层，输入特征维度为64*5*5（根据前面的卷积和池化计算得到），输出特征维度为128
fc1 = torch.nn.Linear(64 * 5 * 5, 128)
fc2 = torch.nn.Linear(128, 10)  # 全连接层，输入特征维度为128，输出特征维度为10（类别数）
softmax = torch.nn.Softmax(dim=1)  # softmax层，在第一个维度上进行归一化
```

对 conv1, conv2, fc1 进行剪枝

``` py
# 剪枝 conv1
tool = PruneTool(conv1, conv2, list(range(conv1_out))) # 对 conv1 的 32 个输出通道，分别对应 0-31 的 id
importance = tool.cal_importance # 得到一个长度为 32 的向量
# ...使用任意算法算出需要被剪去的通道 id
(conv1_prune1, conv2_prune1, patch1) = tool.prune(""" 需要被剪去的 id 列表 """, pic_size)
# conv1_prune1 和 conv2_prune1 是剪枝后的层，patch1 是产生的补丁
```

同理对 (conv2, fc1) 和 (fc1, fc2) 应用剪枝，得到 patch2 和 patch3。
此时把剪枝后的层放回原始模型中，可以直接开始训练。当然与此同时，也可以将剪枝后的层和 patch1 2 3 通过某种方法发送给其他节点。其他节点可以直接训练剪枝后的各种层，也可以选择利用补丁还原成原始模型。
还原的例子如下

``` py
# 假设有被剪枝后的层 
# conv1_1(1, 16, 3) 
# conv2_1(16, 32, 3) 
# fc1_1(32 * 5 * 5, 64)
# fc2_1(64, 10)
# patch1 + patch2 + patch3


# 首先利用patch3还原最后两层
tool = PruneTool(fc1, fc2, """ fc1 中各个输出通道的 id """)
(fc1_2, fc2_2) = tool.recovery(patch3) # 使用 patch3

# 然后同理依次利用patch2和patch1还原其他层，得到恢复后的模型

```
