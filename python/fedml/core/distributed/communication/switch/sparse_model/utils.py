import torch
from torch.utils.data import DataLoader

def test_model(model: torch.nn.Module, data_loader: DataLoader):
    correct = 0  # 初始化正确预测数为0
    total = 0  # 初始化总样本数为0
    with torch.no_grad():  # 不需要计算梯度
        for data in data_loader:  # 遍历每一个批次
            images, labels = data  # 获取输入和标签
            outputs = model(images)  # 计算模型输出
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的类别作为预测结果
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测数
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total
    ))

criterion = torch.nn.CrossEntropyLoss()  # 定义交叉熵损失函数对象

def train_model(model: torch.nn.Module, data_loader: DataLoader, total_worker_num:int, curr_worker_id: int, epochs = 1):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):  # 遍历每一轮
        running_loss = 0.0  # 初始化累计损失为0
        for i, data in enumerate(data_loader):  # 遍历每一个批次
            if i % 200 == 0 and i != 0:  # 每200个批次打印一次平均损失值
                print('[node %d, %d, %5d] loss: %.3f' %
                      (curr_worker_id, epoch + 1, i, running_loss / 200))
                running_loss = 0.0
            if i % total_worker_num != curr_worker_id:
                continue # 每个节点训练不同的数据
            inputs, labels = data  # 获取输入和标签
            optimizer.zero_grad()  # 清零梯度缓存
            outputs = model(inputs)  # 计算模型输出
            loss = criterion(outputs, labels)  # 计算损失函数值
            loss.backward()  # 反向传播梯度
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累加损失值

def create_state_dict(module: torch.nn.Module, data: torch.tensor):
    weight_len = module.weight.numel()
    bias_len = module.bias.numel()
    return (
        {
            "weight": data[0:weight_len].reshape(module.weight.shape),
            "bias": data[weight_len:weight_len + bias_len].reshape(module.bias.shape)
        },
        weight_len + bias_len
    )


def apply_data(module: torch.nn.Module, data: torch.tensor):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        (state_dict, step) = create_state_dict(module, data)
        module.load_state_dict(state_dict)
        return step
    return 0