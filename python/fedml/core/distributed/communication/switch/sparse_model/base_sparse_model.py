import torch

class SparseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
def to_sparse_module(model: torch.nn.Module):
    """
    将普通 nn.Module 转换成稀疏模型
    
    目前仅支持 Conv2d 和 Linear
    """
    pass