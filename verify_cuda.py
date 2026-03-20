# 验证PyTorch的CUDA支持情况
import torch

print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA设备数:', torch.cuda.device_count())
    print('CUDA设备名称:', torch.cuda.get_device_name(0))
    print('CUDA版本:', torch.version.cuda)
    
    # 尝试在GPU上执行简单计算
    x = torch.randn(1000, 1000)
    x_gpu = x.to('cuda')
    result = torch.matmul(x_gpu, x_gpu)
    print('GPU计算成功! 结果形状:', result.shape)
else:
    print('PyTorch未启用CUDA支持')
    print('原因分析:')
    print('1. 可能安装的是CPU版本的PyTorch')
    print('2. 可能系统中的NVIDIA驱动不兼容')
    print('3. 对于Python 3.13，PyTorch官方可能尚未发布支持CUDA的正式版本')