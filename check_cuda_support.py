# 检查Python库的CUDA支持情况

# 检查PyTorch的CUDA支持
try:
    import torch
    print('PyTorch版本:', torch.__version__)
    print('CUDA可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA设备数:', torch.cuda.device_count())
        print('CUDA设备名称:', torch.cuda.get_device_name(0))
        print('CUDA版本:', torch.version.cuda)
    else:
        print('PyTorch未启用CUDA支持')
except ImportError:
    print('PyTorch未安装')

print('\n' + '-'*50 + '\n')

# 检查OpenCV的CUDA支持
try:
    import cv2
    print('OpenCV版本:', cv2.__version__)
    try:
        import cv2.cuda
        print('OpenCV CUDA模块可用')
        # 检查CUDA设备信息
        devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f'OpenCV CUDA设备数: {devices}')
    except (ImportError, AttributeError):
        print('OpenCV CUDA模块不可用')
except ImportError:
    print('OpenCV未安装')

print('\n' + '-'*50 + '\n')

# 检查Ultralytics/YOLOv8的CUDA支持
try:
    import ultralytics
    print('Ultralytics版本:', ultralytics.__version__)
    from ultralytics import YOLO
    
    # 尝试加载模型并检查设备
    model = YOLO('yolov8n-seg.pt')
    print('YOLOv8模型加载成功')
    
    # 检查YOLO使用的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'YOLO将使用的设备: {device}')
    
    # 简单测试推理
    try:
        # 创建一个随机图像进行测试
        import numpy as np
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(img, device=device)
        print('YOLO推理测试成功')
    except Exception as e:
        print(f'YOLO推理测试失败: {e}')
except ImportError:
    print('Ultralytics未安装')
except Exception as e:
    print(f'加载YOLO模型时出错: {e}')