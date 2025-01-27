from models import *
from torchsummary import summary

from models import convnet

reducedres18=resnet.WideResNet18(100,64).cuda()

summary(reducedres18,(3,32,32))

reducedres18=resnet.Reduced_ResNet18(100).cuda()

summary(reducedres18,(3,32,32))

conv3=convnet.ConvNet(100).to('cuda')
summary(conv3,(3,32,32))

import torch.onnx

# 假设你的模型是model，输入示例是input_example
# 创建一个输入张量，形状与模型的输入相匹配
input_example = torch.randn(1, 3, 32, 32).cuda()  # 根据模型的输入大小调整

# 导出模型为ONNX格式
onnx_path = "model.onnx"
torch.onnx.export(
    reducedres18,                # 要导出的模型
    input_example,        # 输入示例，用于推断模型的输入形状
    onnx_path,            # 保存路径
    export_params=True,   # 是否导出模型的参数
    opset_version=12,     # ONNX opset版本
    do_constant_folding=True,  # 是否进行常量折叠优化
    input_names=['input'],     # 输入的名称
    output_names=['output'],   # 输出的名称
    dynamic_axes={
        'input': {0: 'batch_size'},   # 动态批次大小
        'output': {0: 'batch_size'}
    }
)