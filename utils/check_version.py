import torch
import torchvision
import mmdet
from mmcv.ops import get_compiling_cuda_version,get_compiler_version
# 检查pytorch和cuda是否可用
print(torch.__version__, torch.cuda.is_available())
# 检查mmdet是否可用
print(mmdet.__version__)
# 检查mmcv安装是否成功
print(get_compiling_cuda_version())
print(get_compiler_version())