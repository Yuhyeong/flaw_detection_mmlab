import os
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


def train_custom_config():
    cfg = Config.fromfile('../work_dir_custom/customformat.py')

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]

    # 构建检测模型
    model = build_detector(cfg.model)
    # 添加类别文字属性提高可视化效果
    model.CLASSES = datasets[0].CLASSES

    # 创建工作目录并训练模型
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)


def train_other_config():
    cfg = Config.fromfile('../work_dir_custom/other.py')

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]

    # 构建检测模型
    model = build_detector(cfg.model)
    # 添加类别文字属性提高可视化效果
    model.CLASSES = datasets[0].CLASSES

    # 创建工作目录并训练模型
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)


train_custom_config()
