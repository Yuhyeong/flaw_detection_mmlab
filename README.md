# flaw_detection_mmlab

### 目录结构
```
├─checkpoints           预训练权重存储目录
├─configs               mmdetection的默认网络配置目录，迁移过来了
├─datasets              自定义数据集存储目录
│  ├─test
│  │  ├─data
│  │  └─label
│  ├─train
│  │  ├─data
│  │  └─label
│  └─val
│      ├─data
│      └─label
├─result                各种结果存储目录
├─src                   项目核心代码目录
├─test_data             使用小工具测试时所用数据存储目录
│  └─temp               readme中图片存储目录
├─utils                 实用工具，例如读取pkl,可视化操作
└─work_dir_custom       生成的网络配置、训练得到的权重、训练过程日志文件存储目录
```
### 教程

[视频：自定义数据集转中间格式讲解](https://www.bilibili.com/video/BV1bM4y1g7Hf?p=4&vd_source=f71295355febbf9584b3fc0781438a910)

[视频：从环境搭建、数据转换、训练的代码实操](https://www.bilibili.com/video/BV1bM4y1g7Hf?p=4&vd_source=f71295355febbf9584b3fc0781438a910)

[配套ipynb文件](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/lec4.ipynb)
