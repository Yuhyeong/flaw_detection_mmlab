# flaw_detection_mmlab

所有python暂时未写命令行读取，需要自己去文件内改路径。

每个文件内都是函数，没写类。

py文件有详细注释。

## 目录结构
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

## 教程

[视频：自定义数据集转中间格式讲解](https://www.bilibili.com/video/BV1bM4y1g7Hf?p=4&vd_source=f71295355febbf9584b3fc0781438a910)

[视频：从环境搭建、数据转换、训练的代码实操](https://www.bilibili.com/video/BV1bM4y1g7Hf?p=4&vd_source=f71295355febbf9584b3fc0781438a910)

[配套ipynb文件](https://github.com/open-mmlab/OpenMMLabCourse/blob/main/codes/lec4.ipynb)

## 环境搭建

```bash
conda create -n openmmlab python=3.8 pytorch==1.10.2 cudatoolkit=10.2 torchvision -c pytorch -y
conda activate openmmlab
pip install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

## 训练流程

1. 准备好数据与标签，所有图片放在train/data中、所有标签放在train/label中，需保证图片与标签名字一致（图片可无对应标签）

2. 进入src目录，运行dataset_split.py、label_preprocess.py得到中间数据格式的标签文件

   ```sh
   cd src
   python dataset_split.py		#按一定比例，移动一部分训练集到验证集中，train--->val
   python label_preprocess.py  #同时处理train、val目录，得到datasets/train.pkl和datasets/val.pkl
   ```

3. 同样在src目录下，运行create_custom_dataset.py

   ```shell
   python create_custom_dataset.py	#按照自定义配置生成网络配置文件
   ```

4. 训练数据

   ```shell
   python train_custom_dataset.py
   ```
   
## 测试

暂时先只对一整个文件夹内的所有图片进行处理

```bash
python test.py
# 或者
python test_nms.py
```

## utils目录内文件作用

**abspos2box.py**	输入绝对坐标和图片，在图片上画框

**check_version.py**	检查环境是否配置好

**read_pkl.py**	读取pkl文件（生成的中间标签）

**sample开头的文件：**  模板文件，用来参考的

**visualize开头的文件：**  与src中对应，但是加了可视化功能，可视化文件输出的路径要自己输入
