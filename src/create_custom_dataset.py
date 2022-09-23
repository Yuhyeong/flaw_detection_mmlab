import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed

# 获取基本配置文件参数
cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# 修改数据集类型以及文件路径
cfg.dataset_type = 'CustomDataset'
cfg.data_root = '../datasets/'
cfg.classes = ('1', '2', '3')

cfg.data.test.type = 'CustomDataset'
cfg.data.test.data_root = '../datasets/'
cfg.data.test.ann_file = '../datasets/test.pkl'
cfg.data.test.img_prefix = 'test/data'
cfg.data.test.classes = ('1', '2', '3')

cfg.data.train.type = 'CustomDataset'
cfg.data.train.data_root = '../datasets/'
cfg.data.train.ann_file = '../datasets/train.pkl'
cfg.data.train.img_prefix = 'train/data'
cfg.data.train.classes = ('1', '2', '3')

cfg.data.val.type = 'CustomDataset'
cfg.data.val.data_root = '../datasets/'
cfg.data.val.ann_file = '../datasets/val.pkl'
cfg.data.val.img_prefix = 'val/data'
cfg.data.val.classes = ('1', '2', '3')

# 修改bbox_head中的类别数
cfg.model.roi_head.bbox_head.num_classes = 3
# 设置工作目录用于存放log和临时文件
cfg.work_dir = '../work_dir_custom'

# 原本的学习率是在8卡基础上训练设置的，现在双卡需要除以4，单卡则8
cfg.optimizer.lr = 0.000000000015
cfg.lr_config.warmup = None
cfg.log_config.interval = 100

# 由于是自定义数据集，需要修改评价方法
cfg.evaluation.metric = 'mAP'
# 设置evaluation间隔减少运行时间
cfg.evaluation.interval = 1
# 设置存档点间隔减少存储空间的消耗
cfg.checkpoint_config.interval = 1

# 固定随机种子使得结果可复现
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.data.workers_per_gpu = 8

# 打印所有的配置参数
print(f'Config:\n{cfg.pretty_text}')

mmcv.mkdir_or_exist(F'{cfg.work_dir}')
cfg.dump(F'{cfg.work_dir}/customformat.py')
