import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed

# 获取基本配置文件参数
cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# 修改数据集类型以及文件路径
cfg.dataset_type = 'CustomDataset'
cfg.data_root = 'data/'
cfg.classes = ('1', '2', '3')

cfg.data.test.type = 'CustomDataset'
cfg.data.test.data_root = 'data/'
cfg.data.test.ann_file = 'train_middle.pkl'
cfg.data.test.img_prefix = 'test/image_2'
cfg.data.test.classes = ('1', '2', '3')

cfg.data.train.type = 'CustomDataset'
cfg.data.train.data_root = 'data/'
cfg.data.train.ann_file = 'train_middle.pkl'
cfg.data.train.img_prefix = 'train/image_2'
cfg.data.train.classes = ('1', '2', '3')

cfg.data.val.type = 'CustomDataset'
cfg.data.val.data_root = 'data/'
cfg.data.val.ann_file = 'val_middle.pkl'
cfg.data.val.img_prefix = 'val/image_2'
cfg.data.val.classes = ('1', '2', '3')

# 修改bbox_head中的类别数
cfg.model.roi_head.bbox_head.num_classes = 3
# 使用预训练好的faster rcnn模型用于finetuning
cfg.load_from = '../checkpoints/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth'
# 设置工作目录用于存放log和临时文件
cfg.work_dir = './work_dir_custom'

# 原本的学习率是在8卡基础上训练设置的，现在单卡需要除以8
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# 由于是自定义数据集，需要修改评价方法
cfg.evaluation.metric = 'mAP'
# 设置evaluation间隔减少运行时间
cfg.evaluation.interval = 12
# 设置存档点间隔减少存储空间的消耗
cfg.checkpoint_config.interval = 12

# 固定随机种子使得结果可复现
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# 打印所有的配置参数
print(f'Config:\n{cfg.pretty_text}')

mmcv.mkdir_or_exist(F'{cfg.work_dir}')
cfg.dump(F'{cfg.work_dir}/customformat.py')