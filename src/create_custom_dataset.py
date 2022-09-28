import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed


def create_custom_dataset():
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

    # 本模型在学习率1.5e-10上表现较好
    cfg.optimizer.lr = 0.0000000015
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


def create_faster_dataset():
    # 获取基本配置文件参数
    cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

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
    cfg.work_dir = '../work_dir_faster'

    # 本模型在学习率1.5e-10上表现较好
    cfg.optimizer.lr = 0.0000000015
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
    cfg.dump(F'{cfg.work_dir}/faster.py')


def create_cascade_dataset():
    # 获取基本配置文件参数
    cfg = Config.fromfile('../configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py')

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
    cfg.model.roi_head.bbox_head[0].num_classes = 3
    cfg.model.roi_head.bbox_head[1].num_classes = 3
    cfg.model.roi_head.bbox_head[2].num_classes = 3
    # 设置工作目录用于存放log和临时文件
    cfg.work_dir = '../work_dir_cascade'

    # 本模型在学习率1.5e-10上表现较好
    cfg.optimizer.lr = 0.0000000015
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 30

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
    cfg.dump(F'{cfg.work_dir}/cascade.py')


def create_cascade_r101_dataset():
    # 获取基本配置文件参数
    cfg = Config.fromfile('../configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py')

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
    cfg.model.roi_head.bbox_head[0].num_classes = 3
    cfg.model.roi_head.bbox_head[1].num_classes = 3
    cfg.model.roi_head.bbox_head[2].num_classes = 3
    # 设置工作目录用于存放log和临时文件
    cfg.work_dir = '../work_dir_cascade_r101'

    # 本模型在学习率1.5e-10上表现较好
    cfg.optimizer.lr = 0.0000000015
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 30

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
    cfg.dump(F'{cfg.work_dir}/cascade_r101.py')


def create_cascade_s101_dataset():
    # 获取基本配置文件参数
    cfg = Config.fromfile('../configs/resnest/cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py')

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
    cfg.model.roi_head.bbox_head[0].num_classes = 3
    cfg.model.roi_head.bbox_head[1].num_classes = 3
    cfg.model.roi_head.bbox_head[2].num_classes = 3
    cfg.model.roi_head.bbox_head[0].norm_cfg.type = 'BN'
    cfg.model.roi_head.bbox_head[1].norm_cfg.type = 'BN'
    cfg.model.roi_head.bbox_head[2].norm_cfg.type = 'BN'

    # 设置工作目录用于存放log和临时文件
    cfg.work_dir = '../work_dir_cascade_s101'

    # 本模型在学习率1.5e-10上表现较好
    cfg.optimizer.lr = 0.0000000015
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 30

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
    cfg.norm_cfg.type = 'BN'
    cfg.fp16 = dict(loss_scale=512.)

    # 打印所有的配置参数
    print(f'Config:\n{cfg.pretty_text}')

    mmcv.mkdir_or_exist(F'{cfg.work_dir}')
    cfg.dump(F'{cfg.work_dir}/cascade_s101.py')


# create_cascade_r101_dataset()

create_cascade_s101_dataset()
