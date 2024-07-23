
### 流程
本项目根据下图的流程，针对胸腺瘤的CT图像作为数据源，构建resnet实现分类

思路流程：
1、读取图像文件夹，获取数据名称列表，写入到文件。主要用于读取指定文件
2、

pip install nibabel pandas tensorboard scikit-learn tqdm

项目结构：

- wtlizzzResnet:主文件

1、参数设置：

wt_init_setOpts:
```angular2html
    sets.data_root = 'G:\\data-CT\\alldata'
    sets.img_list = './wtData/test.txt'
    sets.n_epochs = 1
    sets.no_cuda = True
    sets.pretrain_path = ''
    sets.num_workers = 0
    sets.model_depth = 10
    sets.resnet_shortcut = 'A'
    sets.input_D = 14
    sets.input_H = 28
    sets.input_W = 28
    sets.pin_memory = True
```
wt_data_test:
检查输入数据的相关参数

2、数据准备：
```angular2html
wt_data_load_1 根据项目文件目录，生成训练集、测试集索引文件
wt_data_load_2 生成data_loader(需要确认输入数据的参数)
```

**3、数据处理过程:**
```angular2html
 读取img & label： nibabel.load(img_name) 
 转： data = data.get_data()
 删掉无效的区域： data, label = self.__drop_invalid_range__(data, label)
 裁剪数据： data, label = self.__crop_data__(data, label)
 重置数据大小： data = self.__resize_data__(data)
 归一化数据： data = self.__itensity_normalize_one_volume__(data)
```

**其他**

查看log， 运行命令
```
tensorboard --logdir=wtLogs --port=6653
```

模型微调：
handle-data:数据图像预处理，将图像做裁剪、resize
handle-data0712-1:数据图像扩大，将边界扩大10
handle-data0712-test:测试集图像，已将边界扩大10