import nibabel as nib
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from wtData.wtData import (
    data_preprocessing,
    wt_create_data_index,
    wt_data_test,
    wt_get_data_label,
    wt_rename_file,
)
from wtData.wtDataset import WtThymusDataset
from wtModel.wtModel import generate_model
from wtModel.wtTrain import wtTrain
from wtSet.wtSetting import parse_opts
from sklearn.model_selection import train_test_split
import argparse
import setproctitle
from wtData.wtDataPre import data_preprocessing2

setproctitle.setproctitle("This is wangwenmiao de nature ^.^!")


def wt_init_setOpts():
    sets = parse_opts()
    print(os.getcwd())
    sets.ct_data_root = "/home/zarchive/wangwenmiao/20240708xxl"
    sets.name_list = "wtlizzz-core/wtliModel/wtData/data/label1/train.csv"
    sets.val_list_path = "wtlizzz-core/wtliModel/wtData/data/label1/val.csv"
    sets.save_weight = "wtlizzz-core/wtliModel/wtWeight/label1/10/"
    sets.label_path = "wtlizzz-core/wtliModel/wtData/label/label1.csv"
    sets.pretrain_path = "./wtlizzz-core/model/pretrain/resnet_10.pth"
    sets.label_flag = "L"
    sets.backbone_model = "resnet"
    sets.epoch = 50
    sets.input_W = 112
    sets.input_H = 112
    sets.input_D = 112
    sets.phase = "train"
    sets.save_intervals = 10
    sets.no_cuda = False
    sets.model_depth = 10
    sets.batch_size = 1
    sets.gpu_id = 0
    return sets


# 创建txt图像所引文件，只需要运行一次
def wt_data_loader_1():
    """
    流程：获取资源目录下所有文件名称，划分为训练集验证集
    """
    image_path = "/home/zarchive/wangwenmiao/NNUNET_RESULT/image"
    label_path = "/home/zarchive/wangwenmiao/NNUNET_RESULT/label"
    out_path = "./wtData/sourceFileName"
    wt_create_data_index(image_path, label_path, out_path)


def wt_data_loader_2(sets):
    mask_path = sets.label_path
    print(f"wt_data_loader_2 mask_path:{mask_path}")
    print(f"dataset_training mask_path:{sets.name_list}")
    dataset_training = WtThymusDataset(sets)
    data_loader_train = DataLoader(
        dataset_training,
        batch_size=sets.batch_size,
        shuffle=True,
        num_workers=sets.num_workers,
        pin_memory=sets.pin_memory,
    )
    dataset_val = WtThymusDataset(sets, False)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=sets.batch_size,
        shuffle=True,
        num_workers=sets.num_workers,
        pin_memory=sets.pin_memory,
    )

    return data_loader_train, data_loader_val


def wt_init_train(sets, data_loader_train, data_loader_val):
    model = generate_model(sets)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    if torch.cuda.is_available():
        device = torch.device("cuda", sets.gpu_id)
    else:
        device = torch.device("cpu")

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([sets.pos_loss_weight])
    ).to(
        device
    )  # 分类不均衡
    print(f"torch.nn.BCEWithLogitsLoss: {sets.pos_loss_weight}")
    # criterion = torch.nn.BCEWithLogitsLoss().to(device)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    summaryWriter = SummaryWriter(sets.summaryWriter_path)
    # training
    wtTrain(
        data_loader_train,
        data_loader_val,
        summaryWriter,
        model,
        optimizer,
        scheduler,
        criterion,
        total_epochs=sets.epoch,
        save_interval=sets.save_intervals,
        save_folder=sets.save_weight,
        sets=sets,
    )


def wt_ttt():
    init_set = wt_init_setOpts()
    x = data_preprocessing2()
    x = torch.tensor(x)
    # x = np.load("/home/zarchive/wangwenmiao/project-thymoma/filename.npy")
    x = x.float()
    model = generate_model(init_set)

    # as_tensor3 = torch.as_tensor(x)
    # from_numpy4 = torch.from_numpy(x)
    tensor1 = torch.tensor(x)
    # tensor2 = torch.Tensor(x)

    logits = model(x)
    print(f"logits:{logits}")


def wt_main():
    init_set = parse_opts()
    # init_set = wt_init_setOpts()
    print(f"init_set:{init_set}")
    print(f"----backbone_model:{init_set.backbone_model}----")
    print(f"----model_depth:{init_set.model_depth}----")
    print(f"----batch_size:{init_set.batch_size}----")
    print(f"----pos_loss_weight:{init_set.pos_loss_weight}----")
    print(f"----pretrain_path:{init_set.pretrain_path}----")
    data_loader_train, data_loader_val = wt_data_loader_2(init_set)
    wt_init_train(init_set, data_loader_train, data_loader_val)


def main():
    init_set = parse_opts()
    print(f"init_set:{init_set}")


if __name__ == "__main__":
    wt_main()
    # data_preprocessing()
