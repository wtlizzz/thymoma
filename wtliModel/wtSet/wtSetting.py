"""
Configs for training & testing
Written by Wtlizzz
"""

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone_model",
        default="resnet",
        type=str,
        help="原始model，resnet||vit",
    )
    parser.add_argument(
        "--pos_loss_weight",
        default="5",
        type=int,
        help="Loss weight",
    )
    parser.add_argument(
        "--ct_data_root",
        default="./data",
        type=str,
        help="ct图像路径，包括image & label",
    )
    parser.add_argument(
        "--name_list",
        default="./data/train.txt",
        type=str,
        help="要训练的ct数据名称，例如：123.nii.gz",
    )
    parser.add_argument(
        "--val_list_path",
        default="../wtlizzz-core/wtliModel/wtData/data/label1/valList.txt",
        type=str,
        help="Gun Du Zi",
    )
    parser.add_argument(
        "--epoch", default=50, type=int, help="Number of total epochs to run"
    )
    parser.add_argument(
        "--test_save_path",
        default="./",
        type=str,
        help="这tm是测试用的，保存出结果目录",
    )
    parser.add_argument(
        "--label_flag", default="L", type=str, help="这tm的是label3个别的flag"
    )
    parser.add_argument("--input_D", default=56, type=int, help="Input size of depth")
    parser.add_argument("--input_H", default=448, type=int, help="Input size of height")
    parser.add_argument("--input_W", default=448, type=int, help="Input size of width")
    parser.add_argument(
        "--save_weight",
        default="./wtlizzz-core/wtliModel/wtWeight/default/",
        type=str,
        help="Input size of width",
    )
    parser.add_argument(
        "--summaryWriter_path",
        default="./wtlizzz-core/wtliModel/wtLogs/testMain",
        type=str,
        help="summaryWriter_path",
    )
    # parser.add_argument(
    #     "--pin_memory",
    #     default="./wtlizzz-core/wtliModel/wtWeight/default/",
    #     type=str,
    #     help="Input size of width",
    # )
    parser.add_argument(
        "--label_path",
        default="./wtlizzz-core/wtliModel/wtWeight/wtData/label/default.txt",
        type=str,
        help="Input size of width",
    )
    parser.add_argument(
        "--n_seg_classes", default=1, type=int, help="Number of segmentation classes"
    )
    parser.add_argument("--pin_memory", default=True, type=bool, help="")
    parser.add_argument(
        "--learning_rate",  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help="Initial learning rate (divided by 10 while training by lr scheduler)",
    )
    parser.add_argument("--num_workers", default=8, type=int, help="Number of jobs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch Size")
    parser.add_argument(
        "--phase", default="train", type=str, help="Phase of train or test"
    )
    parser.add_argument(
        "--save_intervals", default=10, type=int, help="Interation for saving model"
    )

    parser.add_argument(
        "--resume_path", default="", type=str, help="Path for resume model."
    )
    parser.add_argument(
        "--pretrain_path",
        default="pretrain/resnet_50.pth",
        type=str,
        help="Path for pretrained model.",
    )
    parser.add_argument(
        "--new_layer_names",
        # default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=["conv_seg"],
        type=list,
        help="New layer except for backbone",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        help="If true, cuda is not used.",
    )
    parser.add_argument("--gpu_id", default=0, type=int, help="Gpu id lists")
    parser.add_argument(
        "--model_depth",
        default=50,
        type=int,
        help="Depth of resnet (10 | 18 | 34 | 50 | 101)",
    )
    parser.add_argument(
        "--resnet_shortcut",
        default="B",
        type=str,
        help="Shortcut type of resnet (A | B)",
    )
    args = parser.parse_args()
    return args
