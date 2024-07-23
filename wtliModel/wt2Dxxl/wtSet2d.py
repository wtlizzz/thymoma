import argparse


def parse_opts_2d():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"/home/zarchive/wangwenmiao/20240708xxl/png",
        help="The data path",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=r"/home/zarchive/wangwenmiao/20240708xxl/LABEL.csv",
        help="The label path",
    )
    parser.add_argument(
        "--label_flag", default="L", type=str, help="这tm的是label3个别的flag"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
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
    # resnet50预训练权重
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    parser.add_argument(
        "--pretrain_path",
        type=str,
        default=r"../model/pretrain/resnet_50.pth",
        help="initial weights path",
    )  #
    parser.add_argument("--freeze-layers", type=bool, default=False)
    parser.add_argument("--use_cuda", default=True)
    parser.add_argument("--optimizer", type=str, default="SGD")  # Adam
    parser.add_argument("--gpu_id", default=0, type=int, help="Gpu id lists")

    args = parser.parse_args()
    return args
