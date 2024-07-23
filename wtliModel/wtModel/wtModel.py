import torch
from torch import nn
import sys

# from wtVit.efficient import ViT
from wtVit.vit_3d import ViT as Vit3D
from linformer import Linformer

sys.path.append("/home/zarchive/wangwenmiao/project-thymoma/MedicalNet/")
from models import resnet


def generate_model(opt):
    assert opt.backbone_model in ["resnet", "vit"]
    print(
        f"os.environ[CUDA_VISIBLE_DEVICES] = {str(opt.gpu_id)} || {torch.cuda.is_available()} || {torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda", opt.gpu_id)
    else:
        device = torch.device("cpu")
    if opt.backbone_model == "resnet":
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            # fc_input = 256
            fc_input = 512
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 512
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 512
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes,
            )
            fc_input = 2048
        nb_class = 1
        model.conv_seg = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=fc_input, out_features=nb_class, bias=True),
        )
        net_dict = model.state_dict()
        print("loading pretrained model {}".format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        # pretrain = torch.load(opt.pretrain_path, map_location="cuda:0")
        if opt.phase == "train":
            pretrain_dict = {
                k: v for k, v in pretrain["state_dict"].items() if k in net_dict.keys()
            }
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)
        elif opt.phase == "test":
            model.load_state_dict(pretrain)
    elif opt.backbone_model == "vit":
        model = Vit3D(
            image_size=112,  # image size
            frames=112,  # 16 number of frames
            image_patch_size=16,  # image patch size
            frame_patch_size=4,  # 2 frame patch size
            num_classes=1,
            channels=1,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
    model.to(device)
    print(f"sb your model is at: {next(model.parameters()).device}")
    return model
