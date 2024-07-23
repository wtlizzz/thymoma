import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from wtData.wtDataset import WtThymusDataset
from wtModel.wtModel import generate_model
from wtSet.wtSetting import parse_opts

import setproctitle

setproctitle.setproctitle("This is wangwenmiao de nature ^.^!")


def test_init_opt():
    sets = parse_opts()
    sets.ct_data_root = "/home/zarchive/wangwenmiao/20240708xxl/handle-data0712-test"
    sets.name_list = "/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/wtData/data/label1/testList.csv"
    sets.label_path = "/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/wtData/label/label1_test.csv"
    sets.test_save_path = "/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/wtData/test/label1/202407120820"
    sets.label_flag = "L"
    sets.pretrain_path = "/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/wtWeight/202407120820/resnet/label1/best_network.pth"
    sets.phase = "test"
    sets.model_depth = 10
    sets.gpu_id = 0
    return sets


def wt_test(init_set, model):
    from tqdm import tqdm

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    dataset_val = WtThymusDataset(init_set)
    data_loader_test = DataLoader(
        dataset_val,
        batch_size=init_set.batch_size,
        shuffle=True,
        num_workers=init_set.num_workers,
        pin_memory=init_set.pin_memory,
    )
    res_table = {
        "img_name": [],
        "predict_value": [],
        "prob_out": [],
        "predict_stand": [],
        "label": [],
        "result_boolean": [],
    }
    res_df = pd.DataFrame(res_table)
    num_correct = 0
    with torch.no_grad():
        for x, label, img_name in tqdm(data_loader_test):
            x = x.float()
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            logits = logits.reshape([label.cpu().numpy().shape[0]])
            prob_out = nn.Sigmoid()(logits)
            pro_list = prob_out.detach().cpu().numpy()
            for i in range(pro_list.shape[0]):
                pro_list[i]
                if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
                    num_correct += 1

            res = logits.cpu().item()
            # logits = logits.reshape([label.cpu().numpy().shape[0]])

            label_value = label.item()
            prob_out_value = prob_out.item()

            # predict_stand = round(prob_out_value)

            if prob_out_value < 0.5:
                predict_stand = 0
            else:
                predict_stand = 1

            result_boolean = predict_stand == label_value
            print(
                f"res: {res}, prob_out: {prob_out_value}, predict_stand: {predict_stand}, label: {label_value},result_boolean: {result_boolean}"
            )
            res_df.loc[len(res_df)] = [
                img_name[0],
                res,  #
                prob_out_value,
                predict_stand,
                label_value,
                result_boolean,
            ]
    res_df.to_csv(init_set.test_save_path + "/wt_test.csv", index=False)


if __name__ == "__main__":
    # init_set = parse_opts()
    init_set = test_init_opt()
    print(f"init_set:{init_set}")
    model = generate_model(init_set)
    wt_test(init_set, model)
