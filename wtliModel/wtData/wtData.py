import nibabel as nb
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from nibabel.viewers import OrthoSlicer3D


# 创建训练集和测试集的文件索引
def wt_create_data_index(image_path, label_path, out_path):
    img_list = os.listdir(image_path)
    img_list.sort(key=lambda x: int(x[: x.find("_")]))
    label_list = os.listdir(label_path)
    label_list.sort(key=lambda x: int(x[: x.find("_")]))
    input_lines = []

    with open(out_path, "w") as f:
        for idx in range(len(img_list)):
            index_line = img_list[idx] + " " + label_list[idx] + "\n"
            f.write(index_line)


def wt_get_data_label(label_path):
    with open(label_path, "r") as f:
        data = f.readlines()  # txt中所有字符串读入data
    label_list = []
    for line in data:
        label = line.split("\t")[1][:1]
        label_list.append(label)
    return label_list


def wt_data_test(sets):
    """
    验证源数据目录读取 & 确认输入数据参数
    input_D
    input_H
    input_W
    """
    with open(sets.img_list, "r") as f:
        img_list = [line.strip() for line in f]
    print("Processing {} datas".format(len(img_list)))
    root_dir = sets.data_root
    input_D = sets.input_D
    input_H = sets.input_H
    input_W = sets.input_W
    for idx in range(len(img_list)):
        ith_info = img_list[idx].split(" ")
        img_name = os.path.join(root_dir, "imagesTr", ith_info[0])
        label_name = os.path.join(root_dir, "labelsTr", ith_info[1])
        assert os.path.isfile(img_name)
        assert os.path.isfile(label_name)
        img = nb.load(img_name)  # We have transposed the data from WHD format to DHW
        mask = nb.load(label_name)
        assert img is not None
        assert mask is not None
        data_img = np.asanyarray(img.dataobj)
        [depth, height, width] = data_img.shape


file_path = "/home/liwentao/wwm/NNUNET_RESULT/image"


def step_2():
    """划分数据集，将数据集写入到文件中"""
    # img_list = os.listdir(file_path)
    df = pd.read_csv("wtlizzz-core/wtliModel/wtData/label/label1.txt")
    img_list = df["患者编号"] + ".nii.gz"
    train_set, test_set = train_test_split(img_list, test_size=0.4, random_state=42)
    with open("wtlizzz-core/wtliModel/wtData/data/label1/trainList.txt", "w") as f:
        for name in train_set:
            line = name + "\n"
            f.write(line)
    with open("wtlizzz-core/wtliModel/wtData/data/label1/testList.txt", "w") as f:
        for name in test_set:
            line = name + "\n"
            f.write(line)


def check_data(img_path):
    """确认数据目录中数据是都存在的，可能有的文件被删掉"""
    img_list = os.listdir(img_path)
    for label_item in ["label1", "label2", "label3"]:
        for pharse in ["trainList", "valList"]:
            heads = ["img_name", "is_delete"]
            df = pd.read_csv(
                f"wtlizzz-core/wtliModel/wtData/data/{label_item}/{pharse}.csv",
                encoding="utf-8",
                names=heads,
            )
            df["img_name"] = df.img_name.str.lower()
            df["is_delete"] = False
            df_list = df.values.tolist()
            for df_item in df_list:
                if df_item[0] not in img_list:
                    print(f"df_item:{df_item} not in df_list")
                    df.loc[(df[df["img_name"] == df_item[0]].index, "is_delete")] = True
            df.to_csv(
                f"wtlizzz-core/wtliModel/wtData/data/{label_item}/{pharse}2.csv",
                index=False,
                header=False,
            )
            df = df.loc[df["is_delete"] == False]
            df["img_name"].to_csv(
                f"wtlizzz-core/wtliModel/wtData/data/{label_item}/{pharse[:-4]}.csv",
                index=False,
                header=False,
            )


def get_test_label():
    """读取csv，获取label"""
    label_path = "/home/zarchive/wangwenmiao/20240708xxl/originTest.csv"
    df = pd.read_csv(label_path, encoding="utf-8")
    df["患者编号"] = df.患者编号.str.lower()
    df_label1 = df[["患者编号", "label1"]]
    df_label1 = df_label1.rename(columns={"label1": "value"})
    df_label2 = df[df["LABEL2"].notnull()]
    df_label2 = df_label2[["患者编号", "LABEL2"]]
    df_label2 = df_label2.rename(columns={"LABEL2": "value"})
    df_label3 = df[df["LABEL3"].notnull()]
    df_label3 = df_label3[["患者编号", "LABEL3"]]
    df_label3 = df_label3.rename(columns={"LABEL3": "value"})
    df_label1.to_csv(
        "./wtlizzz-core/wtliModel/wtData/label/label1_test.csv", index=False
    )
    df_label2.to_csv(
        "./wtlizzz-core/wtliModel/wtData/label/label2_test.csv", index=False
    )
    df_label3.to_csv(
        "./wtlizzz-core/wtliModel/wtData/label/label3_test.csv", index=False
    )
    df_label1 = df_label1["患者编号"] + ".nii.gz"
    df_label1.to_csv(
        "./wtlizzz-core/wtliModel/wtData/data/label1/testList.csv",
        index=False,
        header=False,
    )
    df_label2 = df_label2["患者编号"] + ".nii.gz"
    df_label2.to_csv(
        "./wtlizzz-core/wtliModel/wtData/data/label2/testList.csv",
        index=False,
        header=False,
    )
    df_label3 = df_label3["患者编号"] + ".nii.gz"
    df_label3.to_csv(
        "./wtlizzz-core/wtliModel/wtData/data/label3/testList.csv",
        index=False,
        header=False,
    )


def get_label():
    """读取csv，获取label"""
    label_path = "/home/zarchive/wangwenmiao/NNUNET_RESULT/LABEL.csv"
    df = pd.read_csv(label_path, encoding="utf-8")
    df_label1 = df[["患者编号", "label1"]]
    df_label1 = df_label1.rename(columns={"label1": "value"})
    df_label2 = df[df["LABEL2"].notnull()]
    df_label2 = df_label2[["患者编号", "LABEL2"]]
    df_label2 = df_label2.rename(columns={"LABEL2": "value"})
    df_label3 = df[df["LABEL3"].notnull()]
    df_label3 = df_label3[["患者编号", "LABEL3"]]
    df_label3 = df_label3.rename(columns={"LABEL3": "value"})
    df_label1.to_csv("./wtlizzz-core/wtliModel/wtData/label/label1.csv", index=False)
    df_label2.to_csv("./wtlizzz-core/wtliModel/wtData/label/label2.csv", index=False)
    df_label3.to_csv("./wtlizzz-core/wtliModel/wtData/label/label3.csv", index=False)


def data_handle(img_path):
    """有的ct文件shape是5维的data.shape:(53, 62, 10, 1, 2)，将5维转化成3维
    img_path = "/home/liwentao/wwm/NNUNET_RESULT/image"
    """
    # img = nb.load("C:\\Users\\61708\\Desktop\\123123\\KF6.nii.gz")  # 读取nii格式文件
    img_list = os.listdir(img_path)
    for img in img_list:
        label = nb.load(img_path + "/" + img)
        data = np.asanyarray(label.dataobj)
        print(f"img:{img},  data.shape:{data.shape}")
        label = None


def data_divide():
    """
    20240705，数据分组更新，重新划分数据集
    """
    df = pd.read_csv(
        "wtlizzz-core/wtliModel/wtData/origin/origin.csv",
        encoding="utf-8",
    )
    """ 生成label，不分训练集和验证集 """
    df_label1 = df[df["label1"].notnull()]
    df_label1 = df_label1[["患者编号", "label1"]]
    df_label1 = df_label1.rename(columns={"label1": "value"})
    df_label1.to_csv("wtlizzz-core/wtliModel/wtData/data/label1/label.csv", index=False)
    df_label2 = df[df["label2"].notnull()]
    df_label2 = df_label2[["患者编号", "label2"]]
    df_label2 = df_label2.rename(columns={"label2": "value"})
    df_label2.to_csv("wtlizzz-core/wtliModel/wtData/data/label2/label.csv", index=False)
    df_label3 = df[df["label3"].notnull()]
    df_label3 = df_label3[["患者编号", "label3"]]
    df_label3 = df_label3.rename(columns={"label3": "value"})
    df_label3.to_csv("wtlizzz-core/wtliModel/wtData/data/label3/label.csv", index=False)

    """ 划分数据集，先划分label3，再划分label2，再划分label1 """
    df_label1["患者编号"] = df_label1["患者编号"] + ".nii.gz"
    train_label1, val_label1 = train_test_split(
        df_label1, test_size=0.2, random_state=42
    )
    print(f"train_label1: {train_label1.value.value_counts()}")
    print(f"val_label1: {val_label1.value.value_counts()}")
    train_label1.to_csv(
        "wtlizzz-core/wtliModel/wtData/data/label1/trainList.csv",
        index=False,
        header=False,
    )
    val_label1.to_csv(
        "wtlizzz-core/wtliModel/wtData/data/label1/valList.csv",
        index=False,
        header=False,
    )

    # df_label3["label"] = df_label3["label"] + ".nii.gz"
    # train_label3, val_label3 = train_test_split(df_label3, test_size=0.2, random_state=42)

    # print(f"label3: {df["value"].value_counts()}")
    # train_label3.to_csv(
    #     "wtlizzz-core/wtliModel/wtData/data/label3/trainList.csv",
    #     index=False,
    #     header=False,
    # )
    # val_label3.to_csv(
    #     "wtlizzz-core/wtliModel/wtData/data/label3/valList.csv",
    #     index=False,
    #     header=False,
    # )

    # df_label2_train = df[(df["分组"] == 1) & (df["label2"].notnull())]
    # df_label2_train = df_label2_train["患者编号"] + ".nii.gz"
    # df_label2_train.to_csv(
    #     "wtlizzz-core/wtliModel/wtData/data/label2/trainList.csv",
    #     index=False,
    #     header=False,
    # )
    # df_label2_val = df[(df["分组"] == 0) & (df["label2"].notnull())]
    # df_label2_val = df_label2_val["患者编号"] + ".nii.gz"
    # df_label2_val.to_csv(
    #     "wtlizzz-core/wtliModel/wtData/data/label2/valList.csv",
    #     index=False,
    #     header=False,
    # )


def ct_pic_handle():
    img = nb.load(
        "G:\\data-CT\\alldata\\imagesTr\\11_0000_0000.nii.gz"
    )  # 读取nii格式文件
    img_affine = img.affine
    data = np.asanyarray(img.dataobj)
    nb.Nifti1Image(data, img_affine).to_filename(".\\img_affine2.nii.gz")
    OrthoSlicer3D(data.transpose(1, 2, 0)).show()


def get_ct_pic():
    image_path = "G:\\data-CT\\alldata\\imagesTr\\11_0000_0000.nii.gz"
    mask_path = "G:\\data-CT\\alldata\\labelsTr\\11_0000.nii.gz"
    img = nb.load(image_path)
    img_affine = img.affine
    mask = nb.load(mask_path)
    data = np.asanyarray(img.dataobj)
    label = np.asanyarray(mask.dataobj)

    # zero_value = data[0, 0, 0]
    zero_value = label[0, 0, 0]
    non_zeros_idx = np.where(label != zero_value)

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    data = data[min_z:max_z, min_h:max_h, min_w:max_w]

    # nb.save(data, ".\\img_affine1.nii.gz")
    nb.Nifti1Image(data, img_affine).to_filename(".\\img_affine1.nii.gz")

    if label is not None:
        return (
            data[min_z:max_z, min_h:max_h, min_w:max_w],
            label[min_z:max_z, min_h:max_h, min_w:max_w],
        )
    else:
        return data[min_z:max_z, min_h:max_h, min_w:max_w]


def __drop_invalid_range__(self, volume, label=None):
    """
    Cut off the invalid area
    """
    zero_value = label[0, 0, 0]
    non_zeros_idx = np.where(label != zero_value)

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    print(f"drop_invalid_range: z:{max_z - min_z},h:{max_h - min_h},w:{max_w - min_w}")
    return volume[min_z:max_z, min_h:max_h, min_w:max_w]


def __itensity_normalize_one_volume__(volume):
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def data_preprocessing():
    data_divide()


if __name__ == "__main__":
    data_divide()
