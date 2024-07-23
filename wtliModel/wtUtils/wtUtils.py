import nibabel as nib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import pandas as pd
import torchvision.transforms as transform


def wt_rename_file(source_path):
    """
    规范文件名，将ct文件名由12_0000_0000.nii.gz -> 12.nii.gz，统一改成小写
    """
    img_list = os.listdir(source_path)
    # img_list.sort(key=lambda x: int(x[: x.find("_")]))
    for file_name in img_list:
        if file_name.__contains__("_"):
            old_name = file_name
            new_name = file_name[: file_name.find("_")].lower() + ".nii.gz"
            os.rename(source_path + "/" + old_name, source_path + "/" + new_name)
        else:
            old_name = file_name
            new_name = file_name[: file_name.find(".")].lower() + ".nii.gz"
            os.rename(source_path + "/" + old_name, source_path + "/" + new_name)


def adjustMethod1(data_resampled, w_width, w_center):
    # 调整窗位窗宽
    # MIN_BOUND = -160
    # MAX_BOUND = 240
    # image = (data_resampled - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image[image > 1] = 1.
    # image[image < 0] = 0.
    # return image

    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = data_resampled.copy()
    data_adjusted[data_resampled < val_min] = val_min
    data_adjusted[data_resampled > val_max] = val_max

    return data_adjusted


def get_ct_data(img_path, label_path, nii_seg_path):
    """
    分割nii原始文件，根据label中不为0的数据
    """
    # 获取瘤和囊的CT值
    num = 0
    img_list = os.listdir(img_path)
    for item in img_list:
        num += 1
        nii_path = os.path.join(img_path, item)
        mask_path = os.path.join(label_path, item)
        img_nib = nib.load(nii_path)
        img_affine = img_nib.affine
        label_nib = nib.load(mask_path)
        img_ct_data = img_nib.get_fdata()  # Hu data
        if len(img_ct_data.shape) == 5:
            img_ct_data = img_ct_data[:, :, :, 0, 0]
        if len(img_ct_data.shape) == 4:
            img_ct_data = img_ct_data[:, :, :, 0]
        label_ct_data = label_nib.get_fdata()
        img_ct_data_new = adjustMethod1(img_ct_data, 400, 40)
        x, y, z = img_ct_data.shape
        print(f"item:{item}  label_ct_data.shape:{label_ct_data.shape}")
        for item_x in range(x):
            print(f"item_x: {item_x}")
            for item_y in range(y):
                for item_z in range(z):
                    if label_ct_data[item_x, item_y, item_z] == 0:
                        img_ct_data_new[item_x, item_y, item_z] = 0
        nib.Nifti1Image(img_ct_data_new, img_affine).to_filename(
            nii_seg_path + rf"\{item}"
        )


if __name__ == "__main__":
    source_path = "/home/wangwenmiao/20240720xxl/image"
    save_path = "/home/zarchive/wangwenmiao/20240723xxl"
    img_path = "/home/wangwenmiao/20240720xxl/image"
    label_path = "/home/wangwenmiao/20240720xxl/label"
    nii_seg_path = "/home/zarchive/wangwenmiao/20240723xxl/seg_ct_origin"
    # wt_rename_file(label_path)
    get_ct_data(img_path, label_path, nii_seg_path)
