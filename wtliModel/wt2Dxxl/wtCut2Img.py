import nibabel as nib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import torchvision.transforms as transform


def adjustMethod1(data_resampled, w_width, w_center):
    MIN_BOUND = -160
    MAX_BOUND = 240
    image = (data_resampled - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image


def read_nii():
    # np.where找到label中不为0的数值索引，然后找到索引的最大值和最小值，进行图像裁剪
    # np.where返回[179,...][210,...][231,...]，即label[179,210,231]的值不为0
    root_path = r"/home/zarchive/wangwenmiao/20240708xxl"
    img_path = root_path + r"/image"
    label_path = root_path + r"/label"
    png_save_path = root_path + r"/png224"
    num = 0
    img_list = os.listdir(img_path)
    for item in img_list:
        print(f"Processing {item}, {num}:{len(img_list)}")
        num += 1
        nii_path = os.path.join(img_path, item)
        mask_path = os.path.join(label_path, item)
        img_nib = nib.load(nii_path)
        label_nib = nib.load(mask_path)
        img_ct_data = img_nib.get_fdata()  # Hu data

        if len(img_ct_data.shape) == 5:
            print(f"CT data shape: {item},{img_ct_data.shape}")
            img_ct_data = img_ct_data[:, :, :, 0, 0]
        if len(img_ct_data.shape) == 4:
            print(f"CT data shape: {item},{img_ct_data.shape}")
            img_ct_data = img_ct_data[:, :, :, 0]

        label_ct_data = label_nib.get_fdata()
        zero_value = label_ct_data[0, 0, 0]
        non_zeros_idx = np.where(label_ct_data != zero_value)
        max_tumor_num = Counter(non_zeros_idx[2]).most_common(
            1
        )  # 找到分割的面积最大的图片
        # img_ct_data_new = adjustMethod1(img_ct_data, 400, 40)

        [max_x, max_y, max_z] = np.max(np.array(non_zeros_idx), axis=1)
        [min_x, min_y, min_z] = np.min(np.array(non_zeros_idx), axis=1)
        data = img_ct_data[min_x:max_x, min_y:max_y, max_tumor_num[0][0]]
        x, y = data.shape
        img0 = Image.fromarray(data)
        img1 = transform.Pad(
            [int((224 - y) / 2), int((224 - x) / 2)], padding_mode="constant"
        )(img0)
        x, y = img1.size
        print(f"name:{item}  pre_img.size:{x},{y}")
        if x != 224 or y != 224:
            img1 = transform.Pad([224 - x, 224 - y, 0, 0], padding_mode="constant")(
                img1
            )
            print(f"*************{img1.size}****************")
        img1 = img1.convert("L")
        img1.save(png_save_path + rf"/{item}.png")

        # cv2.imwrite(
        #     png_save_path + rf"/{item}.png",
        #     img1,
        # )
        # plt.show()

    # img_cut_ct_data = img_ct_data[min_z:max_z, min_h:max_h, min_w:max_w]


def tune_up_pic():
    # 有部分图片的x，y是223，需要调整成224
    root_path = r"/home/zarchive/wangwenmiao/20240708xxl"
    png_save_path = root_path + r"/png224"
    num = 0
    img_list = os.listdir(png_save_path)
    for item in img_list:
        print(f"Processing {item}, {num}:{len(img_list)}")
        num += 1
        item_path = os.path.join(png_save_path, item)
        pre_img = Image.open(item_path)
        x, y = pre_img.size
        print(f"name:{item}  pre_img.size:{x},{y}")
        if x != 224 or y != 224:
            img1 = transform.Pad([224 - x, 224 - y, 0, 0], padding_mode="constant")(
                pre_img
            )
            print(f"*************{img1.size}****************")
            # img1 = img1.convert("L")
            img1.save(png_save_path + rf"/{item}.png")


if __name__ == "__main__":
    read_nii()
    # tune_up_pic()
