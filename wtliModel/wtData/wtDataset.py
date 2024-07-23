"""
Dataset for training
Written by wtlizzz
"""

import math
import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import time
import SimpleITK as sitk


class WtThymusDataset(Dataset):

    def __init__(self, sets, isTrain=True):
        if isTrain:
            with open(sets.name_list, "r") as f:
                self.img_list = [line.strip() for line in f]
        else:
            with open(sets.val_list_path, "r") as f:
                self.img_list = [line.strip() for line in f]
        self.mask_list = pd.read_csv(sets.label_path)
        print("数据总量为 {} ".format(len(self.img_list)))
        self.root_dir = sets.ct_data_root
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase
        self.label_flag = sets.label_flag

    def __nii2tensorarray__(self, data):
        if len(data.shape) == 5:
            data = data[:, :, :, 0, 0]
        if len(data.shape) == 4:
            data = data[:, :, :, 0]
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        ith_info = self.img_list[idx]
        # img_name = os.path.join(self.root_dir, "image", ith_info)
        # print(f"当前数据索引为 {ith_info}  路径为{img_name}")
        # label_name = os.path.join(self.root_dir, "label", ith_info)
        # img_mask_name = ith_info[: ith_info.find(".")]
        # mask = self.mask_list.loc[self.mask_list["患者编号"] == img_mask_name]
        # mask = mask["value"].values
        # mask = 1 if mask == self.label_flag else 0
        # assert os.path.isfile(img_name)
        # img = nibabel.load(
        #     img_name
        # )  # We have transposed the data from WHD format to DHW
        # assert img is not None
        # label = nibabel.load(label_name)
        # assert label is not None
        # img_array = self.__training_data_process__(img, label)
        img_mask_name = ith_info[: ith_info.find(".")]
        img_name = os.path.join(self.root_dir, ith_info)
        assert os.path.isfile(img_name)
        img = nibabel.load(img_name)
        data = np.asanyarray(img.dataobj)
        mask = self.mask_list.loc[self.mask_list["患者编号"] == img_mask_name]
        mask = mask["value"].values
        mask = 1 if mask == self.label_flag else 0
        img_array = self.__nii2tensorarray__(data)
        return img_array, mask, ith_info

    def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # 原来的体素块尺寸
        originSpacing = itkimage.GetSpacing()
        newSize = np.array(newSize, float)
        factor = originSize / newSize
        newSpacing = originSpacing * factor
        newSize = newSize.astype(np.int)  # spacing肯定不能是整数
        resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        return itkimgResampled

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = label[0, 0, 0]
        non_zeros_idx = np.where(label != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        print(
            f"drop_invalid_range: z:{max_z - min_z},h:{max_h - min_h},w:{max_w - min_w}"
        )
        return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __random_center_crop__(self, data, label):
        from random import random

        """
        Random crop
        """
        target_indexs = np.where(label > 0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array(
            [max_D, max_H, max_W]
        ) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return (
            data[Z_min:Z_max, Y_min:Y_max, X_min:X_max],
            label[Z_min:Z_max, Y_min:Y_max, X_min:X_max],
        )

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """

        if len(data.shape) == 5:
            data = data[:, :, :, 0, 0]
        [depth, height, width] = data.shape
        print(f"data.shape:{[depth, height, width]}")
        scale = [
            self.input_D * 1.0 / depth,
            self.input_H * 1.0 / height,
            self.input_W * 1.0 / width,
        ]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = self.__random_center_crop__(data, label)

        return data, label

    def __training_data_process__(self, data, label):
        # crop data according net input size·
        # data = data.get_data()
        data = np.asanyarray(data.dataobj)
        label = np.asanyarray(label.dataobj)

        # drop out the invalid range
        data = self.__drop_invalid_range__(data, label)

        # crop data
        # data, label = self.__crop_data__(data, label)

        # resize data
        data = self.__resize_data__(data)
        # label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    def __testing_data_process__(self, data):
        # crop data according net input size
        # data = data.get_data()
        data = np.asanyarray(data.dataobj)

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data
