import math
import os
import random
import pandas as pd
import numpy as np
import skimage
from torch.utils.data import Dataset
import nibabel as nb
from scipy import ndimage
import time
import SimpleITK as sitk
from nibabel.viewers import OrthoSlicer3D
from glob import glob


def __training_data_process__(data, label, ct_image_affine):

    data = __drop_invalid_range__(data, label)

    nb.Nifti1Image(data, ct_image_affine).to_filename(
        "/home/zarchive/wangwenmiao/20240708xxl/handle-data/xxxxx2.nii.gz"
    )

    # crop data
    # data, label = self.__crop_data__(data, label)
    # resize data
    data = __resize_data__(data)

    nb.Nifti1Image(data, ct_image_affine).to_filename(
        "/home/zarchive/wangwenmiao/20240708xxl/handle-data/xxxxx3.nii.gz"
    )

    # label = self.__resize_data__(label)
    # normalization datas
    data = __itensity_normalize_one_volume__(data)

    nb.Nifti1Image(data, ct_image_affine).to_filename(
        "/home/zarchive/wangwenmiao/20240708xxl/handle-data/xxxxx4.nii.gz"
    )
    return data


def __drop_invalid_range__(volume, label=None):
    """
    Cut off the invalid area
    """
    zero_value = label[0, 0, 0]
    non_zeros_idx = np.where(label != zero_value)

    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

    z_long = max_z - min_z
    h_long = max_h - min_h
    w_long = max_w - min_w

    z_long = z_long if z_long % 2 == 0 else z_long + 1
    h_long = h_long if h_long % 2 == 0 else h_long + 1
    w_long = w_long if w_long % 2 == 0 else w_long + 1

    z_offset = round((224 - z_long) / 2)
    h_offset = round((224 - h_long) / 2)
    w_offset = round((224 - w_long) / 2)

    max_z = max_z + z_offset
    max_h = max_h + h_offset
    max_w = max_w + w_offset
    min_z = min_z - z_offset
    min_h = min_h - h_offset
    min_w = min_w - w_offset

    # min_z = min_z - 10 if min_z - 10 > 0 else 0
    # min_h = min_h - 10 if min_h - 10 > 0 else 0
    # min_w = min_w - 10 if min_w - 10 > 0 else 0
    print(f"drop_invalid_range: z:{max_z - min_z},h:{max_h - min_h},w:{max_w - min_w}")
    data = volume[min_z:max_z, min_h:max_h, min_w:max_w]
    if len(data.shape) == 5:
        data = data[:, :, :, 0, 0]
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    [depth, height, width] = data.shape
    print(f"data.shape:{[depth, height, width]}")
    return data


def __resize_data__(data):
    """
    Resize the data to the input size
    """

    if len(data.shape) == 5:
        data = data[:, :, :, 0, 0]
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    [depth, height, width] = data.shape
    print(f"data.shape:{[depth, height, width]}")
    scale = [
        224 * 1.0 / depth,
        224 * 1.0 / height,
        224 * 1.0 / width,
    ]
    data = ndimage.zoom(data, scale, order=0)

    return data


def __new_resize_data__(data):
    """
    Resize the data to the input size
    """

    if len(data.shape) == 5:
        data = data[:, :, :, 0, 0]
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    [depth, height, width] = data.shape
    print(f"data.shape:{[depth, height, width]}")
    scale = [
        224 * 1.0 / depth,
        224 * 1.0 / height,
        224 * 1.0 / width,
    ]
    data = ndimage.zoom(data, scale, order=0)

    return data


def __itensity_normalize_one_volume__(volume):
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def __nii2tensorarray__(data):
    [z, y, x] = data.shape
    new_data = np.reshape(data, [1, z, y, x])  # 医学图像不能用这个
    new_data = new_data.astype("float32")

    return new_data


def resample(data, ori_space, header, spacing):
    ### Calculate new space
    new_width = round(ori_space[0] * header["pixdim"][1] / spacing[0])
    new_height = round(ori_space[1] * header["pixdim"][2] / spacing[1])
    new_channel = round(ori_space[2] * header["pixdim"][3] / spacing[2])
    new_space = [new_width, new_height, new_channel]

    data_resampled = skimage.transform.resize(
        data,
        new_space,
        order=1,
        mode="reflect",
        cval=0,
        clip=True,
        preserve_range=False,
        anti_aliasing=True,
        anti_aliasing_sigma=None,
    )
    return data_resampled


def adjustMethod1(data_resampled, w_width, w_center):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = data_resampled.copy()
    data_adjusted[data_resampled < val_min] = val_min
    data_adjusted[data_resampled > val_max] = val_max

    return data_adjusted


def saved_preprocessed(savedImg, origin, direction, xyz_thickness, saved_name):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    sitk.WriteImage(newImg, saved_name)


def window_transform(ct_array, windowWidth, windowCenter):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    maxWindow = float(windowCenter) + 0.5 * float(windowWidth)
    ct_copy = ct_array.copy()
    ct_copy[ct_array < minWindow] = minWindow
    ct_copy[ct_array > maxWindow] = maxWindow
    return ct_copy


def nii_load_ct():
    img_path = "/home/zarchive/wangwenmiao/20240708xxl/QD-IMAGE/"
    # img_path = "/home/zarchive/wangwenmiao/20240708xxl/handle-data0712-test"
    saved_path = "/home/zarchive/wangwenmiao/20240708xxl/handle-data0712-test"
    img_list = os.listdir(img_path)
    for img in img_list:
        img = nb.load(os.path.join(img_path, img))
        data = np.asanyarray(img.dataobj)
        print(f"data shape:{data.shape}")


def sitk_load_ct():
    img_path = "/home/zarchive/wangwenmiao/20240708xxl/label/"
    # img_path = "/home/zarchive/wangwenmiao/20240708xxl/handle-data0712-test"
    saved_path = "/home/zarchive/wangwenmiao/20240708xxl/handle-data0713-test"
    img_list = os.listdir(img_path)
    all_num = len(img_list)
    num = 1
    for img in img_list:
        print(f"img name:{img}  {num}:{all_num}")
        num += 1
        ct = sitk.ReadImage(os.path.join(img_path, img))
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        xyz_thickness = ct.GetSpacing()
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(
            sitk.ReadImage(
                os.path.join(
                    "/home/zarchive/wangwenmiao/20240708xxl/label/",
                    img,
                )
            )
        )
        non_zeros_idx = np.where(seg_array != 0)
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        print(f"label.shape:{max_z - min_z}; {max_h - min_h}; {max_w - min_w}")
        print(f"data.shape:{ct_array.shape}")
        # tumor_wl = window_transform(ct_array, 400, 40)
        # res_ct_data = __drop_invalid_range__(tumor_wl, seg_array)
        # res_ct_data = __itensity_normalize_one_volume__(res_ct_data)
        # saved_name = os.path.join(saved_path, img)
        # saved_preprocessed(res_ct_data, origin, direction, xyz_thickness, saved_name)


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32)  # spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def handle_ct_data():
    # image-0715-1:图像调整窗宽窗位
    # image-0715-2:图像调整大小，线性差值，调整为224,224,224
    # image-0716-1:基于0715-1做肿瘤图像分割，分割为224,224,224，不缩放，以肿瘤为中心，其他填充0
    img_path = "/home/zarchive/wangwenmiao/20240708xxl/image-0715-1"
    saved_path = "/home/zarchive/wangwenmiao/20240708xxl/image-0716-1"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
        print("Folder created")
    else:
        print("Folder already exists")
    img_list = os.listdir(img_path)
    all_num = len(img_list)
    num = 1
    for img in img_list:
        print(f"img name:{img}  {num}:{all_num}")
        num += 1
        ct = sitk.ReadImage(os.path.join(img_path, img))
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        xyz_thickness = ct.GetSpacing()
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(
            sitk.ReadImage(
                os.path.join(
                    "/home/zarchive/wangwenmiao/20240708xxl/label/",
                    img,
                )
            )
        )
        # non_zeros_idx = np.where(seg_array != 0)
        # [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        # [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        # print(f"label.shape:{max_z - min_z}; {max_h - min_h}; {max_w - min_w}")
        print(f"data.shape:{ct_array.shape}")
        tumor_wl = window_transform(ct_array, 400, 40)
        res_ct_data = __drop_invalid_range__(tumor_wl, seg_array)
        res_ct_data = __itensity_normalize_one_volume__(res_ct_data)
        saved_name = os.path.join(saved_path, img)
        saved_preprocessed(tumor_wl, origin, direction, xyz_thickness, saved_name)

        # itkimgResampled = resize_image_itk(
        #     itkimage, (112, 112, 112), resamplemethod=sitk.sitkLinear
        # )
        # 目标size为(128,128,128)
        # 这里要注意：mask用最近邻插值sitkNearestNeighbor，CT图像用线性插值sitkLinear
        # saved_name = os.path.join(saved_path, img)
        # sitk.WriteImage(itkimgResampled, saved_name)


def data_preprocessing2():
    handle_ct_data()
    # nii_load_ct()


if __name__ == "__main__":
    data_preprocessing2()
