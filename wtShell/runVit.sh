#!/bin/bash
gpu_id=0
label_list=("label1" "label2" "label3") 
phase="train"
backbone_model="vit"
# python_interpreter="/home/wangwenmiao/anaconda3/envs/py311/bin/python"
python_interpreter="/home/wangwenmiao/anaconda3/bin/python"
python_source_file="/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/main.py"
ct_data_root="/home/zarchive/wangwenmiao/NNUNET_RESULT"
if [ ! -d "../wtliModel/wtLogs/" ];then
  echo "创建日志wtLogs文件夹"
  mkdir ../wtliModel/wtLogs
fi
if [ ! -d "../log-thymoma/pid/" ];then
  echo "创建pid文件夹"
  mkdir ../log-thymoma/pid 
fi
for item in "${label_list[@]}";
do
    if [ ! -d "../wtliModel/wtWeight/$item/vit/" ];then
        echo "创建wtWeight $item 文件夹"
        mkdir -p ../wtliModel/wtWeight/$item/vit
    fi
    if [ ! -d "../wtliModel/wtLogs/$phase/$item/vit/" ];then
        echo "创建wtLogs $item 文件夹"
        mkdir -p ../wtliModel/wtLogs/$phase/$item/vit
    fi
    if [ ! -d "../wtliModel/wtLogs/$phase/$item/vit/" ];then
        echo "创建wtLogs vit $item 文件夹"
        mkdir -p ../wtliModel/wtLogs/$phase/$item/vit
    fi
    label_flag="L"
    if [ $item = "label2" ];then
        label_flag="C"
    fi
    if [ $item = "label3" ];then
        label_flag="HIGH"
    fi
    nohup \
    /usr/bin/env $python_interpreter \
    $python_source_file \
    --ct_data_root $ct_data_root \
    --name_list ../wtliModel/wtData/data/$item/trainList.csv \
    --val_list_path ../wtliModel/wtData/data/$item/valList.csv \
    --save_weight ../wtliModel/wtWeight/$item/vit/ \
    --label_path ../wtliModel/wtData/label/$item.csv \
    --backbone_model $backbone_model \
    --label_flag $label_flag \
    --summaryWriter_path "../wtliModel/wtLogs/$phase/$item/vit" \
    --epoch 100 \
    --input_W 112 \
    --input_H 112 \
    --input_D 112 \
    --batch_size 4 \
    --save_intervals 20 \
    --gpu_id $gpu_id \
    >../log-thymoma/train/log-$item-vit.log >&1 & echo $! > ../log-thymoma/train/pid/pid-$item-vit
    echo "$item task is running gpu_id [$gpu_id]"
    # gpu_id=$(( ($gpu_id + 1) % 2))
done
