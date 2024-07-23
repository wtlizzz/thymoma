#!/bin/bash
gpu_id=0
gpu_count=1
label_list=("label1")
phase="train"
backbone_model="resnet"
model_depth=50 #resnet50=50
python_interpreter="/home/wangwenmiao/anaconda3/bin/python"
python_source_file="/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/main.py"
ct_data_root="/home/zarchive/wangwenmiao/20240708xxl/image-0715-2"
pretrain_path="../model/pretrain/resnet_$model_depth.pth"
pos_loss_weight=5
current_datetime=$(date +'%Y%m%d%H%M')
echo $current_datetime
for item in "${label_list[@]}"; do
    save_weight="../wtliModel/wtWeight/$current_datetime/$backbone_model/$item/"
    summaryWriter_path="../wtliModel/wtLogs/$current_datetime/$backbone_model/$phase/$item/"
    run_log_path="../log-thymoma/train/$current_datetime/$backbone_model/"
    if [ ! -d $save_weight ]; then
        echo "创建save_weight $save_weight/ 文件夹"
        mkdir -p $save_weight
    fi
    if [ ! -d "$summaryWriter_path" ]; then
        echo "创建summaryWriter_path $summaryWriter_path 文件夹"
        mkdir -p $summaryWriter_path
    fi
    if [ ! -d "$run_log_path" ]; then
        echo "创建pid $run_log_path文件夹"
        mkdir -p $run_log_path
    fi
    label_flag="N"
    if [ $item = "label2" ]; then
        label_flag="C"
    fi
    if [ $item = "label3" ]; then
        label_flag="HIGH"
    fi
    nohup \
        /usr/bin/env $python_interpreter \
        $python_source_file \
        --ct_data_root $ct_data_root \
        --name_list ../wtliModel/wtData/data/$item/train.csv \
        --val_list_path ../wtliModel/wtData/data/$item/val.csv \
        --label_path ../wtliModel/wtData/label/$item.csv \
        --backbone_model $backbone_model \
        --label_flag $label_flag \
        --pretrain_path $pretrain_path \
        --model_depth $model_depth \
        --gpu_id $gpu_id \
        --pos_loss_weight $pos_loss_weight \
        --save_weight $save_weight \
        --summaryWriter_path $summaryWriter_path \
        --save_intervals 20 \
        --num_workers 8 \
        --epoch 100 \
        --input_W 112 \
        --input_H 112 \
        --input_D 112 \
        --batch_size 2 \
        >$run_log_path/log-$item-md-$model_depth.log >&1 &
    echo $! >$run_log_path/pid-$item-md-$model_depth
    echo "$item task is running gpu_id [$gpu_id]"
    gpu_id=$((($gpu_id + 1) % $gpu_count))
done
