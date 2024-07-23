#!/bin/bash
gpu_id=0
gpu_count=1
label_list=("label1")
phase="train"
backbone_model="resnet"
model_depth=50 #resnet50=50
python_interpreter="/home/wangwenmiao/anaconda3/bin/python"
python_source_file="/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/wt2Dxxl/wtTrain_2d.py"
data_path="/home/zarchive/wangwenmiao/20240708xxl/png224"
label_path="/home/zarchive/wangwenmiao/20240708xxl/LABEL.csv"
pretrain_path="../model/pretrain/resnet_$model_depth.pth"
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
        --data_path $data_path \
        --label_path $label_path \
        --label_flag $label_flag \
        --pretrain_path $pretrain_path \
        --gpu_id $gpu_id \
        --save_weight $save_weight \
        --summaryWriter_path $summaryWriter_path \
        --num_workers 8 \
        --epochs 100 \
        --batch_size 8 \
        >$run_log_path/log-$item-md-$model_depth.log >&1 &
    echo $! >$run_log_path/pid-$item-md-$model_depth
    echo "$item task is running gpu_id [$gpu_id]"
    gpu_id=$((($gpu_id + 1) % $gpu_count))
done
