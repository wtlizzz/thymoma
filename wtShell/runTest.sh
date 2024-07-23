#!/bin/bash
python_interpreter="/home/wangwenmiao/anaconda3/bin/python"
python_source_file="/home/zarchive/wangwenmiao/project-thymoma/wtlizzz-core/wtliModel/mainTest.py"
gpu_id=0
gpu_count=1
model_depth_list=("10") #resnet50=50
label_list=("label1")   #label3需要改dataset，将L改为HIGH
ct_data_root="/home/zarchive/wangwenmiao/20240708xxl/handle-data0712-test"
model_time="202407120903"
model_depth=50
model_path="../wtliModel/wtWeight/$model_time/resnet/label1/best_network.pth"
#不同服务器只改上面部分
for item in "${label_list[@]}"; do
  # for model_depth in "${model_depth_list[@]}";do
  # summaryWriter_path="../wtliModel/wtLogs/$current_datetime/$backbone_model/$phase/$item/"
  run_log_path="../log-thymoma/test/$current_datetime/$backbone_model/"
  test_save_path="../wtliModel/wtData/test/$item/$model_time"
  if [ ! -d $run_log_path ]; then
    echo "创建wtTest 测试日志文件 $run_log_path 文件夹"
    mkdir $run_log_path
  fi
  if [ ! -d $test_save_path ]; then
    echo "创建wtTest 测试数据结果文件 $test_save_path 文件夹"
    mkdir $test_save_path
  fi
  label_flag="L"
  if [ $item = "label3" ]; then
    label_flag="HIGH"
  fi
  nohup \
    /usr/bin/env $python_interpreter \
    $python_source_file \
    --ct_data_root $ct_data_root \
    --name_list ../wtliModel/wtData/data/$item/testList.csv \
    --label_path ../wtliModel/wtData/label/${item}_test.csv \
    --test_save_path $test_save_path \
    --label_flag $label_flag \
    --model_depth $model_depth \
    --pretrain_path $model_path \
    --phase "test" \
    --gpu_id $gpu_id \
    --num_workers 4 \
    --batch_size 1 \
    >../log-thymoma/test/log-$item-md-$model_time.log >&1 &
  echo $! >../log-thymoma/test/pid-$item-md-$model_time
  echo "$item task is running gpu_id [$gpu_id]"
  # done
done
