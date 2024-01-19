#!/bin/bash

# 定义一个函数来运行 yolo 命令并等待其完成
run_yolo() {
  data_path="$1"
  task_name="$2"
  yolo task=detect mode=train model=yolov8n.yaml data="$data_path" epochs=1000 batch=16 imgsz=640 workers=8 name="$task_name" verbose=True device="0,1,2,3" &
  pid=$!
  echo "Started $task_name with PID $pid"
  wait $pid
}

# 创建一个数组，包含所有的数据文件路径和名称
declare -a tasks=(
"/Data4/student_zhihan_data/data/GC10-DET_brightness_-100/data.yaml GC10-DET_brightness_-100_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_-50/data.yaml GC10-DET_brightness_-50_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_-30/data.yaml GC10-DET_brightness_-30_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_-20/data.yaml GC10-DET_brightness_-20_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_-10/data.yaml GC10-DET_brightness_-10_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_10/data.yaml GC10-DET_brightness_10_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_20/data.yaml GC10-DET_brightness_20_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_30/data.yaml GC10-DET_brightness_30_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_50/data.yaml GC10-DET_brightness_50_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_60/data.yaml GC10-DET_brightness_60_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_70/data.yaml GC10-DET_brightness_70_detect_by_yolov8n_with_dropout(p=0.1)"
"/Data4/student_zhihan_data/data/GC10-DET_brightness_90/data.yaml GC10-DET_brightness_90_detect_by_yolov8n_with_dropout(p=0.1)"
)

# 控制同时运行的任务数量
MAX_JOBS=4
current_jobs=0

# 遍历任务并运行它们，同时保持最多 MAX_JOBS 个任务运行
for task in "${tasks[@]}"; do
  run_yolo $task &
  ((current_jobs++))

  if ((current_jobs >= MAX_JOBS)); then
    wait -n
    ((current_jobs--))
  fi
done

# 等待所有后台任务完成
wait
echo "所有任务完成"
