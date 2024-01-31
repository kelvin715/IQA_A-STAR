#!/bin/bash

# 定义一个函数来运行 yolo 命令并等待其完成
run_yolo() {
  data_path="$1"
  task_name="$2"
  yolo task=detect mode=train model=yolov8-dropblock.yaml data="$data_path" epochs=1000 batch=16 imgsz=640 workers=8 name="$task_name" verbose=True device="0,1,2,3" &
  pid=$!
  echo "Started $task_name with PID $pid"
  wait $pid
}

# 创建一个数组，包含所有的数据文件路径和名称
declare -a tasks=(
"/Data4/student_zhihan_data/data/GC10-DET_BilateralBlur_180/data.yaml GC10-DET_BilateralBlur_180_detect_by_yolov8n_with_dropblock(p=0.05 s=5)"
"/Data4/student_zhihan_data/data/GC10-DET_BilateralBlur_240/data.yaml GC10-DET_BilateralBlur_240_detect_by_yolov8n_with_dropblock(p=0.05 s=5)"
"/Data4/student_zhihan_data/data/GC10-DET_BilateralBlur_300/data.yaml GC10-DET_BilateralBlur_300_detect_by_yolov8n_with_dropblock(p=0.05 s=5)"
)

# 控制同时运行的任务数量
MAX_JOBS=1
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
