#!/bin/bash

# # 定义一个函数来运行 yolo 命令并等待其完成
# run_yolo() {
#   data_path="$1"
#   task_name="$2"
#   yolo task=detect mode=train model=yolov8n.yaml data="$data_path" epochs=1000 batch=16 imgsz=640 workers=8 name="$task_name" verbose=True device="0,1,2,3" &
#   pid=$!
#   echo "Started $task_name with PID $pid"
#   wait $pid

  

# }

declare -a tasks=(
"python /Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/test.py --device=0 --dataset='Sharpen_'"
"python /Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/test.py --device=1 --dataset='Transform'"
"python /Data4/student_zhihan_data/source_code/IQA_A-STAR/source_code/Mydemo/test.py --device=2 --dataset='Median'"
)

for task in "${tasks[@]}"; do
  eval $task &
done


# # 控制同时运行的任务数量
# MAX_JOBS=2
# current_jobs=0

# # 遍历任务并运行它们，同时保持最多 MAX_JOBS 个任务运行
# for task in "${tasks[@]}"; do
#   run_yolo $task &
#   ((current_jobs++))

#   if ((current_jobs >= MAX_JOBS)); then
#     wait -n
#     ((current_jobs--))
#   fi
# done

# # 等待所有后台任务完成
# wait
# echo "所有任务完成"
