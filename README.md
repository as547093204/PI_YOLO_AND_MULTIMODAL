This project only provides the content shown in the paper. Due to the confidentiality of the hospital's later research, it does not provide any completed or pre trained model files.
If you want to reproduce the main directory structure of the project, you need to build it based on the code and paper.  
After configuring the environment, directory structure, and supporting datasets, this project provides datasets, training, ablation experiments, and inference code files.
Yolov8pytorch-Pi.part [num]. rar is a YOLO model dataset consisting of a total of 5 volumes
Ulcer-evm_iPPGSignal-data.rar is the iPPG signal file after EVM processing
Draw_frame_fx.py is used to extract frames from the original video stream dataset of pressure ulcer sites as YOLO's image dataset
YOLO_Ablation_train.sh is the running script for YOLO ablation experiments
All_iPPGsignal-experiments_results.comsv are the results of various training experiments on the iPPG multimodal signal classification model
Article Performance. mp4 is a video demonstration of the complete process of the system method
Combine_demo. py is a demonstration demo of the complete process
Download_images. py can be used to obtain and download open-source PI datasets
Model. py is the architecture file for the iPPG multimodal signal model
Multimods_5classicy_euler_treatment_noUI.Py is a test demo of the iPPG multimodal signal model and can provide nursing advice
Multimods_Ablation_train.cy is a script for running an iPPG multimodal ablation experiment
Real_time_infer.cy is a real-time running script for YOLO to detect PI targets through a camera
Video_image_process.exe is the EVM we implemented
Whole_yolo_desults. csv is the result of various training experiments on the yolo fine-tuning model
Yolov8m_stucture. py is the model architecture file for YOLOv8m
