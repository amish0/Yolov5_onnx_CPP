# Yolov5_onnx_CPP_BatchSize
## Tested On
Platform: Windows 11
CUDA: 11.8
Tensorrt: 12.0
OpenCv: 4.8.0
Visual Studio: 2022
## FPS achieved On USB Webcam
When two WebCam Attached: Batch size: 2---> FPS(avg): 15FPS
When One Web Cam Attached: Batch_Size: 1--> FPS(avg): 24FPS
# Package Installation and Testing
Install the Required package. and Clone this repository
git clone 
cd Yolov5_onnx_CPP
mkdir build
cmake ../
cmake --build .
