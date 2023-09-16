<!-- Correct this file -->
# Yolov5_onnx_CPP_BatchSize
## Tested On
Platform: Windows 11 <br>
CUDA: 11.8 <br>
Tensorrt: 12.0 <br>
OpenCv: 4.8.0 <br>
Visual Studio: 2022 <br>
## FPS achieved On USB Webcam
when Batch Size: 1 --> FPS(avg): 24FPS <br>
when Batch Size: 2 --> FPS(avg): 15FPS <br>
For batch size 2, I have used two USB WebCam. You have to configure all the parameters in test.ini file. <br>
## FPS achieved On Jetson Nano
We need to test on Jetson Nano. <br>
## FPS achieved On Jetson Xavier NX
We need to test on Jetson Xavier NX. <br>
## FPS achieved On Jetson AGX Xavier
We need to test on Jetson AGX Xavier. <br>
## FPS achieved On Jetson TX2
We need to test on Jetson TX2. <br>
# Package Installation and Testing

Install the Required package. and Clone this repository
```bash
git clone https://github.com/amish0/Yolov5_onnx_CPP.git
cd Yolov5_onnx_CPP
git checkout yolov5_onnx_CPP
mkdir build
cd build
cmake ..
make --build .
```
