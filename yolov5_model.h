#ifndef YOLOV5_MODEL_H
#define YOLOV5_MODEL_H
#include <string>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include "config_reader.h"

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class YoloV5Model
{
public:
    YoloV5Model();
    ~YoloV5Model();
    void loadClassList();
    void loadNet();
    cv::Mat formatYolov5(cv::Mat &source);
    void preProcessing(std::vector<cv::Mat> &images);
    void detect(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &output);
    void postProcess(std::vector<cv::Mat> &outputs, std::vector<std::vector<Detection>> &output);
    //void drawFps(cv::Mat &frame, double fps);
    //void run();
    void checkCuda();
    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
    std::vector<std::string> class_name;
private:
    
    cv::dnn::Net net;
    cv::Mat blob_batch;
    float input_width=640;
    float input_height=640;
    double score_threshold = 0.2;
    double nms_threshold = 0.45;
    double confidence_threshold = 0.55;
    const int dimensions = 85;
    const int rows = 25200;
    float x_factors;
    float y_factors;
    int batch_size;
    bool is_cuda = false;
};
#endif // YOLOV5_MODEL_H