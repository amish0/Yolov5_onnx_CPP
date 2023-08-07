#include "yolov5_model.h"

YoloV5Model::YoloV5Model()
{
    input_width = configparameter.yolov5_config_.input_width;
    input_height = configparameter.yolov5_config_.input_height;
    score_threshold = configparameter.yolov5_config_.score_threshold;
    nms_threshold = configparameter.yolov5_config_.nms_threshold;
    confidence_threshold = configparameter.yolov5_config_.confidence_threshold;
    batch_size = configparameter.yolov5_config_.batch_size;
    x_factors = configparameter.width_ / input_width;
    y_factors = configparameter.height_ / input_height;
    loadClassList();
    checkCuda();
    loadNet();
}

YoloV5Model::~YoloV5Model()
{
 // To DO
}

void YoloV5Model::loadClassList()
{
    std::ifstream ifs(configparameter.yolov5_config_.class_list);
    std::string line;
    while (std::getline(ifs, line))
    {
        class_name.push_back(line);
    }
}

void YoloV5Model::checkCuda()
{
    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        std::cout << "CUDA is available" << std::endl;
        is_cuda = true;
    }
    else
    {
        std::cout << "CUDA is not available" << std::endl;
        is_cuda = false;
    }
}

void YoloV5Model::loadNet()
{
    std::cout << "Model Path: " << configparameter.yolov5_config_.model_path << std::endl;
    auto result = cv::dnn::readNet(configparameter.yolov5_config_.model_path);
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        // result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16); // will be slow on my windows system
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

cv::Mat YoloV5Model::formatYolov5(cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    // std::cout << "result size: " << result.size << std::endl;
    return result;
}

void YoloV5Model::preProcessing(std::vector<cv::Mat> &images)
{
    int batch_size = images.size();
    std::vector<cv::Mat> blobs;
    for (auto &image : images)
    {
        auto input_image = formatYolov5(image);
        blobs.push_back(input_image);
    }
    cv::dnn::blobFromImages(blobs, blob_batch, 1. / 255., cv::Size(input_width, input_height), cv::Scalar(), true, false);
}

void YoloV5Model::detect(std::vector<cv::Mat> &images, std::vector<std::vector<Detection>> &output)
{
    preProcessing(images);
    net.setInput(blob_batch);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    postProcess(outputs, output);
}

void YoloV5Model::postProcess(std::vector<cv::Mat> &outputs, std::vector<std::vector<Detection>> &output)
{
    cv::Mat outp;
    outp = outputs[0];
    for (int b = 0; b < batch_size; b++)
    {
        cv::Mat p = outp.row(b);
        float *data = (float *)p.data;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int i = 0; i < rows; ++i)
        {

            float confidence = data[4];
            if (confidence >= confidence_threshold)
            {

                float *classes_scores = data + 5;
                cv::Mat scores(1,class_name.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                if (max_class_score > score_threshold)
                {

                    confidences.push_back(confidence);

                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((x - 0.5 * w) * x_factors);
                    int top = int((y - 0.5 * h) * y_factors);
                    int width = int(w * x_factors);
                    int height = int(h * y_factors);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            data += 85;
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, score_threshold, nms_threshold, nms_result);
        for (int i = 0; i < nms_result.size(); i++)
        {
            int idx = nms_result[i];
            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output[b].push_back(result);
        }
    }
}