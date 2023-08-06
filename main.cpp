#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include "ini_reader.h"
#include "config_reader.h"
#include "camloader.h"

std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("config_files/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net &net, bool is_cuda)
{
    auto result = cv::dnn::readNet("config_files/yolov5s_2.onnx");
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

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const double SCORE_THRESHOLD = 0.2;
const double NMS_THRESHOLD = 0.45;
const double CONFIDENCE_THRESHOLD = 0.55;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    // std::cout << "result size: " << result.size << std::endl;

    return result;
}

void detect(const std::vector<cv::Mat> &images, cv::dnn::Net &net, std::vector<std::vector<Detection>> &output, const std::vector<std::string> &className)
{
    int batch_size = images.size();
    // cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    std::vector<cv::Mat> blobs;
    std::vector<float> x_factors;
    std::vector<float> y_factors;
    cv::Mat blob_batch;
    
    for (const auto &image : images)
    {
        auto input_image = format_yolov5(image);
        x_factors.push_back(image.cols / INPUT_WIDTH);
        y_factors.push_back(image.rows / INPUT_HEIGHT);
        blobs.push_back(input_image);
    }
    cv::dnn::blobFromImages(blobs, blob_batch, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    auto start = std::chrono::high_resolution_clock::now();
    net.setInput(blob_batch);
    std::vector<cv::Mat> outputs;
    cv::Mat outp;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "inference time " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
    const int dimensions = 85;
    const int rows = 25200;
    outp = outputs[0];
    for (int b = 0; b < batch_size; b++)
    {
        cv::Mat p = outp.row(b);
        // std::cout << "outp size: " << p.size << std::endl;
        // std::cout << "outp size: " << p.size << std::endl;
        float *data = (float *)p.data;
        // float *data = (float *)outputs[0].row(b).data;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int i = 0; i < rows; ++i)
        {

            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD)
            {

                float *classes_scores = data + 5;
                cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                if (max_class_score > SCORE_THRESHOLD)
                {

                    confidences.push_back(confidence);

                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((x - 0.5 * w) * x_factors[b]);
                    int top = int((y - 0.5 * h) * y_factors[b]);
                    int width = int(w * x_factors[b]);
                    int height = int(h * y_factors[b]);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            data += 85;
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
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
int main(int argc, char **argv)
{

    std::vector<std::string> class_list = load_class_list();

    std::cout << "Hello World!\n";
    int cam_num = configparameter.cam_num_;
    configparameter.print_cam_info();
    cout << "cam_num: " << cam_num << endl;
    int width = configparameter.width_;
    int height = configparameter.height_;
    LoadWebcam webcam(width, height, cam_num);
    std::vector<cv::Mat> frames(cam_num);
    webcam.init();
    webcam.start();

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    std::cout << "load net" << std::endl;
    load_net(net, is_cuda);
    std::cout << "load net done" << std::endl;

    int frame_count = 0;
    double fps = -1;
    int total_frames = 0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Detection>> output;

    while (true)
    {
        webcam.next(frames);
        output.clear();
        output.resize(frames.size());
        detect(frames, net, output, class_list);
        frame_count++;
        total_frames++;
        int detections = output.size();
        for (int i = 0; i < detections; ++i)
        {

            auto detection = output[i];
            for (int j = 0; j < detection.size(); j++)
            {
                auto box = detection[j].box;
                auto classId = detection[j].class_id;
                const auto color = colors[classId % colors.size()];
                cv::rectangle(frames[i], box, color, 3);

                cv::rectangle(frames[i], cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                cv::putText(frames[i], class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
        //     auto classId = detection.class_id;
        //     const auto color = colors[classId % colors.size()];
        //     cv::rectangle(frames[i], box, color, 3);

        //     cv::rectangle(frames[i], cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        //     cv::putText(frames[i], class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        // }

        if (frame_count >= 30)
        {

            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {

            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();
            // std::cout << "FPS: " << fps;
            for (int i = 0; i < detections; i++)
            {
                cv::putText(frames[i], fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            }
        }

        for (int i = 0; i < cam_num; ++i)
        {
            imshow("Webcam " + std::to_string(i), frames[i]);
        }
        if (waitKey(1) != -1)
        {
            std::cout << "finished by user\n";
            break;
        }
    }
    webcam.stop();
    destroyAllWindows();
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started:
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
