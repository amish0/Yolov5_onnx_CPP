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
#include "yolov5_model.h"
#include "sort_tracker.h"
#include "config.h"
// #include "Hungarian.h"
// #include "KalmanTracker.h"

int main(int argc, char **argv)
{
    std::cout << "Hello World!\n";
    int cam_num = configparameter.cam_num_;
    configparameter.print_cam_info();
    cout << "cam_num: " << cam_num << endl;
    configparameter.print_yolov5_info();
    int width = configparameter.width_;
    int height = configparameter.height_;
    LoadWebcam webcam(width, height, cam_num);
    YoloV5Model yolov5;
    std::vector<cv::Mat> frames(cam_num);
    webcam.init();
    webcam.start();
    // create array of an object of SortTracker class length of array is cam_num
    SortTracker *obj_tracker = new SortTracker[cam_num];
    for (int i = 0; i < cam_num; ++i)
    {
        obj_tracker[i].init(MAX_AGES, MIN_HITS, IOU_THRESHOLD);
    }
    // SortTracker object_tracker(int max_age = 5, int min_hits = 3, double iouThreshold = 0.3);
    int frame_count = 0;
    double fps = -1;
    int total_frames = 0;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Detection>> output;

    while (true)
    {
        webcam.next(frames);
        output.clear();
        std::cout << "frame size" << frames.size() << std::endl;
        output.resize(frames.size());
        yolov5.detect(frames, output);
        for (int i = 0; i < cam_num; ++i)
        {
            if (output[i].size() > 0)
            {
                output[i] = obj_tracker[i].update(output[i]);
            }
            else
            {
                std::cout << "output[i].size() <= 0" << std::endl;
            }
        }
        frame_count++;
        total_frames++;
        int detections = output.size();
        std::cout << "detections: " << detections << std::endl;
        for (int i = 0; i < detections; ++i)
        {
            auto detection = output[i];
            for (int j = 0; j < detection.size(); j++)
            {
                auto box = detection[j].box;
                auto classId = detection[j].class_id;

                const auto color = yolov5.colors[classId % yolov5.colors.size()];
                cv::rectangle(frames[i], box, color, 3);
                // std::cout<< "box.y - 20: "<< box.y - 20 << "\nbox.y - 5: " << box.y - 5 << std::endl;
                cv::rectangle(frames[i], cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                // cv::putText(frames[i], yolov5.class_name[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                cv::putText(frames[i], std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }

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
