#ifndef CONFIG_H
#define CONFIG_H
#include <iostream>
#include <opencv2/opencv.hpp>

#define CONFIG_FILE "config.ini"
#define MAX_AGES 180
#define MIN_HITS 5
#define IOU_THRESHOLD 0.3

struct Detection
{
    int class_id;
    int track_id;
    float confidence;
    cv::Rect box;
};

#endif // CONFIG_H