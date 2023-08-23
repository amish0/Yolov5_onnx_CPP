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

struct cam_info_t
{
    char name[20];
    char location[20];
    char type[20];
    char url[100];
    int id;
    int width;
    int height;

};

#endif // CONFIG_H