#ifndef DISPLAY_FRAME_H
#define DISPLAY_FRAME_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include "config_reader.h"
#include "config.h"

#ifdef _WIN32
    #include <Windows.h>
    #define SYSTEM_TYPE "Windows"
#elif __linux__
    #include <X11/Xlib.h>
    #define SYSTEM_TYPE "Linux"
#else
    #define SYSTEM_TYPE "Unknown"
#endif


class DisplayFrame
{
    private:
        int rows;
        int cols;
        int no_of_cameras;
        int width;
        int height;
        int display_width;
        int display_height;
        std::queue<std::vector<cv::Mat>> frames;
        cv::Mat display_frame;
        string display_title = "Multi-Camera Tracking";
    public:
        DisplayFrame();
        ~DisplayFrame();
        void update(std::vector<cv::Mat> r_frames);
        void display();
};
#endif // DISPLAY_FRAME_H