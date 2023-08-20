#include "display_frame.h"

DisplayFrame::DisplayFrame()
{
    no_of_cameras = configparameter.cam_num_;
    width = configparameter.width_;
    height = configparameter.height_;
    if (SYSTEM_TYPE == "Windows")
    {
        display_width = GetSystemMetrics(SM_CXSCREEN);
        display_height = GetSystemMetrics(SM_CYSCREEN);
    }
    else if (SYSTEM_TYPE == "Linux")
    {
        // Display* disp = XOpenDisplay(NULL);
        // Screen*  scrn = DefaultScreenOfDisplay(disp);
        // display_width = scrn->width;
        // display_height = scrn->height;
        display_width = 1920;
        display_height = 1080;
    }
    else
    {
        display_width = 1920;
        display_height = 1080;
    }

    switch (no_of_cameras)
    {
    case 1:
        rows = 1;
        cols = 1;
        break;
    case 2:
        rows = 1;
        cols = 2;
        break;
    case 3:
    case 4:
        rows = 2;
        cols = 2;
        break;
    default:
        exit(1);
        break;
    }
    width = display_width / cols;
    height = display_height / rows;
    display_height = height * rows;
    display_width = width * cols;
    display_frame = cv::Mat(display_height, display_width, CV_8UC3, cv::Scalar(0, 0, 0));
}

DisplayFrame::~DisplayFrame()
{
    
}

void DisplayFrame::update(std::vector<cv::Mat> r_frames)
{
    frames.push_back(r_frames);
}

void DisplayFrame::display()
{
    if (frames.size()>0)
    {
        std::vector<cv::Mat> frame;
        frame = frames.pop_back();
        

    }
}

