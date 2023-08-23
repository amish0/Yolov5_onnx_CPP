#include "display_frame.h"

DisplayFrame::DisplayFrame()
{
    no_of_cameras = configparameter.cam_num_;
    display_width = 1920;
    display_height = 1080;
    if (no_of_cameras == 1)
    {
        rows = 1;
        cols = 1;
    }
    else if (no_of_cameras == 2)
    {
        rows = 1;
        cols = 2;
    }
    else if ((no_of_cameras == 3) || (no_of_cameras == 4))
    {
        rows = 2;
        cols = 2;
    }
    else
        exit(1);
    // if (SYSTEM_TYPE == "Windows")
    // {
    //     display_width = GetSystemMetrics(SM_CXSCREEN);
    //     display_height = GetSystemMetrics(SM_CYSCREEN);
    // }
    // else if (SYSTEM_TYPE == "Linux")
    // {
    //     // Display* disp = XOpenDisplay(NULL);
    //     // Screen*  scrn = DefaultScreenOfDisplay(disp);
    //     // display_width = scrn->width;
    //     // display_height = scrn->height;
    //     display_width = 1920;
    //     display_height = 1080;
    // }
    // else
    // {
    //     display_width = 1920;
    //     display_height = 1080;
    // }

    // switch (no_of_cameras)
    // {
    // case 1:
    //     rows = 1;
    //     cols = 1;
    //     break;
    // case 2:
    //     rows = 1;
    //     cols = 2;
    //     break;
    // case 3:
    // case 4:
    //     rows = 2;
    //     cols = 2;
    //     break;
    // default:
    //     exit(1);
    //     break;
    // }
    width = int(display_width / cols);
    height = int(display_height / rows);
    display_frame = cv::Mat(display_height, display_width, CV_8UC3, cv::Scalar(0, 0, 0));
}

DisplayFrame::~DisplayFrame()
{
}

void DisplayFrame::update(std::vector<cv::Mat> r_frames)
{
    frames.push(r_frames);
}

void DisplayFrame::process()
{
    if (frames.size() > 0)
    {
        std::vector<cv::Mat> frame;
        frame = frames.front(); // get frame from queue
        add_camera_info(frame); // add camera info to frame
        frames.pop(); // remove frame from queue
        for (int i = 0; i < no_of_cameras; i++)
        {
            cv::Mat temp; // create temp mat
            cv::resize(frame[i], temp, cv::Size(width, height)); // resize frame
            int row = int(i / cols); // get row
            int col = i % cols; // get col
            temp.copyTo(display_frame(cv::Rect(col * width, row * height, width, height))); // copy frame to display frame
        }
    }
}

void DisplayFrame::add_camera_info(std::vector<cv::Mat> &r_frames)
{
    for (int i = 0; i < no_of_cameras; i++)
    {
        cv::putText(r_frames[i], configparameter.cam_list_[i].name, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::putText(r_frames[i], configparameter.cam_list_[i].location, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        if (std::strcmp(configparameter.cam_list_[i].type, "IP") == 0)
        {
            cv::putText(r_frames[i], configparameter.cam_list_[i].url, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        else if (std::strcmp(configparameter.cam_list_[i].type, "Webcam") == 0)
        {
            cv::putText(r_frames[i], std::to_string(configparameter.cam_list_[i].id), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        else
        {
            cv::putText(r_frames[i], "Unknown", cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

    }
}


void DisplayFrame::display()
{
    if (frames.size() > 0)
    {
        process();
        cv::imshow(display_title, display_frame);
        cv::waitKey(1);
    }
}

void DisplayFrame::start()
{
    isRunning = true;
    t1 = std::thread(&DisplayFrame::display, this);
}

void DisplayFrame::stop()
{
    isRunning = false;
    t1.join();
}
