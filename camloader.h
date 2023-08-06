#ifndef CAMLOADER_H
#define CAMLOADER_H
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "config_reader.h"

using namespace std;
using namespace cv;

class LoadWebcam
{
private:
    // pthread_t t1;
    thread *t1;
    bool isRunning = false;
    int whidth;
    int height;
    int cam_no;
    vector<Mat> frames;
    vector<VideoCapture> captures;

public:
    LoadWebcam(int whidth, int height, int cam_no)
    {
        this->whidth = whidth;
        this->height = height;
        this->cam_no = cam_no;
        t1 = new thread[cam_no];
        frames.assign(cam_no, Mat::zeros(height, whidth, CV_8UC3));
        cout << "frame size" << frames.size() << endl;
    }
    ~LoadWebcam()
    {
        cout << "Destructor called" << endl;
        delete[] t1;
    }
    void init();
    void start();
    void stop();
    void run(int id);
    void next(vector<Mat> &frames_retrive);
};

#endif // CAMLOADER_H