#include "camloader.h"

void LoadWebcam::init()
{
    for (int i = 0; i < cam_no; i++)
    {
        VideoCapture capture;
        capture.open(configparameter.cam_list_[i].id);
        captures.push_back(capture);
        captures[i].set(CAP_PROP_FRAME_WIDTH, configparameter.cam_list_[i].width);
        captures[i].set(CAP_PROP_FRAME_HEIGHT, configparameter.cam_list_[i].height);
    }
}

void LoadWebcam::start()
{
    isRunning = true;
    cout << "Start thread" << endl;
    for (int i = 0; i < cam_no; i++)
    {
        cout << "thread started id " << i << endl;
        t1[i] = thread(&LoadWebcam::run, this, i);
    }
}

void LoadWebcam::stop()
{
    isRunning = false;
    cout << "Stop thread" << endl;
    for (int i = 0; i < cam_no; i++)
    {
        t1[i].join();
        captures[i].release();
        cout << "Stop thread " << i << endl;
    }
    captures.clear();
}

void LoadWebcam::run(int id)
{

    while (isRunning)
    {
        Mat frame;
        captures[id] >> frame;
        if (frame.empty())
        {
            cout << "Empty frame" << endl;
            captures[id].release();
            this_thread::sleep_for(chrono::milliseconds(100));
            captures[id].open(configparameter.cam_list_[id].id);
            captures[id].set(CAP_PROP_FRAME_WIDTH, configparameter.cam_list_[id].width);
            captures[id].set(CAP_PROP_FRAME_HEIGHT, configparameter.cam_list_[id].height);
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        else
        {
            frames[id] = frame.clone();
        }
        // this_thread::sleep_for(chrono::milliseconds(5));
    }
}

void LoadWebcam::next(vector<Mat> &frames_retrive)
{
    // for (int i = 0; i < cam_no; i++)
    // {
    //     // cout << "test" << endl;
    //     frames_retrive[i] = frames[i].clone();
    //     this_thread::sleep_for(chrono::milliseconds(0));
    // }
    frames_retrive = frames;
}