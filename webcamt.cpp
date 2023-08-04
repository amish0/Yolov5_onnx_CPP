#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main()
{
    VideoCapture cap(0);
    Mat frame;
    if (cap.isOpened() != 0)
    {
        while (cap.read(frame))
        {
            namedWindow("Webcam Running", WINDOW_AUTOSIZE);
            imshow("Webcam Running", frame);
            if (waitKey(100) == 27)
                break;
        }
    }
    cap.release();
    destroyAllWindows();
    return 0;
}