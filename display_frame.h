#ifndef DISPLAY_FRAME_H
#define DISPLAY_FRAME_H

#include <iostream>

struct cam_info_t
{
    char name[20];
    char location[20];
};


class DisplayFrame
{
    private:
        int no_of_cameras;
        int rows;
        int cols;
    public:
        DisplayFrame();
        ~DisplayFrame();
        void displayFrame();

};
#endif // DISPLAY_FRAME_H