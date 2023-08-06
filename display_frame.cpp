#include "display_frame.h"

// Path: display_frame.cpp
// Compare this snippet from display_frame.h:
// #include "display_frame.h"
//

DisplayFrame::DisplayFrame()
{
    no_of_cameras = 0;
    rows = 0;
    cols = 0;
}

DisplayFrame::~DisplayFrame()
{
    //dtor
}

void DisplayFrame::displayFrame()
{
    std::cout << "Displaying frame" << std::endl;
}