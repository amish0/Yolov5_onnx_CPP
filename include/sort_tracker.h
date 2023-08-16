#ifndef SORTTRACKER_H
#define SORTTRACKER_H

#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/tracking.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

// Uncomment the following line to enable tracking output to console
typedef struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;
} TrackingBox;

class SortTracker
{
public:
    SortTracker(int max_age, int min_hits, double iouThreshold);
    ~SortTracker();
    double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);
	vector<TrackingBox> Update(vector<TrackingBox> boxes);


private:
    int max_age;
	int min_hits;
	double iouThreshold;
    vector<KalmanTracker> trackers;
    vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;

};

#endif // SORTTRACKER_H