#ifndef SORTTRACKER_H
#define SORTTRACKER_H

#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "config.h"

using namespace std;
using namespace cv;

// Uncomment the following line to enable tracking output to console
// typedef struct Detection
// {
//     int frame;
//     int id;
//     Rect_<float> box;
// } Detection;

class SortTracker
{
public:
    SortTracker();
	// SortTracker(int max_age, int min_hits, double iouThreshold);
    ~SortTracker();
	void init(int max_age, int min_hits, double iouThreshold);
    double get_iou(cv::Rect bb_test, cv::Rect bb_gt);
	vector<Detection> update(vector<Detection> boxes);

private:
    int max_age;
	int min_hits;
	double iouThreshold;
	int frame_count = 0;
    vector<KalmanTracker> trackers;
	vector<int> class_id;
    vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<Detection> frameTrackingResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;
};

#endif // SORTTRACKER_H