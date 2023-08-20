#include "sort_tracker.h"

// SortTracker::SortTracker(int max_age, int min_hits, double iouThreshold)
// {
//     this->max_age = max_age;
//     this->min_hits = min_hits;
//     this->iouThreshold = iouThreshold;
//     std::cout << "SortTracker constructor called" << std::endl;
// }

SortTracker::SortTracker()
{
    std::cout << "SortTracker constructor called" << std::endl;
}

void SortTracker::init(int max_age, int min_hits, double iouThreshold)
{
    this->max_age = max_age;
    this->min_hits = min_hits;
    this->iouThreshold = iouThreshold;
    std::cout << "SortTracker init called" << std::endl;
}

SortTracker::~SortTracker()
{
    std::cout << "SortTracker destructor called" << std::endl;
}

double SortTracker::get_iou(cv::Rect bb_test, cv::Rect bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

vector<Detection> SortTracker::update(vector<Detection> boxes)
{
    // Initialize Tracker
    frame_count += 1; // frame_count += 1
    if (trackers.size() == 0)
    {
        // If no trackers yet, then every observation is new and so we add each observation to trackers
        for (int i = 0; i < boxes.size(); i++)
        {
            // initialize kalman trackers using first detections.
            KalmanTracker trk = KalmanTracker(boxes[i].box);
            class_id.push_back(boxes[i].class_id);
            trackers.push_back(trk);
        }
        std::cout << "trackers.size() == 0" << std::endl;
        std::cout << "initialize kalman trackers using first detections." << std::endl;
    }

    // 1. Predict
    // std::cout << "1. Predict" << std::endl;
    predictedBoxes.clear();
    int counter_trackers = 0; // added by me
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        cv::Rect pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            it++;
            counter_trackers += 1; // added by me
        }
        else
        {
            it = trackers.erase(it);
            class_id.erase(class_id.begin() + counter_trackers); // added by me

            // std::cout << "Box invalid at frame " << boxes[0].frame << std::endl;
        }
    }
    // std::cout << "Prediction ended" << std::endl;

    // 2. associate detections to tracked object (both represented as bounding boxes)
    // std::cout << "2. associate detections to tracked object (both represented as bounding boxes)" << std::endl;
    trkNum = predictedBoxes.size();
    detNum = boxes.size();
    // std::cout << "trkNum: " << trkNum << std::endl;
    // std::cout << "detNum: " << detNum << std::endl;
    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));
    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - get_iou(predictedBoxes[i], boxes[j].box);
        }
    }

    // std::cout << "association middle" << std::endl;
    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    // std::cout << "association middle 1" << std::endl;
    assignment.clear();
    // std::cout << "association middle 1.1" << std::endl;
    // std::cout << "iouMatrix.size(): " << iouMatrix.size() << std::endl;
    // std::cout << "iouMatrix[0].size(): " << iouMatrix[0].size() << std::endl;
    // std::cout << "assignment.size(): " << assignment.size() << std::endl;
    HungAlgo.Solve(iouMatrix, assignment);
    // std::cout << "association middle 2" << std::endl;
    // find matches, unmatched_detections and unmatched_predictions
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();
    if (detNum > trkNum) //	there are unmatched detections
    {
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }
    else
    {
        // TO DO
    }

    // std::cout << "association end" << std::endl;
    // filter out matched with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }
    ///////////////////////////////////////
    // 3. updating trackers
    ///////////////////////////////////////
    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(boxes[detIdx].box);
        class_id[trkIdx]=boxes[detIdx].class_id;
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        KalmanTracker tracker = KalmanTracker(boxes[umd].box);
        trackers.push_back(tracker);
        class_id.push_back(boxes[umd].class_id);
    }

    // get trackers' output
    frameTrackingResult.clear();
    counter_trackers = 0; // added by me
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            Detection res;
            res.box = (*it).get_state();
            res.class_id = class_id[counter_trackers]; // added by me
            res.track_id=(*it).m_id + 1;
            res.confidence = 1;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;
        counter_trackers += 1; // added by me
    }
    return frameTrackingResult;
}
