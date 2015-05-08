#include "function.h"

//compute centers of each region
void computePointsCenter(const vector<Point2f>& allPts, Point2f& center);
void computeCenters(const vector<vector<Point2f>>& ptsTab, vector<Point2f>& centers);

//compute perspective transform matrix of each region
//isMatched: 0 - unmatched;  1 - translation;  2 - perspective transform
void computeTransforms(const vector<vector<Point2f>>& srcFeaturesTab, const vector<vector<Point2f>>& refFeaturesTab, vector<Mat>& transforms, vector<int>& isMatched);

//deal with unmatched region
void dealUnMatchedRegion(const Mat& srcImg, const vector<vector<Point2f>>& srcPointsTab, const vector<vector<Point2f>>& srcFeaturesTab, const vector<int>& srcLabels, vector<int>& isMatched, vector<Mat>& transforms );
void dealUnMatchedRegion(vector<int>& isMatched, vector<Mat>& transforms);


//project each region in source
void applyTransforms(const vector<vector<Point2f>>& srcPointsTab, const vector<Mat>& transforms, const vector<int>& isMatched, vector<vector<Point2f>>& refPointsTab);


