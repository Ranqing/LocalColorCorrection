#include "common.h"
#include "basic.h"

void readImage(string fn, Mat& img, int METHOD = CV_LOAD_IMAGE_UNCHANGED);
void readFeatures(string fn, vector<Point2i>& features);
void readMatches(string fn, vector<Point2i>& refFeatures, vector<Point2i>& srcFeatures, vector<vector<bool>>& istaken);
void readLabels(string fn, vector<int>& labels, int& regionum);

void overlayTwoImages(const string& fn1, const string& fn2, const double& alpha, string resultFn);

void saveMatches(string fn, vector<Point2i>& srcFeatures, vector<Point2i>& refFeatures);
void saveLabels(string fn, int width, int height, vector<int>& labels);
void calSegments(Mat img, int regionum, vector<int>& labels, Mat& segments);
void calSegments(Mat img, vector<vector<Point2f>>& pointsTab, Mat& segments, vector<Scalar>& colorsTab);
void calContours(Mat segments, Mat img, Mat& contours, cv::Scalar color = cv::Scalar(255, 255, 255));
void calContours(Mat segments, Mat& contours);


//Labels -> Points Table, take visibility into consideration
void calPointsTab(const vector<int>& labels, int width, int height, vector<vector<Point2f>>& pointsTab);
void calPointsTab(const vector<int>& labels, const Mat& visibility, vector<vector<Point2f>>& pointsTab);

//compute mean color of each region
void computeMeans(const Mat& img, const vector<vector<Point2f>>& pointsTab, vector<Scalar>& means);

void pointsToMask(const vector<Point2f>& points, Mat& mask);
void pointsToMask(const vector<Point2f>& points, const Mat& visibility, Mat& mask);

void OutlierRemoval(vector<Point2i>& srcFeatures, vector<Point2i>& refFeatures, string dispFn = NULL);

void FindRegionCorrespondence(const Mat& srcImg, const Mat& refImg, const vector<Point2i>& srcFeatures, const vector<Point2i>& refFeatures, const Mat& srcVisibility, const Mat& refVisibility,
	const vector<int>& srcLabels, const int& regionum, vector<int>& refLabels);

void BackProjectToSource(const vector<int>& refLabels, const Mat& srcImg, const Mat& srcVis, const int& regionum, const string& dispFn, string backFn);

void saveRegionCorrespondence(Mat refImg, Mat srcImg, Mat refVis, Mat srcVis, vector<int> refLabels, vector<int> srcLabels, int regionum, string out_labelsFn, string out_segmentsFn, string out_contoursFn, string out_correspondFn, string out_correspondConFn);

void LocalColorCorrection(const Mat& src, const Mat& srcVis, const vector<int>& srclabels, const Mat& ref, const Mat& refVis, const vector<int>& reflabels, const int& regionum, Mat& srcResult );	