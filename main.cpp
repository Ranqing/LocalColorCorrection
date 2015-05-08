#include "function.h"

int main(int argc, char * argv[])
{
	if (argc != 4)
	{
		cout << "Usage: " << endl;
		cout << "FeaturesColorCorrection.exe folder refFn srcFn" << endl;
	}

	string folder = string(argv[1]);
	string refFn  = string(argv[2]);
	string srcFn  = string(argv[3]);

	string refImFn = folder + "/" + refFn + ".png";
	string srcImFn = folder + "/" + srcFn + ".png";
	Mat refImg, srcImg;
	readImage(refImFn, refImg);
	readImage(srcImFn, srcImg);

	int width = refImg.size().width;
	int height = refImg.size().height;

	string labelsFn = folder + "/labels_disp5.txt";
	vector<int> labels;
	int regionum;
	readLabels(labelsFn, labels, regionum);
	cout << regionum << " regions." << endl;

	string matchFn = folder + "/matches_view5E2.txt";
	vector<Point2i> refSiftPts, srcSiftPts;
	vector<vector<bool>> isTaken;
	isTaken.resize(height);
	for (int i = 0; i < height; ++i)
	{
		isTaken[i].resize(width, false);
	}
	readMatches(matchFn, refSiftPts, srcSiftPts, isTaken);
	cout << refSiftPts.size() << " features." << endl;
	string srcDispFn = folder + "/disp5.png";
	OutlierRemoval(srcSiftPts, refSiftPts, srcDispFn);

	/*cout << endl << "more features." << endl;
	string featuresFn = folder + "/features_view5E2.txt";
	vector<Point2i> more_srcSiftPts, more_refSiftPts;
	readFeatures(featuresFn, more_srcSiftPts);
	more_refSiftPts.resize(more_srcSiftPts.size());

	dispFn = folder + "/disp5.png";
	OutlierRemoval(more_srcSiftPts, more_refSiftPts, dispFn);
	matchFn = folder + "/Removal_matches_view5E2.txt";
	saveMatches(matchFn, more_srcSiftPts, more_refSiftPts);
	return 1 ;*/
	
	//考虑可见性
	Mat srcVis, refVis;
	readImage(folder + "/visibility_view5.png", srcVis, CV_LOAD_IMAGE_GRAYSCALE);
	readImage(folder + "/visibility_view1.png", refVis, CV_LOAD_IMAGE_GRAYSCALE);
	
	vector<int> refLabels;
	FindRegionCorrespondence(srcImg, refImg, srcSiftPts, refSiftPts, srcVis, refVis, labels, regionum, refLabels);
	cout << "find region correspondence done." << endl;	

	string backSegFn = folder + "/back_segments_" + srcFn + ".png";
	string refDispFn = folder + "/disp1.png";
	BackProjectToSource(refLabels, srcImg, srcVis, regionum, refDispFn, backSegFn);
	
	//叠加valid_segments_view5E2.png 和 back_segments_view5E2.png
	string SegFn = folder + "/valid_segments_" + srcFn + ".png";
	string overFn = folder + "/overlay_contours_" + srcFn + ".png";
	overlayTwoImages(SegFn, backSegFn, 0.5, overFn);
	cout << "save " << overFn << " done." << endl;

#ifdef SAVE_REGIONS
	cout << "save region correspondence." << endl;
	string out_labelsFn = folder + "/labels_view1.txt";
	string out_segmentsFn = folder + "/segments_view1.png";
	string out_contoursFn = folder + "/contours_view1.png";
	string out_correspondFn = folder + "/correspond_segments_view1.png";
	string out_correspondConFn = folder + "/correspond_contours_segments_view1.png";
	saveRegionCorrespondence(refImg, srcImg, refVis, srcVis, refLabels, labels, regionum, out_labelsFn, out_segmentsFn, out_contoursFn, out_correspondFn, out_correspondConFn);
	cout << out_labelsFn << endl << out_segmentsFn << endl << out_contoursFn << endl << out_correspondFn << endl << out_correspondConFn << endl;
#endif
	
	Mat srcResultImg;
	LocalColorCorrection(srcImg, srcVis, labels, refImg, refVis, refLabels, regionum, srcResultImg );	
	string correctFn = folder + "/local_correct_" + srcFn + ".png";
	imwrite(correctFn, srcResultImg);
	cout << "save " << correctFn << endl;
	cout << endl << "local color correction done." << endl;
}