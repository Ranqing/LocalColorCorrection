#include "function.h"

#define __DEBUG

int main(int argc, char * argv[])
{
	if (argc != 5)
	{
		cout << "Usage: " << endl;
		cout << "LocalColorCorrection.exe sceneName folder refFn srcFn" << endl;
		return -1;
	}

	string scene  = string(argv[1]);
	string folder = string(argv[2]);
	string refFn  = string(argv[3]);
	string srcFn  = string(argv[4]);

	//For Debug
	string resultFolder = "../Results/" + scene;
	string fn ;

	string refImFn = folder + "/" + refFn + ".png";
	string srcImFn = folder + "/" + srcFn + ".png";
	Mat refImg, srcImg;
	readImage(refImFn, refImg);
	readImage(srcImFn, srcImg);
	cout << "read input images done." << endl;

	string labelsFn = folder + "/labels_disp5.txt";
	vector<int> labels;
	int regionum;
	readLabels(labelsFn, labels, regionum);
	cout << "read segmentation labels done. " << regionum << " regions." << endl;

#ifdef __DEBUG
	Mat d_srcSegments, d_srcContours;
	calSegments(srcImg, regionum, labels,d_srcSegments);
	calContours(d_srcSegments, d_srcSegments, d_srcContours);
	fn = resultFolder + "/segments_" + srcFn + ".png";
	imwrite(fn, d_srcContours);
#endif
		
	int width = refImg.size().width;
	int height = refImg.size().height;	

	string matchFn = folder + "/matches_" + srcFn + ".txt";
	vector<Point2i> refSiftPts, srcSiftPts;
	vector<vector<bool>> isTaken;
	isTaken.resize(height);
	for (int i = 0; i < height; ++i)
	{
		isTaken[i].resize(width, false);
	}
	readMatches(matchFn, refSiftPts, srcSiftPts, isTaken);
	cout << "read matches done. " << refSiftPts.size() << " matches." << endl;
	
	string srcDispFn = folder + "/disp5.png";
	string refDispFn = folder + "/disp1.png";
	Mat srcDispImg, refDispImg;
	readImage(srcDispFn, srcDispImg, CV_LOAD_IMAGE_GRAYSCALE);
	readImage(refDispFn, refDispImg, CV_LOAD_IMAGE_GRAYSCALE);

	OutlierRemoval(srcSiftPts, refSiftPts, srcDispImg);

#ifdef __DEBUG
	//保存features在srcImg+contours中的分布
	Mat d_srcFeatures;
	calContours(d_srcSegments, srcImg, d_srcContours, cv::Scalar(0,0,0));
	showFeatures(srcSiftPts, d_srcContours, d_srcFeatures);
	fn = resultFolder + "/features_" + srcFn + ".png";
	imwrite(fn, d_srcFeatures);
#endif
		
	Mat srcVis, refVis;
	readImage(folder + "/visibility_view5.png", srcVis, CV_LOAD_IMAGE_GRAYSCALE);
	readImage(folder + "/visibility_view1.png", refVis, CV_LOAD_IMAGE_GRAYSCALE);
	cout << "read visibility prior done." << endl;

	Mat srcValidMsk = Mat(height, width, CV_8UC1, cv::Scalar(255)) - srcVis;
	Mat refValidMsk = Mat(height, width, CV_8UC1, cv::Scalar(255)) - refVis;

#ifdef __DEBUG
	//保存srcValidMsk, refValidMsk
	fn = folder + "/valid_mask_view5.png";
	imwrite(fn, srcValidMsk);
	fn = folder + "/valid_mask_view1.png";
	imwrite(fn, refValidMsk);

	//保存valid_ref.png, valid_src.png, valid_src_segments.png
	Mat d_validRefImg, d_validSrcImg, d_validSrcSegments;
	refImg.copyTo(d_validRefImg, refValidMsk);
	srcImg.copyTo(d_validSrcImg, srcValidMsk);
	d_srcSegments.copyTo(d_validSrcSegments, srcValidMsk);
	fn = resultFolder + "/valid_" + srcFn + ".png";
	imwrite(fn, d_validSrcImg);
	fn = resultFolder + "/valid_" + refFn + ".png";
	imwrite(fn, d_validRefImg);
	fn = resultFolder + "/valid_segments_" + srcFn + ".png";
	imwrite(fn, d_validSrcSegments);
#endif
	
	vector<int> refLabels;
	FindRegionCorrespondence(srcImg, refImg, srcSiftPts, refSiftPts, srcVis, refVis, labels, regionum, refLabels, resultFolder);  //resultFolder用于保存中间结果
	cout << "find region correspondence done." << endl;	

#ifdef __DEBUG
	//保存对应性结果
	//Ref对应性反投影到Src -> back_segments_src.png
	Mat d_srcBackSegments;
	BackProjectToSource(refLabels, srcImg, srcVis, regionum, refDispImg, d_srcBackSegments);
	fn = resultFolder + "/back_segments_" + srcFn + ".png";
	imwrite(fn, d_srcBackSegments);

	//叠加valid_segments_view5E2.png 和 back_segments_view5E2.png 的轮廓 -> overlay_contours_src.png
	string SegFn = resultFolder + "/valid_segments_" + srcFn + ".png";
	string backSegFn = resultFolder + "/back_segments_" + srcFn + ".png";
	string overFn = resultFolder + "/overlay_contours.png";
	overlayTwoImages(SegFn, backSegFn, 0.5, overFn);
	cout << "save " << overFn << " done." << endl;
	//结束保存对应性结果
#endif
	
#ifdef __DEBUG
	//保存src-ref的对应区域，以及估计得到ref的分割
	cout << "save region correspondence." << endl;
	string out_labelsFn = resultFolder + "/estimate_labels_view1.txt";	
	string out_segmentsFn = resultFolder + "/estimate_segments_view1.png";  
	string out_samecolorFn = resultFolder + "/samecolor_segments_view1.png";
	saveRegionCorrespondence(refImg, srcImg, refVis, srcVis, refLabels, labels, regionum, out_labelsFn, out_segmentsFn, out_samecolorFn, resultFolder+"/regions");
	cout << out_labelsFn << endl << out_segmentsFn << endl << out_samecolorFn << endl ;
#endif
	
	//计算color weighted map
	/*cout << endl << "compute color weighted maps. " << endl;
	string srcSegFn = folder + "/segments_" + srcFn + ".png";
	Mat srcSegments;
	readImage(srcSegFn, srcSegments);
	testWeightedLocalColorCorrection(srcImg, srcSegments, srcVis, labels, regionum);
	cout << endl << "compute color weighted maps done." << endl;
	return 1;*/
	
	Mat srcResult;
	LocalColorCorrection(srcImg, srcVis, labels, refImg, refVis, refLabels, regionum, srcResult );	
	string correctFn = folder + "/noweighted_" + srcFn + "_RE.png";
	imwrite(correctFn, srcResult);
	cout << "save " << correctFn << endl;
	cout << endl << "local color correction done." << endl;

	Mat srcWeightedResult;
	Mat srcVisibleResult;
	WeightedLocalColorCorrection(srcImg, srcVis, labels, refImg, refVis, refLabels, regionum, srcVisibleResult, srcWeightedResult, resultFolder );
	correctFn = folder + "/" + srcFn + "_RE.png";
	imwrite(correctFn, srcVisibleResult);
	/*//all weighted
	correctFn = folder + "/" + srcFn + "_RE.png";
	imwrite(correctFn, srcWeightedResult);
	cout << "save " << correctFn << endl;*/
	cout << endl  << "local weighted color correction done." << endl;
}