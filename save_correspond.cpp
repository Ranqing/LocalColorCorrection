#include "correspond.h"

//保存区域对应性结果: 将求解得到的ref的区域对应再反投影回来查看结果
void BackProjectToSource(const vector<int>& refLabels, const Mat& srcImg, const Mat& srcVis, const int& regionum, const Mat& refDisp, Mat& srcBackSegments)
{
	int width = refDisp.size().width;
	int height = refDisp.size().height;
	int sz = width * height;

	vector<int> srcBackLabels(sz, -1);
	vector<vector<Point2f>> srcBackPointsTab(regionum);
	for (int i = 0; i < sz; ++i)
	{
		int x = i % width;
		int y = i / width;
		int d = refDisp.at<uchar>(y,x);
		int l = refLabels[i];
		if (l == -1)
		{
			continue;
		}

		int sx = x - d;
		int sy = y;
		if (sx >= 0 && sx < width && sy >= 0 && sy < height )
		{
			if (srcVis.at<uchar>(sy, sx) != 255)
			{
				srcBackLabels[sy*width+sx] = l;
				srcBackPointsTab[l].push_back(Point2f(sx, sy));	
			}					
		}
	}

	for (int i = 0; i < regionum; ++i)
	{
		vector<Point2f>& ppts = srcBackPointsTab[i];
		if (ppts.size() == 0)
		{
			continue;
		}

		Mat msk = Mat::zeros(height, width, CV_8UC1);
		pointsToMask(ppts, msk);

		//形态学操作： 
		//一次腐蚀
		Mat element = getStructuringElement(MORPH_RECT, cv::Size(3, 3));
		Mat out;
		cv::erode(msk, out, element);
		
		//两次膨胀
		element = getStructuringElement(MORPH_RECT, cv::Size(3,3));
		cv::dilate(out, msk, element);
		cv::dilate(msk, out, element);
		msk = out.clone();

		//mask转换为区域标签
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				if (msk.at<uchar>(y,x) == 255)
				{
					srcBackLabels[y*width+x] = i;
				}
			}
		}
	}
	
	/*Mat tsrcSegments;
	calSegments(srcImg, regionum, srcBackLabels, tsrcSegments);
	calContours(tsrcSegments, tsrcSegments, srcBackSegments);*/
	calSegments(srcImg, regionum, srcBackLabels, srcBackSegments);
}



void Points2Mask(vector<Point2f> pts, Mat& msk)
{
	int sz = pts.size();
	if (sz == 0)
		return;
	for (int i = 0; i < sz; ++i)
	{
		int x = pts[i].x;
		int y = pts[i].y;

		msk.at<uchar>(y,x) = 255;
	}
}

//prefix: "vis_" - 考虑可见性
void saveRegions(Mat srcImg, Mat refImg, vector<vector<Point2f>> srcPointsTab, vector<vector<Point2f>> refPointsTab, string outfolder, string prefix)
{
	int regionum = srcPointsTab.size();
	int width  = srcImg.size().width; 
	int height = srcImg.size().height;

	for (int i = 0; i < regionum; ++i)
	{
		vector<Point2f>& srcPts = srcPointsTab[i];
		vector<Point2f>& refPts = refPointsTab[i];

		Mat srcMsk = Mat::zeros(height, width, CV_8UC1);
		Points2Mask(srcPts, srcMsk);
		Mat refMsk = Mat::zeros(height, width, CV_8UC1);
		Points2Mask(refPts, refMsk);

		Mat srcRegion , refRegion;

		srcImg.copyTo(srcRegion, srcMsk);
		refImg.copyTo(refRegion, refMsk);

		string fn;
		fn = outfolder + "/" + prefix + "src_region_" + type2string(i) + ".png";
		imwrite(fn, srcRegion);
		fn = outfolder + "/" + prefix + "ref_region_" + type2string(i) + ".png";
		imwrite(fn, refRegion);

		string mskfn;
		fn = outfolder + "/" + prefix + "src_mask_" + type2string(i) + ".png";
		imwrite(fn, srcMsk);
	}

}

void saveRegionCorrespondence(Mat refImg, Mat srcImg, Mat refVis, Mat srcVis, vector<int> refLabels, vector<int> srcLabels, int regionum, 
	string out_labelsFn, string out_segmentsFn, string out_samecolorFn, string regionFolder /*= ""*/)
{
	int width = refImg.size().width;
	int height = refImg.size().height;

	saveLabels(out_labelsFn, width, height, refLabels);
	
	Mat refSegments; 
	Mat refSegContours; //segments + contours;
	
	calSegments(refImg, regionum, refLabels, refSegments);
	calContours(refSegments, refSegments, refSegContours);
	imwrite(out_segmentsFn, refSegContours);

	vector<vector<Point2f>> srcPointsTab(regionum);
	vector<vector<Point2f>> refPointsTab(regionum);
	calPointsTab(srcLabels, width, height, srcPointsTab);
	calPointsTab(refLabels, width, height, refPointsTab);

	vector<vector<Point2f>> srcVisPointsTab(regionum);
	vector<vector<Point2f>> refVisPointsTab(regionum);
	calPointsTab(srcLabels, srcVis, srcVisPointsTab);
	calPointsTab(refLabels, refVis, refVisPointsTab);

	cout << "region folder = " <<  regionFolder << endl;
	if (regionFolder != "")
	{
		cout << "save regions correspondence in " << regionFolder << endl;		
		saveRegions(srcImg, refImg, srcPointsTab, refPointsTab, regionFolder, "");
		
		cout << "save regions correspondence with visibility in " << regionFolder << endl;
		saveRegions(srcImg, refImg, srcVisPointsTab, refVisPointsTab, regionFolder, "vis_");
	}
	
	//保存相同颜色的区域结果
	Mat srcSegments;
	vector<Scalar> srcColorTab;
	calSegments(srcImg, srcPointsTab, srcSegments, srcColorTab); //计算得到每个区域的均值，并且颜色值放在srcColorTab
	
	//same color segments 
	Mat samecolor_refSegments = Mat::zeros(height, width, CV_8UC3); 
	for (int i = 0; i < regionum; ++i)
	{
		vector<Point2f>& refPts = refPointsTab[i];
		if (refPts.size()== 0)
			continue;
		for (int j = 0; j < refPts.size(); ++j )
		{
			int x = refPts[j].x;
			int y = refPts[j].y;
			samecolor_refSegments.at<Vec3b>(y,x) = Vec3b(srcColorTab[i].val[0],srcColorTab[i].val[1], srcColorTab[i].val[2]);
		}
	}
	Mat samecolor_refContours = Mat::zeros(height, width, CV_8UC3);
	calContours(samecolor_refSegments, samecolor_refSegments, samecolor_refContours);
	imwrite(out_samecolorFn, samecolor_refContours);
}