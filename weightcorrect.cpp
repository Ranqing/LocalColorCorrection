#include "function.h"

void computeWeightMap(const cv::Mat& img, const uchar mean, cv::Mat& weight)
{
	int width = img.size().width;
	int height = img.size().height;
	double maxw = 0.0;
	double sigma = 10;

	Mat dweight = Mat::zeros(height, width, CV_64FC1);

	for (int y = 0; y < height; ++y)
	{
		double * pweight = dweight.ptr<double>(y);
		const uchar * color = img.ptr<uchar>(y);

		for(int x = 0; x < width; ++x)
		{
			double delta = (color[x] - mean) * 1.0;
			pweight[x] = exp( -(delta * delta)/sigma );
			if (pweight[x] > maxw)
				maxw = pweight[x];
		}
	}

	cv::Mat tmp;
	dweight.convertTo(tmp, CV_64FC1, 1.0/maxw );
	weight = Mat::zeros(height, width, CV_8UC1);	
	tmp.convertTo(weight, CV_8UC1, 255);
}

void weightedCorrect(const vector<cv::Scalar>& srcMeans, const vector<cv::Scalar>& refMeans, const vector<double>& Lfactors, const vector<double>& Afactors, const vector<double>& Bfactors, const cv::Vec3b& rawcolor, cv::Vec3b& newcolor)
{
	double weightSumL = 0.0, weightSumA = 0.0, weightSumB = 0.0;
	double sumL = 0.0, sumA = 0.0, sumB = 0.0;
	double sigma = 1/5.0;

	int regionum = srcMeans.size();
	for (int i = 0; i < regionum; ++i)
	{
		double deltaL = rawcolor[0] - srcMeans[i].val[0];
		double deltaA = rawcolor[1] - srcMeans[i].val[1];
		double deltaB = rawcolor[2] - srcMeans[i].val[2];

		double weightL = exp(-deltaL*deltaL*sigma);
		double weightA = exp(-deltaA*deltaA*sigma);
		double weightB = exp(-deltaB*deltaB*sigma);

		weightSumL += weightL;
		weightSumA += weightA;
		weightSumB += weightB;

		sumL += ( weightL * (Lfactors[i] * deltaL + refMeans[i].val[0] ) );
		sumA += ( weightA * (Afactors[i] * deltaA + refMeans[i].val[1] ) );
		sumB += ( weightB * (Bfactors[i] * deltaB + refMeans[i].val[2] ) );		
	}

	if (weightSumL != 0.0)
		newcolor[0] = sumL / weightSumL;
	if (weightSumA != 0.0)
		newcolor[1] = sumA / weightSumA;
	if (weightSumB != 0.0)
		newcolor[2] = sumB / weightSumB;	
}

void WeightedLocalColorCorrection(const Mat& src, const Mat& srcVis, const vector<int>& srclabels, const Mat& ref, const Mat& refVis, const vector<int>& reflabels, const int& regionum, Mat& srcResult)
{
	int width = src.size().width;
	int height = src.size().height;

	vector<vector<Point2f>> srcPointsTab(regionum);
	vector<vector<Point2f>> refPointsTab(regionum);
	calPointsTab(srclabels, width, height, srcPointsTab );
	calPointsTab(reflabels, width, height, refPointsTab );

	//用于计算color correction function
	vector<vector<Point2f>> srcVisPointsTab(regionum);
	vector<vector<Point2f>> refVisPointsTab(regionum);
	calPointsTab(srclabels, srcVis, srcVisPointsTab);	
	calPointsTab(reflabels, refVis, refVisPointsTab);

	vector<bool> isMatched(regionum, true);
	for (int i = 0; i < regionum; ++i)
	{
		if (refVisPointsTab[i].size() == 0)
		{
			isMatched[i] = false;
		}
	}

	Mat srcLab, refLab;
	cv::cvtColor(src, srcLab, cv::COLOR_BGR2Lab);
	cv::cvtColor(ref, refLab, cv::COLOR_BGR2Lab);

	//Global Means, Stddevs
	cv::Scalar srcGLmeans, srcGLstddev;
	cv::Scalar refGLmeans, refGLstddev;
	cv::meanStdDev(srcLab, srcGLmeans, srcGLstddev);
	cv::meanStdDev(refLab, refGLmeans, refGLstddev);	

	cout << "src global: " << srcGLmeans << ' ' << srcGLstddev << endl;
	cout << "ref global: " << refGLmeans << ' ' << refGLstddev << endl;

	//Local Means, Stddevs
	vector<cv::Scalar> srcMeans(regionum), srcStddevs(regionum);
	vector<cv::Scalar> refMeans(regionum), refStddevs(regionum);
	computeMeanStddev(srcLab, srcVisPointsTab, srcMeans, srcStddevs );
	computeMeanStddev(refLab, refVisPointsTab, refMeans, refStddevs );

	for (int i = 0; i < regionum; ++i)
	{
		if (isMatched[i] == false)
		{
			srcMeans[i] = srcGLmeans;
			srcStddevs[i] = srcGLstddev;
			refMeans[i] = refGLmeans;
			refStddevs[i] = refGLstddev;
		}
	}
	
	vector<double> LFactors(regionum), AFactors(regionum), BFactors(regionum);
	for (int i = 0; i < regionum; ++i)
	{
		LFactors[i] = refMeans[i].val[0] / srcMeans[i].val[0];
		AFactors[i] = refMeans[i].val[1] / srcMeans[i].val[1];
		BFactors[i] = refMeans[i].val[2] / srcMeans[i].val[2];
	}


	//为每个像素求解新的颜色
	Mat srcResultLab = Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < regionum; ++i)
	{
		const vector<Point2f>& ppt = srcVisPointsTab[i];
		Mat msk = Mat::zeros(height, width, CV_8UC1);

		pointsToMask(ppt, msk);
		correct(srcLab, msk, srcMeans[i], srcStddevs[i], refMeans[i], refStddevs[i], srcResultLab);
	}

	cv::cvtColor(srcResultLab, srcResult, cv::COLOR_Lab2BGR);
	imshow("visible pixels correction", srcResult);
	waitKey(0);
	destroyWindow("visible pixels correction");

	//计算未匹配区域，遮挡区域等像素： 不可见区域最好能膨胀一下
	vector<Point2f> srcSpecPointsTab(0);
	for (int i = 0; i < regionum; ++i)
	{
		if (isMatched[i] == false)
		{
			// 未匹配区域的像素
			copy(srcPointsTab[i].begin(), srcPointsTab[i].end(), std::back_inserter(srcSpecPointsTab) );
		}
		else
		{
			// 匹配区域的不可见像素
			Mat msk = Mat::zeros(height, width, CV_8UC1);
			Mat srcRegionVis ;
			
			vector<Point2f>& ppt = srcPointsTab[i];
			vector<Point2f> sppt(0);

			pointsToMask(ppt, msk);
			srcVis.copyTo(srcRegionVis, msk);
			maskToPoints(srcRegionVis, sppt);

			copy(sppt.begin(), sppt.end(), std::back_inserter(srcSpecPointsTab) );
		}

		/*string fn = "output/weighted_correct_pixels_" + type2string(i) + ".png";
		Mat msk = Mat::zeros(height, width, CV_8UC1);
		pointsToMask(srcSpecPointsTab[i], msk);
		imwrite(fn, msk);*/
	}
	string fn = "output/weighted_correct_pixels.png" ;
	Mat msk = Mat::zeros(height, width, CV_8UC1);
	pointsToMask(srcSpecPointsTab, msk);
	imwrite(fn, msk);


	//处理specular points： weighted color correction, 权重函数！！！！！！！
	for (int j = 0; j < srcSpecPointsTab.size(); ++j)
	{
		Point2f pt = srcSpecPointsTab[j];
		cv::Vec3b color = srcLab.at<Vec3b>(pt.y, pt.x);
		cv::Vec3b newcolor;
		weightedCorrect(srcMeans, refMeans, LFactors, AFactors, BFactors, color, newcolor );

		srcResultLab.at<Vec3b>(pt.y, pt.x) = newcolor;
	}
	
	cv::cvtColor(srcResultLab, srcResult, cv::COLOR_Lab2BGR);
	imshow("all pixels correction", srcResult);
	waitKey(0);
	destroyWindow("all pixels correction");	

	cout << "end of weighted color correction." << endl;
}