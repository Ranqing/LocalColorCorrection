#include "function.h"

void computeRAList(const vector<int>& labels, const int& width, const int& height, vector<vector<bool>>& RALists)
{
	int regionum = RALists.size();
	int dx[8] = {-1,0,1,-1,1,-1,0,1};
	int dy[8] = {-1,-1,-1,0,0,1,1,1};

	for (int i = 0; i < regionum; ++i)
	{
		RALists[i].resize(regionum, false);
	}

	vector<vector<Point2f>> pointsTab(regionum);
	calPointsTab(labels, width, height, pointsTab );

	for (int i = 0; i < regionum; ++ i)
	{
		vector<Point2f>& ppts = pointsTab[i];
		int idx = i;
		for(int j = 0; j < ppts.size(); ++j)
		{
			int x  = ppts[j].x;
			int y  = ppts[j].y;
			for (int k = 0; k < 8; ++k)
			{
				int deltax = dx[k];
				int deltay = dy[k];
				int curx = x + deltax;
				int cury = y + deltay;

				if (curx < 0 || curx >= width || cury < 0 || cury >= height )
					continue;

				int newindex = labels[cury*width + curx];
				if (newindex == idx)
					continue;
				if (RALists[i][newindex] == true)
					continue;

				RALists[i][newindex] = true;
			}
		}
	}

	fstream fout("output/RAList.txt", ios::out);
	fout << regionum << endl;
	for (int i = 0; i < regionum; ++i)
	{
		fout << i << ": " ; 
		for (int j = 0; j < regionum; ++j)
		{
			if (RALists[i][j] == true )
				fout << j << ' ';					 
		}
		fout << endl;
	}
	fout.close();
}

void weightedCorrect(const int& idx, const vector<Point2f>& points, const vector<bool>& RAList, const vector<cv::Scalar>& srcMeans, const vector<cv::Scalar>& refMeans,
	const vector<cv::Scalar>& srcStddevs, const vector<cv::Scalar>& refStddevs, const vector<double>& LFactors, const vector<double>& AFactors, const vector<double>& BFactors, 
	const Mat& srcLab, Mat& srcResultLab )
{
	int regionum = srcMeans.size();

	for (int j = 0; j < points.size(); ++j)
	{
		int y = points[j].y;
		int x = points[j].x;

		Vec3b rawcolor = srcLab.at<Vec3b>(y,x);
		double weightSumL = 0.0, weightSumA = 0.0, weightSumB = 0.0;
		double sumL = 0.0, sumA = 0.0, sumB = 0.0;

		for(int i = 0; i < regionum; ++i)
		{
			if ( i != idx && RAList[i] != true)
				continue;

			double deltaL = rawcolor[0] - srcMeans[i].val[0];
			double deltaA = rawcolor[1] - srcMeans[i].val[1];
			double deltaB = rawcolor[2] - srcMeans[i].val[2];

			double sigmaL = srcStddevs[i].val[0];
			double sigmaA = srcStddevs[i].val[1];
			double sigmaB = srcStddevs[i].val[2];

			double weightL = exp(-deltaL*deltaL / (2*sigmaL*sigmaL) ) / (sqrt(2*_PI)*sigmaL);
			double weightA = exp(-deltaA*deltaA / (2*sigmaA*sigmaA) ) / (sqrt(2*_PI)*sigmaA);
			double weightB = exp(-deltaB*deltaB / (2*sigmaB*sigmaB) ) / (sqrt(2*_PI)*sigmaB);

			weightSumL += weightL;
			weightSumA += weightA;
			weightSumB += weightB;


			sumL += ( weightL * (LFactors[i] * deltaL + refMeans[i].val[0] ) );
			sumA += ( weightA * (AFactors[i] * deltaA + refMeans[i].val[1] ) );
			sumB += ( weightB * (BFactors[i] * deltaB + refMeans[i].val[2] ) );	
		}

		if (weightSumL != 0.0)
			srcResultLab.at<Vec3b>(y,x)[0] = sumL / weightSumL;
		if (weightSumA != 0.0)
			srcResultLab.at<Vec3b>(y,x)[1] = sumA / weightSumA;
		if (weightSumB != 0.0)
		    srcResultLab.at<Vec3b>(y,x)[2] = sumB / weightSumB;	
	}

}


void WeightedLocalColorCorrection(const Mat& src, const Mat& srcVis, const vector<int>& srclabels, const Mat& ref, const Mat& refVis, const vector<int>& reflabels, const int& regionum, Mat& srcVisibleResult, Mat& srcResult, string resultFolder)
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

	//处理无匹配区域
	for (int i = 0; i < regionum; ++i)
	{
		if (isMatched[i] == false || i == 25 || i == 22)
		{
			srcMeans[i] = srcGLmeans;
			srcStddevs[i] = srcGLstddev;
			refMeans[i] = refGLmeans;
			refStddevs[i] = refGLstddev;
		}
		if (i == 23)
		{
			srcMeans[i] = srcMeans[9];
			srcStddevs[i] = srcStddevs[9];
			refMeans[i] = refMeans[9];
			refStddevs[i] = refStddevs[9];
		}
		if (i == 15)
		{
			srcMeans[i] = srcMeans[5];
			srcStddevs[i] = srcMeans[5];
			refMeans[i] = refMeans[5];
			refStddevs[i] = refMeans[5];
		}
	}
	
	vector<double> LFactors(regionum), AFactors(regionum), BFactors(regionum);
	for (int i = 0; i < regionum; ++i)
	{
		LFactors[i] = refMeans[i].val[0] / srcMeans[i].val[0];
		AFactors[i] = refMeans[i].val[1] / srcMeans[i].val[1];
		BFactors[i] = refMeans[i].val[2] / srcMeans[i].val[2];
	}

	//为每个有匹配的区域用color transfer求解新的颜色
	Mat srcResultLab = Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < regionum; ++i)
	{
		const vector<Point2f>& ppt = srcPointsTab[i];
		Mat msk = Mat::zeros(height, width, CV_8UC1);

		pointsToMask(ppt, msk);
		correct(srcLab, msk, srcMeans[i], srcStddevs[i], refMeans[i], refStddevs[i], srcResultLab);
	}

	cv::cvtColor(srcResultLab, srcVisibleResult, cv::COLOR_Lab2BGR);
	imshow("visible pixels correction", srcVisibleResult);
	waitKey(0);
	destroyWindow("visible pixels correction");

	//计算未匹配区域，遮挡区域等像素： 不可见区域最好能膨胀一下
	vector<vector<Point2f>> srcSpecPointsTab(regionum);
	Mat srcWeightedMsk = Mat::zeros(height, width, CV_8UC1);
	for (int i = 0; i < regionum; ++i)
	{
		if (isMatched[i] == false)
		{
			// 未匹配区域的像素
			copy(srcPointsTab[i].begin(), srcPointsTab[i].end(), std::back_inserter(srcSpecPointsTab[i]) );
		}
		else
		{
			// 匹配区域的不可见像素
			/*Mat msk = Mat::zeros(height, width, CV_8UC1);
			Mat srcRegionVis ;

			vector<Point2f>& ppt = srcPointsTab[i];
			vector<Point2f> sppt(0);

			pointsToMask(ppt, msk);
			srcVis.copyTo(srcRegionVis, msk);
			maskToPoints(srcRegionVis, sppt);

			copy(sppt.begin(), sppt.end(), std::back_inserter(srcSpecPointsTab[i]) );*/
		}
		pointsToMask(srcSpecPointsTab[i], srcWeightedMsk);
	}
	string fn = resultFolder + "/weighted_correct_pixels.png";
	imwrite(fn, srcWeightedMsk);
	
	
	//计算区域邻接表, 用于计算权重
	vector<vector<bool>> RALists(regionum);
	computeRAList(srclabels, width, height, RALists);

	//为没匹配区域的像素使用weighted color transfer的方法计算新颜色：权重只考虑邻近区域
	for (int i = 0; i < regionum; ++ i)
	{
		if (srcSpecPointsTab[i].size() == false)
			continue;

		weightedCorrect(i, srcSpecPointsTab[i], RALists[i], srcMeans, refMeans, srcStddevs, refStddevs, LFactors, AFactors, BFactors, srcLab, srcResultLab );
	}
	cv::cvtColor(srcResultLab, srcResult, cv::COLOR_Lab2BGR);
	imshow("all pixels correction", srcResult);
	waitKey(0);
	destroyWindow("all pixels correction");	

	cout << "end of weighted color correction." << endl;	
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//计算color weighted map
//每个区域只影响那些相邻的区域
void computeWeightMap(const int& idx, const Mat& img, const float& mean, const float& stddev, const vector<vector<bool>>& RALists, const vector<int>& labels, Mat& weight)
{
	int width = img.size().width;
	int height = img.size().height;

	double maxw = 0.0;
	double sigma = stddev;
	
	Mat dweight = Mat::zeros(height, width, CV_64FC1);

	for (int y = 0; y < height; ++y)
	{
		for(int x = 0; x < width; ++x)
		{
			int l = labels[y * width + x];
			if (l != idx && RALists[l][idx] != true)
				continue;

			double delta = (img.at<uchar>(y,x) - mean) * 1.0;
			double w = exp( -(delta * delta)/(2*sigma*sigma ) ) / sqrt(2*_PI)*sigma;
			
			dweight.at<double>(y,x) = w;
			if (w > maxw) maxw = w;
		}
	}

	cv::Mat tmp;
	dweight.convertTo(tmp, CV_64FC1, 1.0/maxw );
	weight = Mat::zeros(height, width, CV_8UC1);	
	tmp.convertTo(weight, CV_8UC1, 255);
}

void computeWeightMap(const cv::Mat& img, const cv::Scalar& mean, const cv::Scalar& stddev, cv::Mat& weight)
{
	int width = img.size().width;
	int height = img.size().height;

	double maxw = 0.0f;
	double sigma = stddev.val[0] * stddev.val[0] + stddev.val[1] * stddev.val[1] + stddev.val[2] * stddev.val[2] ;
	 
	Mat dweight = Mat::zeros(height, width, CV_64FC1);
	for (int y = 0; y < height; ++ y)
	{
		for (int x = 0; x < width; ++x)
		{
			double delta = (img.at<Vec3b>(y,x)[0]-mean.val[0]) * (img.at<Vec3b>(y,x)[0]-mean.val[0]) +  (img.at<Vec3b>(y,x)[1]-mean.val[1]) * (img.at<Vec3b>(y,x)[1]-mean.val[1]) +
				(img.at<Vec3b>(y,x)[2]-mean.val[2]) * (img.at<Vec3b>(y,x)[2]-mean.val[2]);
			double w = exp(-delta/(2*sigma) ) / sqrt(2*_PI*sigma);
			dweight.at<double>(y,x) = w;

			 if (w > maxw)
				 maxw = w;
		}
	}

	cv::Mat tmp;
	dweight.convertTo(tmp, CV_64FC1, 1.0/maxw );
	weight = Mat::zeros(height, width, CV_8UC1);
	tmp.convertTo(weight, CV_8UC1, 255);
}

void testWeightedLocalColorCorrection(const Mat& src, const Mat& srcSegments, const Mat& srcVis, const vector<int>& srcLabels, const int& regionum )
{
	int width = src.size().width;
	int height = src.size().height;

	//建立区域邻接表:  第i行为第i区域的Region Adjacent list
	vector<vector<bool>> RALists(regionum);
	computeRAList(srcLabels, width, height, RALists);
	
	//用于计算 color correction function
	vector<vector<Point2f>> srcVisPointsTab(regionum);
	calPointsTab(srcLabels, srcVis, srcVisPointsTab);

	Mat srcLab;
	vector<Mat> labMats(3);
	cv::cvtColor(src, srcLab, cv::COLOR_BGR2Lab);
	cv::split(srcLab, labMats);
	
	vector<cv::Scalar> srcMeans(regionum), srcStddevs(regionum);
	computeMeanStddev(srcLab, srcVisPointsTab, srcMeans, srcStddevs );
		
	for (int i = 0; i < regionum; ++i)
	{
		cv::Mat weightL ;
		cv::Mat weightA ;
		cv::Mat weightB ;
		
		computeWeightMap(i, labMats[0], srcMeans[i].val[0], srcStddevs[i].val[0], RALists, srcLabels, weightL);
		computeWeightMap(i, labMats[1], srcMeans[i].val[1], srcStddevs[i].val[1], RALists, srcLabels, weightA);
		computeWeightMap(i, labMats[2], srcMeans[i].val[2], srcStddevs[i].val[2], RALists, srcLabels, weightB);
		
		string lwfn = "output/weight/weightL_" + type2string<int>(i) + ".png";
		string awfn = "output/weight/weightA_" + type2string<int>(i) + ".png";
		string bwfn = "output/weight/weightB_" + type2string<int>(i) + ".png";

		imwrite(lwfn, weightL);
		imwrite(awfn, weightA);
		imwrite(bwfn, weightB);

		/*cv::Mat weight;
		computeWeightMap(srcLab, srcMeans[i], srcStddevs[i], weight);	

		string wfn = "output/weight/weight_" + type2string<int>(i)+".png";
		imwrite(wfn, weight);*/
	}	
}


//void weightedCorrect(const vector<cv::Scalar>& srcMeans, const vector<cv::Scalar>& refMeans, const vector<cv::Scalar>& srcStddevs, const vector<cv::Scalar>& refStddevs, const vector<double>& Lfactors, const vector<double>& Afactors, const vector<double>& Bfactors, const cv::Vec3b& rawcolor, cv::Vec3b& newcolor)
//{
//	double weightSumL = 0.0, weightSumA = 0.0, weightSumB = 0.0;
//	double sumL = 0.0, sumA = 0.0, sumB = 0.0;
//	//double sigma = 15.0;
//
//	int regionum = srcMeans.size();
//	for (int i = 0; i < regionum; ++i)
//	{
//		double deltaL = rawcolor[0] - srcMeans[i].val[0];
//		double deltaA = rawcolor[1] - srcMeans[i].val[1];
//		double deltaB = rawcolor[2] - srcMeans[i].val[2];
//
//		double sigmaL = srcStddevs[i].val[0];
//		double sigmaA = srcStddevs[i].val[1];
//		double sigmaB = srcStddevs[i].val[2];
//
//		double weightL = exp(-deltaL*deltaL / (2*sigmaL*sigmaL) ) / (sqrt(2*_PI)*sigmaL);
//		double weightA = exp(-deltaA*deltaA / (2*sigmaA*sigmaA) ) / (sqrt(2*_PI)*sigmaA);
//		double weightB = exp(-deltaB*deltaB / (2*sigmaB*sigmaB) ) / (sqrt(2*_PI)*sigmaB);
//
//		weightSumL += weightL;
//		weightSumA += weightA;
//		weightSumB += weightB;
//
//
//		sumL += ( weightL * (Lfactors[i] * deltaL + refMeans[i].val[0] ) );
//		sumA += ( weightA * (Afactors[i] * deltaA + refMeans[i].val[1] ) );
//		sumB += ( weightB * (Bfactors[i] * deltaB + refMeans[i].val[2] ) );		
//	}
//
//	if (weightSumL != 0.0)
//		newcolor[0] = sumL / weightSumL;
//	if (weightSumA != 0.0)
//		newcolor[1] = sumA / weightSumA;
//	if (weightSumB != 0.0)
//		newcolor[2] = sumB / weightSumB;	
//}