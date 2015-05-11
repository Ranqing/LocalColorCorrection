#include "function.h"

void computeMeanStddev(const Mat& img, const vector<vector<Point2f>>& pointsTab, vector<Scalar>& means, vector<Scalar>& stddevs )
{
	int width = img.size().width;
	int height = img.size().height;
	int regionum = pointsTab.size();

	for (int i = 0; i < regionum; ++i)
	{
		const vector<Point2f>& ppt = pointsTab[i];
		cv::Scalar m, s;

		if ( ppt.size() == 0 )
		{
			continue;
		}

		Mat msk = Mat::zeros(height, width, CV_8UC1);
		
		pointsToMask(ppt, msk);
		meanStdDev(img, m, s, msk);

		means[i] = m;
		stddevs[i] = s;
	}
}

void saveMeansStddvs(const string& fn , const vector<Scalar>& means, const vector<Scalar>& stddvs)
{
	int regionum = means.size();
	fstream fout(fn.c_str(), ios::out);
	if (fout.is_open() == false)
	{
		cout << "write in " << fn << " failed." << endl;
		return;
	}

	fout << regionum << endl;
	for (int i = 0; i < regionum; ++i)
	{
		fout << i  << ": " << setw(5) << means[i].val[0] << ' ' << setw(5)  << means[i].val[1] << ' ' << setw(5)  << means[i].val[2] << '\t' 
			 << setw(5) << stddvs[i].val[0] << ' ' << setw(5) << stddvs[i].val[1] << ' ' << setw(5) << stddvs[i].val[2] << endl;  
	}
	fout.close();
}

void correct(const Mat& src, const Mat& msk, const cv::Scalar& srcMean, const cv::Scalar& srcStddev, const cv::Scalar& refMean, const cv::Scalar& refStddev, Mat& srcResult)
{
	int width = src.size().width;
	int height = src.size().height;

	double Lfactor = refStddev.val[0] / srcStddev.val[0];
	double Afactor = refStddev.val[1] / srcStddev.val[1];
	double Bfactor = refStddev.val[2] / srcStddev.val[2];

	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			if (msk.at<uchar>(y,x) != 255)
				continue;

			double transferL = refMean.val[0] + Lfactor * (src.at<Vec3b>(y,x)[0] - srcMean.val[0]);
			double transferA = refMean.val[1] + Afactor * (src.at<Vec3b>(y,x)[1] - srcMean.val[1]);
			double transferB = refMean.val[2] + Bfactor * (src.at<Vec3b>(y,x)[2] - srcMean.val[2]);

			transferL = min(max(0.0, transferL), 255.0);
			transferA = min(max(0.0, transferA), 255.0);
			transferB = min(max(0.0, transferB), 255.0);

			srcResult.at<Vec3b>(y,x) = Vec3b(transferL, transferA, transferB);
		}
	}
}

void LocalColorCorrection(const Mat& src, const Mat& srcVis, const vector<int>& srclabels, const Mat& ref, const Mat& refVis, const vector<int>& reflabels, const int& regionum, Mat& srcResult )	
{
	vector<vector<Point2f>> srcPointsTab(regionum);
	vector<vector<Point2f>> srcVisPointsTab(regionum);
	vector<vector<Point2f>> refPointsTab(regionum);
	vector<vector<Point2f>> refVisPointsTab(regionum);

	int width = src.size().width;
	int height = src.size().height;

	//用于计算color correction function
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

	cv::Scalar srcGLmeans, srcGLstddev;
	cv::Scalar refGLmeans, refGLstddev;
	cv::meanStdDev(srcLab, srcGLmeans, srcGLstddev);
	cv::meanStdDev(refLab, refGLmeans, refGLstddev);	

	cout << "src global: " << srcGLmeans << ' ' << srcGLstddev << endl;
	cout << "ref global: " << refGLmeans << ' ' << refGLstddev << endl;

	vector<Scalar> srcMeans(regionum), refMeans(regionum);
	vector<Scalar> srcStddevs(regionum), refStddevs(regionum);

	computeMeanStddev(srcLab, srcVisPointsTab, srcMeans, srcStddevs );
	computeMeanStddev(refLab, refVisPointsTab, refMeans, refStddevs );

	string tmpfn ;
	tmpfn = "src_means_stddvs.txt";
	saveMeansStddvs(tmpfn, srcMeans, srcStddevs);
	tmpfn = "ref_means_stddvs.txt";
	saveMeansStddvs(tmpfn, refMeans, refStddevs);
	

	//被遮挡，无匹配的区域使用Global的参数进行代替

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

	Mat srcResultLab (height, width, CV_8UC3);
	Mat allmsk(height, width, CV_8UC1, cv::Scalar(255));
	correct(srcLab, allmsk, srcGLmeans, srcGLstddev, refGLmeans, refGLstddev, srcResultLab);

	for (int i = 0; i < regionum; ++i)
	{
		const vector<Point2f>& ppt = srcVisPointsTab[i];
		Mat msk = Mat::zeros(height, width, CV_8UC1);

		pointsToMask(ppt, msk);
		correct(srcLab, msk, srcMeans[i], srcStddevs[i], refMeans[i], refStddevs[i], srcResultLab);
	}

	cv::cvtColor(srcResultLab, srcResult, cv::COLOR_Lab2BGR);


	return;
}