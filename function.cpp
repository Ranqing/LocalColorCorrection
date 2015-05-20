#include "function.h"

void readImage(string fn, Mat& img, int METHOD/* = CV_LOAD_IMAGE_UNCHANGED*/)
{
	img = imread(fn, METHOD);
	if (img.data == NULL)
	{
		cout << "failed to open " << fn << endl;
		return;
	}
}

void readFeatures(string fn, vector<Point2i>& features)
{
	fstream fin(fn.c_str(), ios::in);
	if (fin.is_open() == false)
	{
		cout << "failed to open " << fn << endl;
		return ;
	}

	int cnt;
	int x, y;

	fin >> cnt;
	for (int i = 0; i < cnt; ++i)
	{
		fin >> x >> y;
		features.push_back(Point2i(x,y));
	}
}

void showFeatures(const vector<Point2i>& features , const Mat& img, Mat& featuresImg, const cv::Scalar color /*= cv::Scalar(255,0,0)*/ )
{
	featuresImg = img.clone();

	for (int i = 0; i < features.size(); ++i)
	{
		Point2i pt = features[i];
		cv::circle(featuresImg, pt, 2, color);
	}
}

void readMatches(string fn, vector<Point2i>& refFeatures, vector<Point2i>& srcFeatures, vector<vector<bool>>& istaken)
{
	fstream fin(fn.c_str(), ios::in);
	if (fin.is_open() == false)
	{
		cout << "failed to open " << fn << endl;
		return ;
	}

	int sz;
	fin >> sz;

	for (int i = 0; i < sz; ++ i)
	{
		int rx, ry, sx, sy;
		fin >> rx >> ry >> sx >> sy;

		/*if (abs(ry - sy) >= 10)
		{
			continue;
		}*/

		if (istaken[sy][sx] == true)
		{
			continue;
		}

		/*if (abs(rx - sx) >= 250)
		{
			continue;
		}*/

		refFeatures.push_back(Point2i(rx, sy));
		srcFeatures.push_back(Point2i(sx, sy));
		istaken[sy][sx] = true;
	}
}

void readLabels(string fn, vector<int>& labels, int& regionum)
{
	fstream fin(fn.c_str(), ios::in);
	if (fin.is_open() == false)
	{
		cout << "failed to open " << fn << endl;
		return ;
	}

	int width, height;
	fin >> height >> width ;

	int sz = height * width;
	regionum = 0;

	labels.resize(sz);
	for (int i = 0; i < sz; ++ i)
	{
		fin >> labels[i];
		regionum = ( (labels[i] > regionum) ? labels[i] : regionum );		
	}
	regionum ++;

	fin.close();
}

void saveValidImages(const Mat& srcImg, const Mat& refImg, const Mat& srcSegments, const Mat& srcVis, const Mat& refVis, const string& folder, const string& srcFn, const string& refFn )
{
	int width = srcImg.size().width;
	int height = srcImg.size().height;

	Mat validSrcImg , validRefImg, validSrcSegments;
	Mat occSrcImg, occRefImg;
	Mat srcValidMsk = Mat(height, width, CV_8UC1, cv::Scalar(255)) - srcVis;
	Mat refValidMsk = Mat(height, width, CV_8UC1, cv::Scalar(255)) - refVis;

	srcImg.copyTo(validSrcImg, srcValidMsk);
	srcImg.copyTo(occSrcImg, srcVis);
	refImg.copyTo(validRefImg, refValidMsk);
	refImg.copyTo(occRefImg, refVis);

	srcSegments.copyTo(validSrcSegments, srcValidMsk);

	string savefn;

	savefn = folder + "/valid_" + srcFn + ".png";
	imwrite(savefn, validSrcImg);
	savefn = folder + "/valid_" + refFn + ".png";
	imwrite(savefn, validRefImg);
	savefn = folder + "/valid_segments_" + srcFn + ".png";
	imwrite(savefn, validSrcSegments);
	savefn = folder + "/occluded_" + srcFn + ".png";
	imwrite(savefn, occSrcImg);
	savefn = folder + "/occluded_" + refFn + ".png";
	imwrite(savefn, occRefImg);
	cout << "save valid and occluded images done." << endl;

	
}

void overlayTwoImages(const string& srcFn, const string& backFn, const double& alpha, string resultFn)
{
	Mat srcSegments, backSegments;

	readImage(srcFn, srcSegments);
	readImage(backFn, backSegments);
	
	int width = srcSegments.size().width;
	int height = backSegments.size().height;

	//contours得到mask

	Mat tmp(height, width, CV_8UC3, cv::Scalar(255,255,255));
	Mat srcContours, backContours;
	//tmp = srcSegments.clone();
	calContours(srcSegments, tmp, srcContours, Scalar(0,0,255));
	//tmp = backSegments.clone(); 
	calContours(backSegments, tmp, backContours, Scalar(255,0,0));

	Mat result;
	addWeighted(srcContours, alpha, backContours, 1-alpha, 0.0, result);
	imwrite(resultFn, result);
}

void saveMatches(string fn, vector<Point2i>& srcFeatures, vector<Point2i>& refFeatures)
{
	fstream fout(fn.c_str(), ios::out);
	if (fout.is_open() == false)
	{
		cout << "failed to write " << fn << endl;
		return ;
	}

	int sz = refFeatures.size();
	fout << sz << endl;

	for (int i = 0; i < sz; ++ i)
	{
		int rx, ry, sx, sy;

		rx = refFeatures[i].x;
		ry = refFeatures[i].y;
		sx = srcFeatures[i].x;
		sy = srcFeatures[i].y;
		
		fout << rx << ' ' << ry << ' ' << sx << ' ' << sy << endl; 

	}
	fout.close();
}

void saveLabels(string fn, int width, int height, vector<int>& labels)
{
	fstream fout(fn.c_str(), ios::out);
	if (fout.is_open() == false)
	{
		cout << "failed to write " << fn << endl;
		return ;
	}

	fout << width << ' ' << height << endl;
	for (int j = 0; j < height; ++j)
	{
		for (int i = 0; i < width; ++i)
		{
			fout << labels[j*width+i] << ' ';
		}
		fout << endl;
	}
	fout.close();
}

void calSegments(Mat img, int regionum, vector<int>& labels, Mat& segments)
{
	int width = img.size().width;
	int height = img.size().height;
	int sz = width * height;

	vector<vector<Point2i>> labelsTab(regionum);  //属于每个标签有那些像素点
	for (int i = 0; i < sz; ++i )
	{
		int l = labels[i];
		if (l == -1)
		{
			continue;
		}
		int x = i % width;
		int y = i / width;

		labelsTab[l].push_back(Point2i(x,y));
	}

	segments = Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < regionum; ++i)
	{
		vector<Point2i>& ppt = labelsTab[i];
		if (ppt.size() == 0)
			continue;

		Mat msk = Mat::zeros(height, width, CV_8UC1);
		for (int j = 0; j < ppt.size(); ++j)
		{
			msk.at<uchar>(ppt[j].y, ppt[j].x) = 255;
		}
		/*imshow("msk", msk);
		waitKey(0);
		destroyAllWindows();*/

		Scalar mean, stddv;
		cv::meanStdDev(img, mean, stddv, msk);
		//cout << mean << endl;
		for (int j = 0; j < ppt.size(); ++j)
		{
			segments.at<Vec3b>(ppt[j].y, ppt[j].x)[0] = mean.val[0];
			segments.at<Vec3b>(ppt[j].y, ppt[j].x)[1] = mean.val[1];
			segments.at<Vec3b>(ppt[j].y, ppt[j].x)[2] = mean.val[2];
		}
	}
}

//输出：segments - mean-shift filter结果
//colorsTab - 每个区域的颜色值
void calSegments(Mat img, vector<vector<Point2f>>& pointsTab, Mat& segments, vector<Scalar>& colorsTab)
{
	int regionum = pointsTab.size();
	int height = img.size().height;
	int width  = img.size().width;

	segments = Mat::zeros(height, width, CV_8UC3);
	for (int i = 0; i < regionum; ++i)
	{
		vector<Point2f>& ppt = pointsTab[i];
		if (ppt.size() == 0)
		{
			colorsTab.push_back(cv::Scalar(0,0,0));
			continue;
		}

		Mat msk = Mat::zeros(height, width, CV_8UC1);
		for (int j = 0; j < ppt.size(); ++j)
		{
			msk.at<uchar>(ppt[j].y, ppt[j].x) = 255;
		}

		Scalar mean, stddv;
		cv::meanStdDev(img, mean, stddv, msk);
		colorsTab.push_back(mean);
		//cout << mean << endl;
		for (int j = 0; j < ppt.size(); ++j)
		{
			segments.at<Vec3b>(ppt[j].y, ppt[j].x)[0] = mean.val[0];
			segments.at<Vec3b>(ppt[j].y, ppt[j].x)[1] = mean.val[1];
			segments.at<Vec3b>(ppt[j].y, ppt[j].x)[2] = mean.val[2];
		}
	}
}

//segments: 分块后的图像
//img: 显示轮廓的图像 : 三通道
//contours: 轮廓结果
void calContours(Mat segments, Mat img, Mat& contours, cv::Scalar color/* = cv::Scalar(255,255,255)*/)
{
	// Pixel offsets around the centre pixels starting from left, going clockwise
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	int width = img.size().width;    // w * h
	int height = img.size().height;
	int ch = img.channels();

	int sz = width * height;
	vector<bool> istaken(sz, false);
	vector<uchar> segmentsvec(0);
	vector<uchar> imgvec(0);
	Mat2PixelsVector(segments, segmentsvec);
	Mat2PixelsVector(img, imgvec);

	int mainindex  = 0;
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np = 0;
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;
					if( false == istaken[index] )
					{
						if( (int)segmentsvec[mainindex * ch] != (int)segmentsvec[index * ch] ) np++;
					}
				}
			}
			if( np > 2 )//1 for thicker lines and 2 for thinner lines
			{
				if (ch == 3)
				{
					imgvec[(j*width + k) * ch] = color.val[0];
					imgvec[(j*width + k) * ch + 1] = color.val[1];
					imgvec[(j*width + k) * ch + 2] = color.val[2];
				}
				/*else if (ch == 1)
				{
				imgvec[j*width+k] = color.val[0];
				}*/
				
				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}

	PixelsVector2Mat(imgvec, width, height, ch, contours);
}

//计算得到轮廓点
void calContours(Mat segments, Mat& contours)
{
	
}


void pointsToMask(const vector<Point2f>& points, Mat& mask)
{
	int width = mask.size().width;
	int height = mask.size().height;

	for (int j = 0; j < points.size(); ++j)
	{
		int x = (int)points[j].x;
		int y = (int)points[j].y;
		if (y>=0 && y<height && x>=0 && x<width)
		{
			mask.at<uchar>(y,x) = 255;
		}
	}
}

//考虑可见性
void pointsToMask(const vector<Point2f>& points, const Mat& visibility, Mat& mask)
{
	int width = mask.size().width;
	int height = mask.size().height;

	for (int j = 0; j < points.size(); ++j)
	{
		int x = (int)points[j].x;
		int y = (int)points[j].y;
		if (y>=0 && y<height && x>=0 && x<width )
		{
			if (visibility.at<uchar>(y,x)!=255)
			{
				mask.at<uchar>(y,x) = 255;
			}
		}
	}
}

//Mask中的有效像素
void maskToPoints(const Mat& mask, vector<Point2f>& points)
{
	int height = mask.size().height;
	int width = mask.size().width;
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			if (mask.at<uchar>(y,x) == 255)
			{
				points.push_back(Point2f(x,y));
			}			
		}
	}
}

void calPointsTab(const vector<int>& labels, const Mat& visibility, vector<vector<Point2f>>& pointsTab)
{
	int width = visibility.size().width;
	int height = visibility.size().height;
	int sz = labels.size();

	for (int i = 0; i < sz; ++i)
	{
		int l = labels[i];
		if (l == -1)
		{
			continue;
		}

		int x = i % width;
		int y = i / width;

		if (visibility.at<uchar>(y,x) != 255) //可见
		{
			pointsTab[l].push_back(Point2f(x, y));
		}	
	}
}

void calPointsTab(const vector<int>& labels, int width, int height, vector<vector<Point2f>>& pointsTab)
{
	int sz = labels.size();
	for (int i = 0; i < sz; ++i)
	{
		int l = labels[i];
		if (l == -1)
		{
			continue;
		}

		int x = i % width;
		int y = i / width;
		pointsTab[l].push_back(Point2f(x, y));
	}
}

void computeMeans(const Mat& img, const vector<vector<Point2f>>& pointsTab, vector<Scalar>& means)
{
	int width = img.size().width;
	int height = img.size().height;

	int regionum = means.size();
	for (int i = 0; i < regionum; ++i)
	{
		const vector<Point2f>& ppt = pointsTab[i];
		if (ppt.size() == 0)
			continue;

		Mat msk = Mat::zeros(height, width, CV_8UC1);
		for (int j = 0; j < ppt.size(); ++j)
		{
			msk.at<uchar>(ppt[j].y, ppt[j].x) = 255;
		}

		means[i] = cv::mean(img, msk);
	}
}
