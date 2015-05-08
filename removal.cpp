#include "function.h"

void OutlierRemoval(vector<Point2i>& srcFeatures, vector<Point2i>& refFeatures, string dispFn/* = NULL*/)
{
	Mat disp = imread(dispFn, CV_LOAD_IMAGE_GRAYSCALE);
	if (disp.data == NULL)
	{
		cout << "failed to open " << dispFn << endl;
		return ;
	}

	int width = disp.size().width;
	int height = disp.size().height;

	int sz = srcFeatures.size();
	for (int i = 0; i < sz; ++i)
	{
		int sx = srcFeatures[i].x;
		int sy = srcFeatures[i].y;
		if (sx < 0 || sx >= width || sy < 0 || sy >= height)
		{
			cout << "out of boundary" << endl;
			refFeatures[i].x = -1;
			refFeatures[i].y = -1;
		}
		else
		{
			int rx = sx + disp.at<uchar>(sy, sx);
			int ry = sy;

			refFeatures[i].x = rx;
			refFeatures[i].y = ry;
		}
	}

	//解决边界问题
	vector<Point2i> tmpsrc, tmpref;
	for (int i = 0; i < sz; ++i)
	{
		if (srcFeatures[i].x >= 0 && srcFeatures[i].x < width && srcFeatures[i].y >= 0 && srcFeatures[i].y < height &&
			refFeatures[i].x >= 0 && refFeatures[i].x < width && refFeatures[i].y >= 0 && refFeatures[i].y < height)
		{
			tmpsrc.push_back(srcFeatures[i]);
			tmpref.push_back(refFeatures[i]);
		}
	}

	srcFeatures.clear();
	srcFeatures.resize(tmpsrc.size());
	refFeatures.clear();
	refFeatures.resize(tmpref.size());
	copy(tmpsrc.begin(), tmpsrc.end(), srcFeatures.begin());
	copy(tmpref.begin(), tmpref.end(), refFeatures.begin());

	cout << "Outliers Removal done." << endl;	
	
}