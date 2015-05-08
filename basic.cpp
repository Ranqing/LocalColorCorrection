#include "basic.h"

float idot(const cv::Vec3f a, const cv::Vec3f b)
{
	float innerp = a.val[0] * b.val[0] + a.val[1] * b.val[1] + a.val[2] * b.val[2];	
	return innerp;
}

float idot(const cv::Vec4f a, const cv::Vec4f b)
{
	float innerp = a.val[0] * b.val[0] + a.val[1] * b.val[1] + a.val[2] * b.val[2] + a.val[3] * b.val[3];
	return innerp;
}

void mul(const cv::Mat mtx, const cv::Vec4f a, cv::Vec4f& ret)
{
	if (mtx.cols != 4)
	{
		std::cerr << "can not multiply the mtx and the vector. mtx rows != 4" << std::endl;
		exit(1);
	}

	int rows = mtx.rows;
	int cols = mtx.cols;

	for (int i = 0; i < rows; ++ i)
	{
		cv::Vec4f tmpvec; 
		for(int j = 0; j < cols; ++ j)
			tmpvec.val[j] = mtx.at<float>(i, j);			

		ret.val[i] = idot(tmpvec, a);
	}
}

Scalar randomColor(RNG& rng)
{
	int icolor = (unsigned) rng;
	return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}
