//一些简单或常用的函数的声明

#ifndef SUMMER_BASICFUNC_H
#define SUMMER_BASICFUNC_H

#include "common.h"

#define _PI 3.14159265358979323846
#define NMAX 3000

//pixels vector convert to mat
template<typename PixelType> 
void PixelsVector2Mat(const std::vector<PixelType>& inpixels, int width, int height, int channel, cv::Mat& outmtx)
{
	if (channel == 3)
		outmtx.create(height, width, CV_8UC3);
	else if (channel == 1)
		outmtx.create(height, width, CV_8UC1);

	int i, j, index;
	for (j = 0; j < height; j ++)
	{
		for (i = 0; i < width; i ++)
		{
			index = ( j * width + i ) * channel;

			if (channel == 3)
				outmtx.at<cv::Vec3b>(j, i) = cv::Vec3i( (uchar)(int) (inpixels[index]/* + 0.5*/),
				(uchar)(int) (inpixels[index + 1]/* + 0.5*/),
				(uchar)(int) (inpixels[index + 2] /*+ 0.5*/) );
			else if (channel == 1)
				outmtx.at<uchar>(j, i) = (uchar)(int)(inpixels[index]/* + 0.5*/);				
		}
	}	
}

template<typename PixelType>
void Mat2PixelsVector(const cv::Mat inmtx, std::vector<PixelType>& outpixels)
{
	int width = inmtx.cols;
	int height = inmtx.rows;
	int channel = inmtx.channels();
	int index;

	outpixels.clear();
	outpixels.resize(width * height * channel);

	for (int j = 0; j < height; j ++)
	{
		for (int i = 0; i < width; i ++)
		{
			index = ( j * width + i ) * channel; 
			if (channel == 3)
			{
				cv::Vec3i tcolor = inmtx.at<cv::Vec3b>(j, i);
				outpixels[index] = tcolor[0];
				outpixels[index + 1] = tcolor[1];
				outpixels[index + 2] = tcolor[2];
			}
			else if (channel == 1)
			{
				int tgray = inmtx.at<uchar>(j, i);
				outpixels[index] = tgray;
			}
		}
	}
}

template<typename T>
std::string type2string(T num)
{
	std::stringstream ss;
	ss << num;
	return ss.str();
}

template<typename T>
T string2type(std::string str)
{
	std::stringstream stream;

	int num = 0;
	stream << str;
	stream >> num;

	return num;	
}

//float / double
template<typename T>
void Gauss1DFilter(int ksize, T sigma, T* &filter1d)
{
	assert(ksize % 2);

	filter1d = new T[ksize];

	T sigma2 = sigma * sigma;
	T fenmu = sqrt(2 * _PI) * sigma;
	T norm = 0.0;

	int offset = ksize / 2;
	for (int i = - offset; i <= offset; i ++)
	{
		filter1d[i + offset] = exp( - (i * i) / (2 * sigma2) );
		filter1d[i + offset] /= fenmu;
		norm += filter1d[i + offset];
	}
	for (int i = 0; i < ksize; i ++)
		filter1d[i] /= norm;
}

template<typename T>
void Gauss2DFilter(int ksize, T sigma, T* &filter2d)
{
	assert(ksize % 2);

	filter2d = new T[ksize * ksize];

	T * filter1d;
	Gauss1DFilter(ksize, sigma, filter1d);

	T norm = 0.0f;
	int offset = ksize / 2;
	for (int i = -offset; i <= offset; i ++)
	{
		for (int j = -offset; j <= offset; j ++)
		{
			int index = (i + offset) * ksize + (j + offset);
			filter2d[index] = filter1d[i + offset] * filter1d[j + offset];
			norm += filter2d[index];
		}
	}
	for (int i = 0; i < ksize * ksize; i ++)
		filter2d[i] /= norm;
}

float idot(const cv::Vec3f a, const cv::Vec3f b);
float idot(const cv::Vec4f a, const cv::Vec4f b);
void mul(const cv::Mat mtx, const cv::Vec4f a, cv::Vec4f& ret);

//生成随机颜色
Scalar randomColor(RNG& rng);

//获取当前目录
//char pwd[1024];
//_getcwd(pwd, 1021);
//cout << pwd << endl;

#endif