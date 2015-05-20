#include "correspond.h"

void computeTransforms(const vector<vector<Point2f>>& srcFeaturesTab, const vector<vector<Point2f>>& refFeaturesTab, vector<Mat>& transforms, vector<int>& isMatched)
{
	int regionum = transforms.size();
	for (int i = 0; i < regionum; ++i)
	{
		const vector<Point2f>& srcReFts = srcFeaturesTab[i];
		const vector<Point2f>& refReFts = refFeaturesTab[i];

		Mat perspective ;

		if (refReFts.size() == 0)
		{
			perspective = Mat::zeros(3,3,CV_64FC1);
			isMatched[i] = 0;
		}
		else if (refReFts.size() <= 4)
		{
			int deltax = 0, cnt = refReFts.size();
			for (int j = 0; j < cnt; ++j)
			{
				deltax += (refReFts[j].x - srcReFts[j].x);
			}
			deltax /= cnt;

			perspective = Mat::zeros(3,3,CV_64FC1);
			perspective.at<double>(0,2) = deltax * 1.0;
			isMatched[i] = 1;
		}
		else
		{
			perspective = findHomography(srcReFts, refReFts, RANSAC);
			isMatched[i] = 2;
		}

		transforms[i] = perspective.clone();
	}	
}

void applyTransforms(const vector<vector<Point2f>>& srcPointsTab, const vector<Mat>& transforms, const vector<int>& isMatched, vector<vector<Point2f>>& refPointsTab)
{
	int regionum = refPointsTab.size();	
	for (int i = 0; i < regionum; ++i)
	{
		//��֪
		const vector<Point2f>& srcPts = srcPointsTab[i];
		const Mat perspective = transforms[i];
		
		//δ֪
		vector<Point2f>& refPts = refPointsTab[i];

		if (isMatched[i] == 0)
		{
			//cout << i << "-th region: no matches." << endl; 
			refPointsTab[i].clear();
			refPointsTab[i].resize(0);
		}
		else if (isMatched[i] == 1)
		{
			//cout << i << "-th region: degrade into an translation." << endl;
			int deltax = (int)perspective.at<double>(0,2);
			for (int j = 0; j < srcPts.size(); ++j)
			{
				Point2f sPt = srcPts[j];
				Point2f rPt = Point2f(sPt.x + deltax, sPt.y); 

				refPts.push_back(rPt);
			}
		}
		else
		{
			//cout << i << "-th region: perspective transform." << endl;
			cv::perspectiveTransform(srcPts, refPts, perspective);
		}
	}

}

void computeRefLabels(const vector<vector<Point2f>>& refPointsTab, const int& width, const int& height, const Mat& refVisibility, vector<int>& refLabels)
{
	//computeLabels(refPointsTab, width, height, refLabels);

	refLabels.clear();
	refLabels.resize(width * height, -1);

	int regioum = refPointsTab.size();
	for (int i = 0; i < regioum; ++i)
	{
		const vector<Point2f>& ppts = refPointsTab[i];
		if (ppts.size() == 0)
		{
			continue;
		}

		Mat msk = Mat::zeros(height, width, CV_8UC1);
		pointsToMask(ppts, refVisibility, msk);

		//��̬ѧ����
		//һ������
		Mat element = getStructuringElement(MORPH_RECT, cv::Size(3, 3));
		Mat out;
		cv::dilate(msk, out, element);

		//�������θ�ʴ
		element = getStructuringElement(MORPH_RECT, cv::Size(3,3));
		cv::erode(out, msk, element);
		cv::erode(msk, out, element);
		cv::erode(out, msk, element);

		//maskת��Ϊ�����ǩ
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				if (msk.at<uchar>(y,x) == 255)
					refLabels[y*width+x] = i;
			}
		}
	}
}

//������ƥ������
void FindRegionCorrespondence(const Mat& srcImg, const Mat& refImg, const vector<Point2i>& srcFeatures, const vector<Point2i>& refFeatures, const Mat& srcVisibility, const Mat& refVisibility,
	const vector<int>& srcLabels, const int& regionum, vector<int>& refLabels, string resultFolder)
{
	vector<vector<Point2f>> srcFeaturesTab(regionum);
	vector<vector<Point2f>> refFeaturesTab(regionum);
	int sz = srcFeatures.size();
	int width = srcImg.size().width;
	int height = srcImg.size().height;
	for (int i = 0; i < sz; ++i)
	{
		int index = srcFeatures[i].y * width + srcFeatures[i].x;
		int l = srcLabels[index];

		srcFeaturesTab[l].push_back(srcFeatures[i]);
		refFeaturesTab[l].push_back(refFeatures[i]);	
	}

	//������ƥ������
	vector<int> unmatchedIdx(0);
	for (int i = 0; i < regionum; ++i)
		if (srcFeaturesTab[i].size() == 0)
		{
			unmatchedIdx.push_back(i);
		}
	vector<vector<Point2f>> d_srcPointsTab(regionum);
	calPointsTab(srcLabels, width, height, d_srcPointsTab);
	
	string fn;
	Mat unmatchedMsk = Mat::zeros(height, width, CV_8UC1), unmatched;
	cout << "unmatched regions: " << endl;
	for (int i = 0; i < unmatchedIdx.size(); ++i)
	{
		int idx = unmatchedIdx[i];
		cout << idx << ' ';
		pointsToMask(d_srcPointsTab[idx], unmatchedMsk);
	}
	cout << endl;
	fn = resultFolder + "/unmatched_regions_mask.png";
	imwrite(fn, unmatchedMsk);
	srcImg.copyTo(unmatched, unmatchedMsk);
	fn = resultFolder + "/unmatched_regions.png";
	imwrite(fn, unmatched);
	

	//ÿ����������ص�
	vector<vector<Point2f>> srcPointsTab(regionum);  //��֪
	vector<vector<Point2f>> refPointsTab(regionum);  //δ֪
	calPointsTab(srcLabels, srcVisibility, srcPointsTab);   //ÿ������ֻ�����ɼ���Ϊ1������
	//calPointsTab(srcLabels, width, height, srcPointsTab);

	vector<Mat> transforms(regionum);
	vector<int> isMatched(regionum);
	computeTransforms(srcFeaturesTab, refFeaturesTab, transforms, isMatched);

	//deal with unmatched region
	//cout << endl << "copy with no-matched region." << endl;
	dealUnMatchedRegion(isMatched, transforms);
	//dealUnMatchedRegion(srcImg, srcPointsTab, srcFeaturesTab, srcLabels, isMatched, transforms);
	
	//apply transform to each region
	applyTransforms(srcPointsTab, transforms, isMatched, refPointsTab);

	//���refLabels
	computeRefLabels(refPointsTab, width, height, refVisibility, refLabels);	
}


void computePointsCenter(const vector<Point2f>& allPts, Point2f& center)
{
	center = Point2f(0.0f, 0.0f);

	int cnt = allPts.size();
	for (int i = 0; i < cnt; ++i)
	{
		center.x += allPts[i].x;
		center.y += allPts[i].y;
	}
	center.x /= cnt;
	center.y /= cnt;
}


void computeCenters(const vector<vector<Point2f>>& ptsTab, vector<Point2f>& centers)
{
	int regionum = centers.size();
	for (int i = 0; i < regionum; ++i)
	{
		const vector<Point2f>& ppts = ptsTab[i];
		computePointsCenter(ppts, centers[i]);
	}	
}

void buildColorNeighbors(const vector<Scalar>& meanColors, vector<vector<int>>& colorNeighbors)
{
	//���ÿ��������ɫ���������
	int regionum = colorNeighbors.size();
	for (int i = 0; i < regionum; ++ i)
	{
		Scalar baseColor = meanColors[i];
		vector<int>& pneigbors = colorNeighbors[i];

		pneigbors.resize(0);
		for (int j = 0; j < regionum && j != i; ++j )
		{
			//ʹ����ɫ��ŷʽ������Ϊ����
			Scalar recolor = meanColors[j];
			float colorDis = (recolor.val[0] - baseColor.val[0]) * (recolor.val[0] - baseColor.val[0]) +    //b
				(recolor.val[1] - baseColor.val[1]) * (recolor.val[1] - baseColor.val[1]) +    //g
				(recolor.val[2] - baseColor.val[2]) * (recolor.val[2] - baseColor.val[2]) ;    //r
			if (colorDis < 400)
			{
				pneigbors.push_back(j);
			}
		}
	} 
}

//��points���ҵ���anchor����ĵ�
bool findNearestPoint(const vector<Point2f>& points, const Point2f& anchor, Point2f& nearestPt)
{
	int sz = points.size();
	if (sz == 0)
	{
		//	cout << "no color similar points." << endl;
		nearestPt = Point2f(-1.0, -1.0);
		return false;
	}

	//����10000����ʹ��KDTree���в���
	if (sz > 10000)
	{
		Mat pointsMat(sz,2,CV_32FC1);
		for (int i = 0; i < sz; ++i)
		{
			float * data = pointsMat.ptr<float>(i);
			data[0] = points[i].x;
			data[1] = points[i].y;
		}

		Mat anchorPt(1,2,CV_32FC1);
		anchorPt.at<float>(0,0) = anchor.x; 
		anchorPt.at<float>(0,1) = anchor.y;
		
		//ʹ��Flann��
		cv::flann::Index flann_Idx(pointsMat, cv::flann::KDTreeIndexParams(4),cvflann::FLANN_DIST_EUCLIDEAN);
		//cout << "flann based KDTree have been done." << endl;

		//��������� 
		Mat idx, dist;	
		flann_Idx.knnSearch(anchorPt, idx, dist, 1, cv::flann::SearchParams());
		nearestPt = points[idx.at<int>(0,0)];
		return true;
	}

	int  minIdx = 0;
	float minDis = sqrt( pow((points[0].x - anchor.x), 2) + pow((points[0].y - anchor.y), 2) );
	for (int i = 1; i < sz; ++i)
	{
		float dis = sqrt( pow((points[i].x - anchor.x), 2) + pow((points[i].y - anchor.y), 2) );
		if (dis < minDis)
		{
			minDis = dis;
			minIdx = i;
		}
	}
	nearestPt = points[minIdx];
	return true;
}

void fillRegion(const vector<Point2f>& pts, const cv::Scalar color, Mat& img)
{
	for (int j = 0; j < pts.size(); ++j)
	{
		img.at<Vec3b>(pts[j].y, pts[j].x) = Vec3b(color.val[0], color.val[1], color.val[2]);
	}
}

// deal with unmatched region
void dealUnMatchedRegion(const Mat& srcImg, const vector<vector<Point2f>>& srcPointsTab, const vector<vector<Point2f>>& srcFeaturesTab, const vector<int>& srcLabels, vector<int>& isMatched, vector<Mat>& transforms )
{
	int regionum = transforms.size();
	int width = srcImg.size().width;
	int height = srcImg.size().height;

	vector<Scalar> means(regionum);
	computeMeans(srcImg, srcPointsTab, means);
	vector<Point2f> centers(regionum);
	computeCenters(srcPointsTab, centers);

	vector<vector<int>> colorNeighbors(regionum);
	buildColorNeighbors(means, colorNeighbors);
		
	//����δƥ�������
	for (int i = 0; i < regionum; ++ i)
	{
		if (isMatched[i] != 0) continue;
		
		//������ɫ�����������
		int reIdx = i;
		vector<Point2f> nearFeatures(0);
		for (int j = 0; j < colorNeighbors[i].size(); ++j)
		{
			int nearIdx = colorNeighbors[i][j];
			
			const vector<Point2f>& nearReFts = srcFeaturesTab[nearIdx];
			copy(nearReFts.begin(), nearReFts.end(), std::back_inserter(nearFeatures));
		}

		//��ɫ�����������������ĵ�
		Point2f nearest;
		bool isFind = findNearestPoint(nearFeatures, centers[reIdx], nearest);
		if (isFind == true)
		{
			int nearIdx = srcLabels[(int)nearest.y * width + (int)nearest.x];
			transforms[reIdx] = transforms[nearIdx].clone();
			isMatched[reIdx] =  isMatched[nearIdx];

			// debug : ��ʾ�������
			cout << i << "-th unmatched region: the nearest one is " << nearIdx << endl;
			
			Mat test = Mat::zeros(height, width, CV_8UC3);
			fillRegion(srcPointsTab[i], means[i], test);
			circle(test, centers[i], 5, cv::Scalar(255,255,255));
			
			for (int j = 0; j < colorNeighbors[i].size(); ++ j)
			{
				int nearIdx = colorNeighbors[i][j];
				fillRegion(srcPointsTab[nearIdx], means[nearIdx], test);
			}			
			circle(test, centers[nearIdx], 5, cv::Scalar(255, 255, 255));
			
			string savefn = "output/near_regions_" + type2string(i) + ".png";
			imwrite(savefn, test);
			//��ʾ����
		}
		else
		{
			cout << i << "-th unmatched region: no similar colored region." << endl;
		}		
	}
}


//deal unmatched region with manual
void dealUnMatchedRegion(vector<int>& isMatched, vector<Mat>& transforms)
{
	int regionum = transforms.size();

	vector<int> nearIdx(regionum); 
	//int manualInfo[100] = {4,7,7,7,4,4,10,5,9,17,37,17,37,42,42,17,18,42,47,51,47,51} ; 
	int manualInfo[100] = {4,7,7,7,11,6,10,5,9,17,37,17,37,42,42,39,44,51,47,51,51,51};

	cout << "deal unmatched region manually." << endl;

	//��ƥ��ľ�������
	int cnt = 0;
	for (int i = 0; i < regionum; ++i)
	{
		if (isMatched[i] != 0)
		{
			nearIdx[i] = i;
		}
		else
		{
		   nearIdx[i] =  manualInfo[cnt];
		   isMatched[i] =  isMatched[nearIdx[i]];
		   transforms[i] = transforms[nearIdx[i]];
		   cnt++;
		}
	}
	// 
}