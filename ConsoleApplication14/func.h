#pragma once

#include <opencv2/core/core.hpp>
#include <iostream>
#include <time.h>
#include <vector>
#include <opencv2/ml/ml.hpp>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2\core\core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <fstream>
#include <cstring>
#include <iterator>
#include <vector>
#include <map>
#include<fstream>
#include <opencv2\imgproc\types_c.h>
#include "opencv2/imgcodecs/legacy/constants_c.h"
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv::ml;

Mat ReadFloatImg(const char* szFilename);
class IPoint :public Point2f
{
public:
	//float x;
	//float y;
	float dx;
	float dy;
	float scale;
	float orientation;
	float laplacian;
	float descriptor[64];
	float operator-(const IPoint& rhs);
	static void GetMatches(vector<IPoint>& ipts1, vector<IPoint>& ipts2, vector< pair<IPoint, IPoint> >& matches);
};
class IntegralImg
{
public:
	int Width;		//图片的宽
	int Height;		//图片的高
	Mat Original;	//原始图片
	Mat Integral;	//积分图像
	IntegralImg(Mat img);
	float AreaSum(int x, int y, int dx, int dy);
};

class ResponseLayer
{
public:
	//本层图像的宽度
	int Width;
	//本层图像的高度
	int Height;
	//模板作用的步长
	int Step;
	//模板的长度的1/3
	int Lobe;
	//Lobe*2-1
	int Lobe2;
	//模板的长度一半，边框
	int Border;
	//模板长度
	int Size;
	//模板元素个数
	int Count;
	//金字塔级数
	int Octave;
	//金字塔层数
	int Interval;
	//高斯卷积后的图片
	Mat* Data;
	//Laplacian矩阵
	Mat* LapData;

	ResponseLayer(IntegralImg* img, int octave, int interval);
	void BuildLayerData(IntegralImg* img);
	float GetResponse(int x, int y, int step);
	float GetLaplacian(int x, int y, int step);
};

class FastHessian
{
public:

	IntegralImg Img;
	//图像堆的组数
	int Octaves;
	//为图像堆中每组中的中间层数，该值加2等于每组图像中所包含的层数
	int Intervals;
	//Hessian矩阵行列式响应值的阈值
	float Threshold;

	map<int, ResponseLayer*> Pyramid;
	//特征点矢量数组
	vector<IPoint> IPoints;
	//构造函数
	FastHessian(IntegralImg iImg, int octaves, int intervals, float threshold);
	void GeneratePyramid();
	void GetIPoints();
	void ShowIPoint();
	bool IsExtremum(int r, int c,
		int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	void InterpolateExtremum(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	void InterpolateStep(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b,
		double* xi, double* xr, double* xc);
	Mat Deriv3D(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	Mat Hessian3D(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
};

class SurfDescriptor
{
public:
	IntegralImg& Img;
	std::vector<IPoint>& IPoints;

	void GetOrientation();
	void GetDescriptor();

	float gaussian(int x, int y, float sig);
	float gaussian(float x, float y, float sig);
	float haarX(int row, int column, int s);
	float haarY(int row, int column, int s);
	float getAngle(float X, float Y);
	float RotateX(float x, float y, float si, float co);
	float RotateY(float x, float y, float si, float co);
	int fRound(float flt);
	void DrawOrientation();

	SurfDescriptor(IntegralImg& img, std::vector<IPoint>& iPoints);
};

class Visualize
{
public:

	void DrawIPoint(char* name, cv::Mat img, std::vector<IPoint> ipts);
	void DrawMatch(cv::Mat img1, cv::Mat img2, std::vector< std::pair<IPoint, IPoint> > matches);
	void DrawMatchStep(cv::Mat img1, cv::Mat img2, std::vector< std::pair<IPoint, IPoint> > matches);
	int fRound(float flt);
};

class Surf
{
public:

	vector<IPoint> GetAllFeatures(Mat img);
};

class categorizer
{
private:
	// //从类目名称到数据的map映射
	map<string, Mat> result_objects;
	//存放所有训练图片的BOW
	map<string, Mat> allsamples_bow;
	//从类目名称到训练图集的映射，关键字可以重复出现
	multimap<string, Mat> train_set;
	// 训练得到的SVM
	Ptr<SVM>* stor_svms;
	//类目名称，也就是TRAIN_FOLDER设置的目录名
	vector<string> category_name;
	//类目数目
	int categories_size;
	//用SURF特征构造视觉词库的聚类数目
	int clusters;
	//存放训练图片词典
	Mat vocab;
	Surf surf;
	Visualize v;

	Ptr<SURF> featureDecter;
	Ptr<BOWKMeansTrainer> bowtrainer;
	Ptr<BFMatcher> descriptorMacher;
	Ptr<BOWImgDescriptorExtractor> bowDescriptorExtractor;

	//构造训练集合
	void make_train_set();
	// 移除扩展名，用来讲模板组织成类目
	string remove_extention(string);

	Mat getMyMat(Mat);

public:
	//构造函数
	categorizer(int);
	// 聚类得出词典
	void bulid_vacab();
	//构造BOW
	void compute_bow_image();
	//训练分类器
	void trainSvm();
	//将测试图片分类
	void category_By_svm();
};


//Mat Mycluster(const Mat& _descriptors)
//{
//	Mat labels, vocabulary;
//	//kmeans(_descriptors, 1000, labels, termcrit, attempts, flags, vocabulary);
//	return vocabulary;
//}

