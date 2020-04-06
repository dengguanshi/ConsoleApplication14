#pragma once
#include <opencv2/core/core.hpp>
#include <iostream>
#include <time.h>
#include <vector>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2\core\core_c.h>
using namespace std;
using namespace cv;

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
	int Width;		//ͼƬ�Ŀ�
	int Height;		//ͼƬ�ĸ�
	Mat Original;	//ԭʼͼƬ
	Mat Integral;	//����ͼ��
	IntegralImg(Mat img);
	float AreaSum(int x, int y, int dx, int dy);
};

class ResponseLayer
{
public:
	//����ͼ��Ŀ��
	int Width;
	//����ͼ��ĸ߶�
	int Height;
	//ģ�����õĲ���
	int Step;
	//ģ��ĳ��ȵ�1/3
	int Lobe;
	//Lobe*2-1
	int Lobe2;
	//ģ��ĳ���һ�룬�߿�
	int Border;
	//ģ�峤��
	int Size;
	//ģ��Ԫ�ظ���
	int Count;
	//����������
	int Octave;
	//����������
	int Interval;
	//��˹������ͼƬ
	Mat* Data;
	//Laplacian����
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
	//ͼ��ѵ�����
	int Octaves;
	//Ϊͼ�����ÿ���е��м��������ֵ��2����ÿ��ͼ�����������Ĳ���
	int Intervals;
	//Hessian��������ʽ��Ӧֵ����ֵ
	float Threshold;

	map<int, ResponseLayer*> Pyramid;
	//������ʸ������
	vector<IPoint> IPoints;
	//���캯��
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


