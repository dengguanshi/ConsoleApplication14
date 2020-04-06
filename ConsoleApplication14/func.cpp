// surftest.cpp : 定义控制台应用程序的入口点。
//

#include <opencv2/core/core.hpp>
#include <iostream>
#include <time.h>
#include <vector>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2\core\core_c.h>
#include "func.h"
using namespace std;
using namespace cv;

IntegralImg::IntegralImg(Mat img)
{
	this->Original = img;
	integral(this->Original, this->Integral);
	this->Width = img.cols;
	this->Height = img.rows;
	cout << "this->Width" << endl;
	cout << this->Width << endl;
	cout << "this->Height" << endl;
	cout << this->Height << endl;
}
//计算积分图像 A-B-C+D计算左上角起始点指定的矩形内的像素总和坐标和大小
float IntegralImg::AreaSum(int x, int y, int dx, int dy)
{
	int r1;
	int c1;
	int r2;
	int c2;
	r1 = std::min(x, Height)-1;
	c1 = std::min(y, Width)-1;
	r2 = std::min(x + dx, Height)-1;
	c2 = std::min(y + dy, Width)-1;
	r1 = std::max(r1, 0);
	c1 = std::max(c1, 0);
	r2 = std::max(r2, 0);
	c2 = std::max(c2, 0);
	double A = this->Integral.at<double>(r1, c1);
	double B = this->Integral.at<double>(r2, c1);
	double C = this->Integral.at<double>(r1, c2);
	double D = this->Integral.at<double>(r2, c2);
	return (float)std::max(0.0, A + D - B - C);
}


ResponseLayer::ResponseLayer(IntegralImg* img, int octave, int interval)
{
	this->Step = (int)pow(2.0, octave - 1);
	this->Width = img->Width / this->Step;
	this->Height = img->Height / this->Step;
	this->Lobe = (int)pow(2.0, octave) * interval + 1;
	this->Lobe2 = this->Lobe * 2 - 1;
	this->Size = 3 * this->Lobe;
	this->Border = this->Size / 2;
	this->Count = this->Size * this->Size;
	this->Octave = octave;
	this->Interval = interval;
	this->Data = new Mat(this->Height, this->Width, CV_32FC1);
	this->LapData = new Mat(this->Height, this->Width, CV_32FC1);
	this->BuildLayerData(img);
}

void ResponseLayer::BuildLayerData(IntegralImg* img)
{
	float inverse_area = 1.0 / this->Count;
	float Dxx, Dyy, Dxy;

	for (int r = 0, x = 0; x < Height; r += this->Step, x += 1)
	{
		for (int c = 0, y = 0; y < Width; c += this->Step, y += 1)
		{
			Dxx = img->AreaSum(r - Lobe + 1, c - Border, Lobe2, Size) - img->AreaSum(r - Lobe + 1, c - Lobe / 2, Lobe2, Lobe) * 3;
			Dyy = img->AreaSum(r - Border, c - Lobe + 1, Size, Lobe2) - img->AreaSum(r - Lobe / 2, c - Lobe + 1, Lobe, Lobe2) * 3;
			Dxy = img->AreaSum(r - Lobe, c + 1, Lobe, Lobe) + img->AreaSum(r + 1, c - Lobe, Lobe, Lobe)
				- img->AreaSum(r - Lobe, c - Lobe, Lobe, Lobe) - img->AreaSum(r + 1, c + 1, Lobe, Lobe);
			Dxx *= inverse_area;
			Dyy *= inverse_area;
			Dxy *= inverse_area;

			this->Data->at<float>(x, y) = (Dxx * Dyy - 0.81f * Dxy * Dxy);
			this->LapData->at<float>(x, y) = (Dxx + Dyy >= 0 ? 1 : 0);
		}
	}
}
float ResponseLayer::GetResponse(int x, int y, int step)
{
	int scale = step / this->Step;
	//std::cout<<this->Data->at<float>((x*scale),(y*scale))<<std::endl;
	return this->Data->at<float>((x * scale), (y * scale));
}

float ResponseLayer::GetLaplacian(int x, int y, int step)
{
	int scale = step / this->Step;
	return this->LapData->at<float>((x * scale), (y * scale));
}


//! Gets the distance in descriptor space between Ipoints
float IPoint::operator-(const IPoint& rhs)//运算符重载
{
	float sum = 0.f;
	for (int i = 0; i < 64; ++i)
		sum += (this->descriptor[i] - rhs.descriptor[i]) * (this->descriptor[i] - rhs.descriptor[i]);
	return sqrt(sum);//sqrt求非负数的平方根
}

//! Populate IpPairVec with matched ipts 
void IPoint::GetMatches(vector<IPoint>& ipts1, vector<IPoint>& ipts2, vector< pair<IPoint, IPoint> >& matches)
{
	float dist, d1, d2;
	IPoint* match = NULL;

	matches.clear();

	for (unsigned int i = 0; i < ipts1.size(); i++)
	{
		d1 = d2 = FLT_MAX;

		for (unsigned int j = 0; j < ipts2.size(); j++)
		{
			dist = ipts1[i] - ipts2[j];

			if (dist < d1) // if this feature matches better than current best
			{
				d2 = d1;
				d1 = dist;
				match = &ipts2[j];
			}
			else if (dist < d2) // this feature matches better than second best
			{
				d2 = dist;
			}
		}

		// If match has a d1:d2 ratio < 0.65 ipoints are a match
		if (d1 / d2 < 0.65)
		{
			// Store the change in position
			ipts1[i].dx = match->x - ipts1[i].x;
			ipts1[i].dy = match->y - ipts1[i].y;
			matches.push_back(std::make_pair(ipts1[i], *match));
		}
	}
}


FastHessian::FastHessian(IntegralImg iImg, int octaves, int intervals, float threshold)
	:Octaves(octaves), Intervals(intervals), Img(iImg), Threshold(threshold)
{
	GeneratePyramid();
}
//生成金字塔
void FastHessian::GeneratePyramid()
{

	for (int o = 1; o <= Octaves; o++)
	{
		for (int i = 1; i <= Intervals; i++)
		{
			int size = 3 * ((int)pow(2.0, o) * i + 1);
			if (!this->Pyramid.count(size))
			{
			
				this->Pyramid[size] = new ResponseLayer(&Img, o, i);
				//imshow("d",abs((*(Pyramid[size])->Data)*100));
				//cv::waitKey();
			}
		}
	}
}
void FastHessian::GetIPoints()
{
	// Clear the vector of exisiting IPoints
	this->IPoints.clear();

	// Get the response layers
	ResponseLayer* b, * m, * t;
	//对Octave循环
	for (int o = 1; o <= this->Octaves; ++o)
	{
		//一个Octave中两个Interval的Size的差值
		int step = (int)(3 * pow(2.0, o));
		//计算当前层所需要的算子的Size
		int size = step + 3;
		//该算子的作用步长
		int s = (int)pow(2.0, o - 1);
		//卷积后图片的宽度
		int width = this->Img.Width / s;
		//卷积后图片的长度
		int height = this->Img.Height / s;

		//对Interval循环
		for (int i = 1; i <= this->Intervals - 2; ++i)
		{

			b = this->Pyramid[size];			//最底层
			m = this->Pyramid[size + step];		//中间层
			t = this->Pyramid[size + 2 * step];		//最高层

													//计算Border，在Border内的像素不记录为关键点
													//这里的Border计算有些疑惑，为何要除以Step？
			int border = (t->Border + 1) / (t->Step);

			//遍历所有的点，寻找符合极大抑制的点
			//OpenSurf里面遍历了所有包含Border在内的点
			//这里直接忽略那些点，从第一个有意义像素开始
			for (int r = border + 1; r < height - border; ++r)
			{
				for (int c = border + 1; c < width - border; ++c)
				{
					//判断中间层的中间元素是否是其周围26个元素中最大的
					if (IsExtremum(r, c, s, t, m, b))
					{
						//进行亚像素级别的差值，寻找亚像素级别的特征点
						InterpolateExtremum(r, c, s, t, m, b);
						//cout<<'('<<r<<','<<c<<')'<<endl;
					}
				}
			}
			//上升一个层
			size += step;
		}
	}
	//ShowIPoint();
}
//在图片上画出关键点
void FastHessian::ShowIPoint()
{
	for (int i = 0; i < this->IPoints.size(); i++)
	{
		cv::circle(this->Img.Original, cv::Point(this->IPoints[i].x, this->IPoints[i].y), 3, cv::Scalar(255, 0, 0, 100));
	}
	imshow("d", this->Img.Original);
}
//极大值抑制
bool FastHessian::IsExtremum(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	// check the candidate point in the middle layer is above thresh 
	float candidate = m->GetResponse(r, c, step);
	if (candidate < this->Threshold)
		return 0;

	for (int rr = -1; rr <= 1; ++rr)
	{
		for (int cc = -1; cc <= 1; ++cc)
		{
			// if any response in 3x3x3 is greater candidate not maximum
			if (
				t->GetResponse(r + rr, c + cc, step) >= candidate ||							//与顶层9个元素比较
				((rr != 0 || cc != 0) && m->GetResponse(r + rr, c + cc, step) >= candidate) ||	//与中间层8个元素比较
				b->GetResponse(r + rr, c + cc, step) >= candidate								//与底层9个元素比较
				)
				return 0;
		}
	}
	return 1;
}
//亚像素求解
void FastHessian::InterpolateExtremum(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	// get the step distance between filters
	// check the middle filter is mid way between top and bottom
	int filterStep = (m->Size - b->Size);
	assert(filterStep > 0 && t->Size - m->Size == m->Size - b->Size);

	// Get the offsets to the actual location of the extremum
	double xi = 0, xr = 0, xc = 0;
	InterpolateStep(r, c, step, t, m, b, &xi, &xr, &xc);

	// If point is sufficiently close to the actual extremum
	if (fabs(xi) < 0.5f && fabs(xr) < 0.5f && fabs(xc) < 0.5f)
	{
		IPoint p;
		p.x = static_cast<float>((c + xc) * step);
		p.y = static_cast<float>((r + xr) * step);
		p.scale = static_cast<float>((0.1333f) * (m->Size + xi * filterStep));
		p.laplacian = static_cast<int>(m->GetLaplacian(r, c, step));
		this->IPoints.push_back(p);
	}
}
//用泰勒展开求解极值点
void FastHessian::InterpolateStep(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b,
	double* xi, double* xr, double* xc)
{
	Mat dD, H, H_inv, X;

	dD = Deriv3D(r, c, step, t, m, b);
	//cout<<dD<<endl;
	H = Hessian3D(r, c, step, t, m, b);
	//cout<<H<<endl;
	invert(H, H_inv, CV_SVD);
	//cout<<H_inv<<endl;
	gemm(H_inv, dD, -1, NULL, 0, X, 0);

	*xc = X.at<double>(0, 0);
	*xr = X.at<double>(1, 0);
	*xi = X.at<double>(2, 0);
}
//计算一阶导数
Mat FastHessian::Deriv3D(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	double dx, dy, ds;
	dx = (m->GetResponse(r, c + 1, step) - m->GetResponse(r, c - 1, step)) / 2.0;
	dy = (m->GetResponse(r + 1, c, step) - m->GetResponse(r - 1, c, step)) / 2.0;
	ds = (t->GetResponse(r, c, step) - b->GetResponse(r, c, step)) / 2.0;

	//构造一阶导数
	Mat dI = (Mat_<double>(3, 1) << dx, dy, ds);

	return dI;
}

//计算二阶导数
Mat FastHessian::Hessian3D(int r, int c, int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b)
{
	double v, dxx, dyy, dss, dxy, dxs, dys;

	v = m->GetResponse(r, c, step);
	dxx = m->GetResponse(r, c + 1, step) + m->GetResponse(r, c - 1, step) - 2 * v;
	dyy = m->GetResponse(r + 1, c, step) + m->GetResponse(r - 1, c, step) - 2 * v;
	dss = t->GetResponse(r, c, step) + b->GetResponse(r, c, step) - 2 * v;
	dxy = (m->GetResponse(r + 1, c + 1, step) - m->GetResponse(r + 1, c - 1, step) -
		m->GetResponse(r - 1, c + 1, step) + m->GetResponse(r - 1, c - 1, step)) / 4.0;
	dxs = (t->GetResponse(r, c + 1, step) - t->GetResponse(r, c - 1, step) -
		b->GetResponse(r, c + 1, step) + b->GetResponse(r, c - 1, step)) / 4.0;
	dys = (t->GetResponse(r + 1, c, step) - t->GetResponse(r - 1, c, step) -
		b->GetResponse(r + 1, c, step) + b->GetResponse(r - 1, c, step)) / 4.0;

	//构造Hessian矩阵
	Mat H = (Mat_<double>(3, 3) <<
		dxx, dxy, dxs,
		dxy, dyy, dys,
		dxs, dys, dss);

	return H;
}

//! SURF priors (these need not be done at runtime)
const float pi = 3.14159f;

//! lookup table for 2d gaussian (sigma = 2.5) where (0,0) is top left and (6,6) is bottom right
const float gauss25[7][7] = {
	0.02546481,	0.02350698,	0.01849125,	0.01239505,	0.00708017,	0.00344629,	0.00142946,
	0.02350698,	0.02169968,	0.01706957,	0.01144208,	0.00653582,	0.00318132,	0.00131956,
	0.01849125,	0.01706957,	0.01342740,	0.00900066,	0.00514126,	0.00250252,	0.00103800,
	0.01239505,	0.01144208,	0.00900066,	0.00603332,	0.00344629,	0.00167749,	0.00069579,
	0.00708017,	0.00653582,	0.00514126,	0.00344629,	0.00196855,	0.00095820,	0.00039744,
	0.00344629,	0.00318132,	0.00250252,	0.00167749,	0.00095820,	0.00046640,	0.00019346,
	0.00142946,	0.00131956,	0.00103800,	0.00069579,	0.00039744,	0.00019346,	0.00008024
};

//-------------------------------------------------------

SurfDescriptor::SurfDescriptor(IntegralImg& img, std::vector<IPoint>& iPoints) :Img(img), IPoints(iPoints)
{

}




//我主要提取了有旋转不变性的特征

//提取当前关键点在附近区域的主方向
void SurfDescriptor::GetOrientation()
{
	for (int i = 0; i < this->IPoints.size(); i++)
	{
		const int pCount = 109;
		IPoint& p = IPoints[i];
		float gauss = 0.f;
		int s = fRound(p.scale), r = fRound(p.y), c = fRound(p.x);
		float resX[pCount], resY[pCount], Ang[pCount];
		int id[] = { 6,5,4,3,2,1,0,1,2,3,4,5,6 };

		int idx = 0;

		//计算6倍scale的区域的haar特征
		for (int i = -6; i <= 6; i++)
		{
			for (int j = -6; j <= 6; j++)
			{
				if (i * i + j * j < 36)
				{
					//用4倍scale的haar特征提取x y方向上的梯度特征
					//为何是4倍sigma？
					gauss = gauss25[id[i + 6]][id[j + 6]];
					resX[idx] = gauss * haarX(r + j * s, c + i * s, 4 * s);
					resY[idx] = gauss * haarY(r + j * s, c + i * s, 4 * s);
					//计算当前点的方向特征
					Ang[idx] = getAngle(resX[idx], resY[idx]);
					idx++;
				}
			}
		}

		//计算主方向
		float sumX = 0.f, sumY = 0.f;
		float maxX = 0.f, maxY = 0.f;
		float max = 0.f, orientation = 0.f;
		float ang1 = 0.f, ang2 = 0.f;

		//计算pi/3扇形的特征点
		//步长为0.15
		float pi3 = pi / 3.0f;
		for (ang1 = 0; ang1 < 2 * pi; ang1 += 0.15f)
		{
			ang2 = (ang1 + pi3 > 2 * pi ? ang1 - 5.0f * pi3 : ang1 + pi3);
			sumX = sumY = 0.f;
			for (int k = 0; k < pCount; k++)
			{
				const float& ang = Ang[k];
				if (ang1 < ang2 && ang1 < ang && ang < ang2)
				{
					sumX += resX[k];
					sumY += resY[k];
				}
				//居然在or上面少了一对括号
				else if (ang1 > ang2 &&
					((0 < ang && ang < ang2) || (ang1 < ang && ang < 2 * pi)))
				{
					sumX += resX[k];
					sumY += resY[k];
				}
			}

			//找到主方向，也就是模最大的方向
			if (sumX * sumX + sumY * sumY > max)
			{
				max = sumX * sumX + sumY * sumY;
				maxX = sumX;
				maxY = sumY;
			}
		}

		p.orientation = getAngle(maxX, maxY);
	}
}

//画出主方向特征
void SurfDescriptor::DrawOrientation()
{
	int r1, c1, c2, r2;
	for (int i = 0; i < this->IPoints.size(); i++)
	{
		r1 = fRound(IPoints[i].y);
		c1 = fRound(IPoints[i].x);
		c2 = fRound(10 * cos(IPoints[i].orientation)) + c1;
		r2 = fRound(10 * sin(IPoints[i].orientation)) + r1;
		cv::line(this->Img.Original, cv::Point(c1, r1), cv::Point(c2, r2), cv::Scalar(0, 255, 0));
	}
	imshow("d", this->Img.Original);
}

//根据主方向的值，提取4*4*4=64维的Haar特征
void SurfDescriptor::GetDescriptor()
{
	//OpenSURF 在这个函数写得让人很难理解
	//不像是为了效率，反正写得很纠结
	//由于将特征点附近区域划分成4*4的子区域
	//那么特征点是某个子区域的中心点
	int o[] = { -7, -2, 3, 8 };
	//int so[]={-2, -1, 0, 1, 2};

	for (int t = 0; t < this->IPoints.size(); t++)
	{
		IPoint& p = IPoints[t];
		float scale = p.scale;
		float* desp = p.descriptor;
		int x = fRound(p.x);
		int y = fRound(p.y);
		float co = cos(p.orientation);
		float si = sin(p.orientation);
		float cx = -0.5f, cy = 0.f; //Subregion centers for the 4x4 gaussian weighting
		int count = 0;
		float len = 0.f;

		for (int i = 0; i < 4; i++)
		{
			cx += 1.f;
			cy = -0.5f;
			for (int j = 0; j < 4; j++)
			{
				int xs = fRound(RotateX(scale * o[i], scale * o[j], si, co) + x);
				int ys = fRound(RotateY(scale * o[i], scale * o[j], si, co) + y);
				float dx = 0.f, dy = 0.f, mdx = 0.f, mdy = 0.f;

				cy += 1.f;
				for (int k = o[i] - 5; k <= o[i] + 3; k++)
				{
					for (int l = o[j] - 5; l <= o[j] + 3; l++)
					{
						int sample_x = fRound(RotateX(scale * k, scale * l, si, co) + x);
						int sample_y = fRound(RotateY(scale * k, scale * l, si, co) + y);

						//为何是2.5*scale，文章里面写的是3.3*scale
						float gauss_s1 = gaussian(xs - sample_x, ys - sample_y, 2.5f * scale);
						float rx = haarX(sample_y, sample_x, 2 * fRound(scale));
						float ry = haarY(sample_y, sample_x, 2 * fRound(scale));

						float rrx = gauss_s1 * RotateX(rx, ry, si, co);
						float rry = gauss_s1 * RotateY(rx, ry, si, co);

						dx += rrx;
						dy += rry;
						mdx += fabs(rrx);
						mdy += fabs(rry);
					}
				}

				float gauss_s2 = gaussian(cx - 2.f, cy - 2.f, 1.5f);

				desp[count++] = dx * gauss_s2;
				desp[count++] = dy * gauss_s2;
				desp[count++] = mdx * gauss_s2;
				desp[count++] = mdy * gauss_s2;

				len += (dx * dx + dy * dy + mdx * mdx + mdy * mdy) * gauss_s2 * gauss_s2;
			}
		}

		len = sqrt(len);
		for (int i = 0; i < 64; ++i)
			desp[i] /= len;

	}

}

//以下为辅助函数
//根据角度，旋转坐标
inline float SurfDescriptor::RotateX(float x, float y, float si, float co)
{
	return -x * si + y * co;
}

inline float SurfDescriptor::RotateY(float x, float y, float si, float co)
{
	return x * co + y * si;
}

//! Round float to nearest integer
inline int SurfDescriptor::fRound(float flt)
{
	return (int)floor(flt + 0.5f);
}
//-------------------------------------------------------

//! Calculate the value of the 2d gaussian at x,y
inline float SurfDescriptor::gaussian(int x, int y, float sig)
{
	return (1.0f / (2.0f * pi * sig * sig)) * exp(-(x * x + y * y) / (2.0f * sig * sig));
}

//-------------------------------------------------------

//! Calculate the value of the 2d gaussian at x,y
inline float SurfDescriptor::gaussian(float x, float y, float sig)
{
	return 1.0f / (2.0f * pi * sig * sig) * exp(-(x * x + y * y) / (2.0f * sig * sig));
}

//-------------------------------------------------------

//! Calculate Haar wavelet responses in x direction
inline float SurfDescriptor::haarX(int row, int column, int s)
{
	return Img.AreaSum(row - s / 2, column, s, s / 2)
		- Img.AreaSum(row - s / 2, column - s / 2, s, s / 2);
}

//-------------------------------------------------------

//! Calculate Haar wavelet responses in y direction
inline float SurfDescriptor::haarY(int row, int column, int s)
{
	return Img.AreaSum(row, column - s / 2, s / 2, s)
		- Img.AreaSum(row - s / 2, column - s / 2, s / 2, s);
}

//-------------------------------------------------------

//! Get the angle from the +ve x-axis of the vector given by (X Y)
float SurfDescriptor::getAngle(float X, float Y)
{
	if (X > 0 && Y >= 0)
		return atan(Y / X);

	if (X < 0 && Y >= 0)
		return pi - atan(-Y / X);

	if (X < 0 && Y < 0)
		return pi + atan(Y / X);

	if (X > 0 && Y < 0)
		return 2 * pi - atan(-Y / X);

	return 0;
}

//画出关键点
void Visualize::DrawIPoint(char* name, cv::Mat img, std::vector<IPoint> ipts)
{
	cout << "Surf Found: " << ipts.size() << " Interest Points." << endl;
	for (int i = 0; i < ipts.size(); i++)
	{
		int scale = static_cast<int>(ipts[i].scale * 2.5);
		if (ipts[i].laplacian == 0)
		{
			cv::circle(img, ipts[i], scale, cv::Scalar(255, 0, 0));
		}
		else if (ipts[i].laplacian == 1)
		{
			cv::circle(img, ipts[i], scale, cv::Scalar(0, 0, 255));
		}
		int r1 = fRound(ipts[i].y);
		int c1 = fRound(ipts[i].x);
		int c2 = fRound(scale * cos(ipts[i].orientation)) + c1;
		int r2 = fRound(scale * sin(ipts[i].orientation)) + r1;
		cv::line(img, cv::Point(c1, r1), cv::Point(c2, r2), cv::Scalar(0, 255, 0));
	}
	imshow(name, img);
}

//一次画出所有匹配的点对
void Visualize::DrawMatch(cv::Mat img1, cv::Mat img2, std::vector< std::pair<IPoint, IPoint> > matches)
{
	int Width = img1.cols + img2.cols;
	int Height = max(img1.rows, img2.rows);
	//Mat Conbine=Mat::zeros(Height,Width,CV_8UC3);
	//img1.copyTo(Conbine(Rect(0, 0, img1.cols, img1.rows)));
	//img2.copyTo(Conbine(Rect(img1.cols,0,img2.cols,img2.rows)));
	//imshow("d",Conbine);
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		circle(img1, matches[i].first, 4, Scalar(255, 0, 0));
		circle(img2, matches[i].second, 4, Scalar(255, 0, 0));

		const int& w = img1.cols;
		line(img1, matches[i].first, Point(matches[i].second.x + w, matches[i].second.y), Scalar(255, 255, 255), 1);
		line(img2, Point(matches[i].first.x - w, matches[i].first.y), matches[i].second, Scalar(255, 255, 255), 1);
	}

	imshow("1", img1);
	imshow("2", img2);
}

//一步一步画出匹配的点对
void Visualize::DrawMatchStep(cv::Mat img1, cv::Mat img2, std::vector< std::pair<IPoint, IPoint> > matches)
{
	int Width = img1.cols + img2.cols;
	int Height = max(img1.rows, img2.rows);
	//Mat Conbine=Mat::zeros(Height,Width,CV_8UC3);
	//img1.copyTo(Conbine(Rect(0, 0, img1.cols, img1.rows)));
	//img2.copyTo(Conbine(Rect(img1.cols,0,img2.cols,img2.rows)));
	//imshow("d",Conbine);
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		circle(img1, matches[i].first, 4, Scalar(255, 0, 0));
		circle(img2, matches[i].second, 4, Scalar(255, 0, 0));

		const int& w = img1.cols;
		line(img1, matches[i].first, Point(matches[i].second.x + w, matches[i].second.y), Scalar(255, 255, 255), 1);
		line(img2, Point(matches[i].first.x - w, matches[i].first.y), matches[i].second, Scalar(255, 255, 255), 1);
		imshow("1", img1);
		imshow("2", img2);
		waitKey(1000);
	}


}

//! Round float to nearest integer
inline int Visualize::fRound(float flt)
{
	return (int)floor(flt + 0.5f);
}


vector<IPoint> Surf::GetAllFeatures(Mat img)
{
	//初始化图像对象，属性有原始图像，积分图像，高，宽
	IntegralImg IImg(img);
	FastHessian fh(IImg, 4, 4, 0.0001);

	fh.GetIPoints();
	SurfDescriptor sd(IImg, fh.IPoints);
	sd.GetOrientation();
	sd.GetDescriptor();

	//clock_t start;
	//clock_t end;
	//start = clock();
	//IntegralImg IImg(img);
	//end = clock();
	//cout << "IntegralImg took: " << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	//start = clock();
	//FastHessian fh(IImg, 4, 4, 0.0001);
	//fh.GetIPoints();
	//end = clock();
	//std::cout << "FastHessian took: " << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	//start = clock();
	//SurfDescriptor sd(IImg, fh.IPoints);
	//sd.GetOrientation();
	//sd.GetDescriptor();
	//end = clock();
	//std::cout << "Descriptor took: " << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
	return fh.IPoints;
}
Mat ReadFloatImg(const char* szFilename)
{
	Mat iImg = imread(szFilename, 0);
	Mat fImg;
	iImg.convertTo(fImg, CV_32FC1);
	fImg /= 255.0;
	return fImg;
}
//int main()
//{
//	clock_t start;
//	clock_t end;
//	start = clock();
//	Mat colorImg1 = imread("1.jpg");
//	string s;
//	Mat img1 = ReadFloatImg("1.jpg");
//	Mat colorImg2 = imread("2.jpg");
//	Mat img2 = ReadFloatImg("2.jpg");
//	Surf surf;
//	Visualize v;
//	vector<IPoint> ips1 = surf.GetAllFeatures(img1);
//	vector<IPoint> ips2 = surf.GetAllFeatures(img2);
//	//v.DrawIPoint("img1", colorImg1, ips1);
//	//imshow("原图",img1);
//	vector< pair<IPoint, IPoint> > matches;
//	IPoint::GetMatches(ips1, ips2, matches);
//	end = clock();
//	cout << "总时间: " << float(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
//	v.DrawMatch(colorImg1, colorImg2, matches);
//	waitKey(0);
//	return 0;
//}

