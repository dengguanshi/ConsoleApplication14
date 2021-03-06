
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "common.h"
#include "ANNKmeans.h"
#include "ConsoleApplication16.cpp"

///////////////////////////////// K-Means ///////////////////////////////
int test_opencv_kmeans()
{
	//训练的文件夹
	const std::string image_path{ "C:/Users/huangzb/source/repos/ConsoleApplication14/ConsoleApplication14/data/digit/handwriting_0_and_1/" };
	//读取
	cv::Mat tmp = cv::imread(image_path + "0_1.jpg", 0);
	//获取数据
	CHECK(tmp.data != nullptr && tmp.channels() == 1);
	const int samples_number{ 80 }, every_class_number{ 20 }, categories_number{ samples_number / every_class_number };

	//创建80*tmp总长度的图片
	cv::Mat samples_data(samples_number, tmp.rows * tmp.cols, CV_32FC1);
	//创建80*1的图片
	cv::Mat labels(samples_number, 1, CV_32FC1);
	//创建指向80*归一化的图像
	float* p1 = reinterpret_cast<float*>(labels.data);

	for (int i = 1; i <= every_class_number; ++i) {
		static const std::vector<std::string> digit{ "0_", "1_", "2_", "3_" };
		CHECK(digit.size() == categories_number);
		static const std::string suffix{ ".jpg" };

		for (int j = 0; j < categories_number; ++j) {
			std::string image_name = image_path + digit[j] + std::to_string(i) + suffix;
			cv::Mat image = cv::imread(image_name, 0);
			CHECK(!image.empty() && image.channels() == 1);
			image.convertTo(image, CV_32FC1);
			//将读出来的图像灰度化并转化成一行
			image = image.reshape(0, 1);
			//创建和测试图像一样长的训练图像
			tmp = samples_data.rowRange((i - 1) * categories_number + j, (i - 1) * categories_number + j + 1);
			//将tmp赋值给image
			image.copyTo(tmp);

			//指向所有的训练的图图像
			p1[(i - 1) * categories_number + j] = j;
		}
	}
	//长度为行数的vector
	std::vector<std::vector<float>> data(samples_data.rows);

	for (int i = 0; i < samples_data.rows; ++i) {
		data[i].resize(samples_data.cols);
		const float* p = (const float*)samples_data.ptr(i);
		memcpy(data[i].data(), p, sizeof(float) * samples_data.cols);
	}
	const int K{ 4 }, attemps{ 100 }, max_iter_count{ 100 };
	const double epsilon{ 0.001 };
	const int flags = ANN::KMEANS_RANDOM_CENTERS;
	std::vector<int> best_labels;
	double compactness_measure{ 0. };
	std::vector<std::vector<float>> centers;
	ANN::kmeans<float>(data, K, best_labels, centers, compactness_measure, max_iter_count, epsilon, attemps, flags);
	fprintf(stdout, "K = %d, attemps = %d, iter count = %d, compactness measure =  %f\n",
		K, attemps, max_iter_count, compactness_measure);
	CHECK(best_labels.size() == samples_number);
	const auto* p2 = best_labels.data();
	for (int i = 1; i <= every_class_number; ++i) {
		for (int j = 0; j < categories_number; ++j) {
			fprintf(stdout, "  %d  ", *p2++);
		}
		fprintf(stdout, "\n");
	}
	return 0;
}