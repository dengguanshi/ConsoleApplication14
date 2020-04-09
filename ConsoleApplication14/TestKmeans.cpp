
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "common.h"
#include "TestKmeans.h"
#include "ConsoleApplication16.cpp"

///////////////////////////////// K-Means ///////////////////////////////
int test_opencv_kmeans()
{
	//ѵ�����ļ���
	const std::string image_path{ "C:/Users/huangzb/source/repos/ConsoleApplication14/ConsoleApplication14/data/digit/handwriting_0_and_1/" };
	//��ȡ
	cv::Mat tmp = cv::imread(image_path + "0_1.jpg", 0);
	//��ȡ����
	CHECK(tmp.data != nullptr && tmp.channels() == 1);
	const int samples_number{ 80 }, every_class_number{ 20 }, categories_number{ samples_number / every_class_number };

	//����80*tmp�ܳ��ȵ�ͼƬ
	cv::Mat samples_data(samples_number, tmp.rows * tmp.cols, CV_32FC1);
	//����80*1��ͼƬ
	cv::Mat labels(samples_number, 1, CV_32FC1);
	//����ָ��80*��һ����ͼ��
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
			//����������ͼ��ҶȻ���ת����һ��
			image = image.reshape(0, 1);
			//�����Ͳ���ͼ��һ������ѵ��ͼ��
			tmp = samples_data.rowRange((i - 1) * categories_number + j, (i - 1) * categories_number + j + 1);
			//��tmp��ֵ��image
			image.copyTo(tmp);

			//ָ�����е�ѵ����ͼͼ��
			p1[(i - 1) * categories_number + j] = j;
		}
	}

	const int K{ 4 }, attemps{ 100 };
	//����������ս�����
	const cv::TermCriteria term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.01);
	cv::Mat labels_, centers_;
	int valueInt=kmeans(samples_data, K, labels_, term_criteria, attemps, cv::KMEANS_RANDOM_CENTERS, centers_);
	double value = cv::kmeans(samples_data, K, labels_, term_criteria, attemps, cv::KMEANS_RANDOM_CENTERS, centers_);
	fprintf(stdout, "K = %d, attemps = %d, iter count = %d, compactness measure =  %f\n",
		K, attemps, term_criteria.maxCount, valueInt);
	std::cout << "===============" ;
	fprintf(stdout, "K = %d, attemps = %d, iter count = %d, compactness measure =  %f\n",
		K, attemps, term_criteria.maxCount, value);
	CHECK(labels_.rows == samples_number);
	int* p2 = reinterpret_cast<int*>(labels_.data);
	for (int i = 1; i <= every_class_number; ++i) {
		for (int j = 0; j < categories_number; ++j) {
			fprintf(stdout, "  %d  ", *p2++);
		}
		fprintf(stdout, "\n");
	}

	return 0;
}