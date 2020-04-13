// ConsoleApplication16.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "ANNKmeans.h"
#include <algorithm>
#include <limits>
#include<random>
#include "common.h"

namespace ANN {

	namespace {

		template<typename T>
		void generate_random_center(const std::vector<std::vector<T>>& box, std::vector<T>& center)
		{
			std::random_device rd;
			std::mt19937 generator(rd());
			std::uniform_real_distribution<T> distribution((T)0, (T)0.0001);

			int dims = box.size();
			T margin = 1.f / dims;
			for (int j = 0; j < dims; j++) {
				center[j] = (distribution(generator) * (1. + margin * 2.) - margin) * (box[j][1] - box[j][0]) + box[j][0];
			}
		}

		template<typename T>
		inline T norm_L2_Sqr(const T* a, const T* b, int n)
		{
			double s = 0.f;
			for (int i = 0; i < n; i++) {
				double v = double(a[i] - b[i]);
				s += v * v;
			}
			return s;
		}

		template<typename T>
		void distance_computer(std::vector<double>& distances, std::vector<int>& labels, const std::vector<std::vector<T>>& data,
			const std::vector<std::vector<T>>& centers, bool only_distance = false)
		{
			const int K = centers.size();
			const int dims = centers[0].size();

			for (int i = 0; i < distances.size(); ++i) {
				const std::vector<T> sample = data[i];

				if (only_distance) {
					const std::vector<T> center = centers[labels[i]];
					distances[i] = norm_L2_Sqr(sample.data(), center.data(), dims);
					continue;
				}

				int k_best = 0;
				double min_dist = std::numeric_limits<double>::max(); // DBL_MAX

				for (int k = 0; k < K; ++k) {
					const std::vector<T> center = centers[k];
					const double dist = norm_L2_Sqr(sample.data(), center.data(), dims);

					if (min_dist > dist) {
						min_dist = dist;
						k_best = k;
					}
				}

				distances[i] = min_dist;
				labels[i] = k_best;
			}
		}

	} // namespace

	template<typename T>
	int kmeans(const std::vector<std::vector<T>>& data, int K, std::vector<int>& best_labels, std::vector<std::vector<T>>& centers, double& compactness_measure,
		int max_iter_count, double epsilon, int attempts, int flags)
	{
		CHECK(flags == KMEANS_RANDOM_CENTERS);
		//获取输入矩阵长度
		int N = data.size();
		//矩阵长度需要大过指定聚类时划分为几类；
		CHECK(K > 0 && N >= K);

		//指第一行的列数
		int dims = data[0].size();
		//理想初始聚类中心
		attempts = std::max(attempts, 1);
		//分配输出矩阵的长度
		best_labels.resize(N);
		//定义中间矩阵label
		std::vector<int> labels(N);

		//输出最终的均值点的矩阵
		centers.resize(K);
		std::vector<std::vector<T>> centers_(K), old_centers(K);
		//初始化temp为输入矩阵的第一行列数大小，全为0.0
		std::vector<T> temp(dims, (T)0.);
		//初始化三个矩阵为聚类行+第一行列数大小的矩阵
		for (int i = 0; i < K; ++i) {
			centers[i].resize(dims);
			centers_[i].resize(dims);
			old_centers[i].resize(dims);
		}

		//最大值
		compactness_measure = std::numeric_limits<double>::max(); // DBL_MAX
		//初始化为0.0
		double compactness = 0.;


		epsilon = std::max(epsilon, (double)0.);
		epsilon *= epsilon;

		max_iter_count = std::min(std::max(max_iter_count, 2), 100);

		if (K == 1) {
			attempts = 1;
			max_iter_count = 2;
		}

		std::vector<std::vector<T>> box(dims);
		for (int i = 0; i < dims; ++i) {
			box[i].resize(2);
		}

		std::vector<double> dists(N, 0.);
		std::vector<int> counters(K);

		const T* sample = data[0].data();
		for (int i = 0; i < dims; ++i) {
			box[i][0] = sample[i];
			box[i][1] = sample[i];
		}

		for (int i = 1; i < N; ++i) {
			sample = data[i].data();

			for (int j = 0; j < dims; ++j) {
				T v = sample[j];
				box[j][0] = std::min(box[j][0], v);
				box[j][1] = std::max(box[j][1], v);
			}
		}

		for (int a = 0; a < attempts; ++a) {
			double max_center_shift = std::numeric_limits<double>::max(); // DBL_MAX

			for (int iter = 0;;) {
				centers_.swap(old_centers);

				if (iter == 0 && (a > 0 || true)) {
					for (int k = 0; k < K; ++k) {
						generate_random_center(box, centers_[k]);
					}
				}
				else {
					// compute centers
					for (auto& center : centers_) {
						std::for_each(center.begin(), center.end(), [](T& v) {v = (T)0; });
					}

					std::for_each(counters.begin(), counters.end(), [](int& v) {v = 0; });

					for (int i = 0; i < N; ++i) {
						sample = data[i].data();
						int k = labels[i];
						auto& center = centers_[k];

						for (int j = 0; j < dims; ++j) {
							center[j] += sample[j];
						}
						counters[k]++;
					}

					if (iter > 0) max_center_shift = 0;

					for (int k = 0; k < K; ++k) {
						if (counters[k] != 0) continue;

						// if some cluster appeared to be empty then:
						//   1. find the biggest cluster
						//   2. find the farthest from the center point in the biggest cluster
						//   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
						int max_k = 0;
						for (int k1 = 1; k1 < K; ++k1) {
							if (counters[max_k] < counters[k1])
								max_k = k1;
						}

						double max_dist = 0;
						int farthest_i = -1;
						auto& new_center = centers_[k];
						auto& old_center = centers_[max_k];
						auto& _old_center = temp; // normalized
						T scale = (T)1.f / counters[max_k];
						for (int j = 0; j < dims; j++) {
							_old_center[j] = old_center[j] * scale;
						}

						for (int i = 0; i < N; ++i) {
							if (labels[i] != max_k)
								continue;
							sample = data[i].data();
							double dist = norm_L2_Sqr(sample, _old_center.data(), dims);

							if (max_dist <= dist) {
								max_dist = dist;
								farthest_i = i;
							}
						}

						counters[max_k]--;
						counters[k]++;
						labels[farthest_i] = k;
						sample = data[farthest_i].data();

						for (int j = 0; j < dims; ++j) {
							old_center[j] -= sample[j];
							new_center[j] += sample[j];
						}
					}

					for (int k = 0; k < K; ++k) {
						auto& center = centers_[k];
						CHECK(counters[k] != 0);

						T scale = (T)1.f / counters[k];
						for (int j = 0; j < dims; ++j) {
							center[j] *= scale;
						}

						if (iter > 0) {
							double dist = 0;
							const auto old_center = old_centers[k];
							for (int j = 0; j < dims; j++) {
								T t = center[j] - old_center[j];
								dist += t * t;
							}
							max_center_shift = std::max(max_center_shift, dist);
						}
					}
				}

				bool isLastIter = (++iter == std::max(max_iter_count, 2) || max_center_shift <= epsilon);

				// assign labels
				std::for_each(dists.begin(), dists.end(), [](double& v) {v = 0; });

				distance_computer(dists, labels, data, centers_, isLastIter);
				std::for_each(dists.cbegin(), dists.cend(), [&compactness](double v) { compactness += v; });

				if (isLastIter) break;
			}

			if (compactness < compactness_measure) {
				compactness_measure = compactness;
				for (int i = 0; i < K; ++i) {
					memcpy(centers[i].data(), centers_[i].data(), sizeof(T) * dims);
				}
				memcpy(best_labels.data(), labels.data(), sizeof(int) * N);
			}
		}

		return 0;
	}

	template int kmeans<float>(const std::vector<std::vector<float>>&, int K, std::vector<int>&, std::vector<std::vector<float>>&, double&,
		int max_iter_count, double epsilon, int attempts, int flags);
	template int kmeans<double>(const std::vector<std::vector<double>>&, int K, std::vector<int>&, std::vector<std::vector<double>>&, double&,
		int max_iter_count, double epsilon, int attempts, int flags);

} // namespace ANN
//double cv::kmeans( InputArray _data, int K,
//                   InputOutputArray _bestLabels,
//                   TermCriteria criteria, int attempts,
//                   int flags, OutputArray _centers )
//{
//    CV_INSTRUMENT_REGION();
//    const int SPP_TRIALS = 3;
//    Mat data0 = _data.getMat();
//    const bool isrow = data0.rows == 1;
//    const int N = isrow ? data0.cols : data0.rows;
//    const int dims = (isrow ? 1 : data0.cols)*data0.channels();
//    const int type = data0.depth();
//
//    attempts = std::max(attempts, 1);
//    CV_Assert( data0.dims <= 2 && type == CV_32F && K > 0 );
//    CV_CheckGE(N, K, "Number of clusters should be more than number of elements");
//
//    Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));
//
//    _bestLabels.create(N, 1, CV_32S, -1, true);
//
//    Mat _labels, best_labels = _bestLabels.getMat();
//    if (flags & CV_KMEANS_USE_INITIAL_LABELS)
//    {
//        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
//                  best_labels.cols*best_labels.rows == N &&
//                  best_labels.type() == CV_32S &&
//                  best_labels.isContinuous());
//        best_labels.reshape(1, N).copyTo(_labels);
//        for (int i = 0; i < N; i++)
//        {
//            CV_Assert((unsigned)_labels.at<int>(i) < (unsigned)K);
//        }
//    }
//    else
//    {
//        if (!((best_labels.cols == 1 || best_labels.rows == 1) &&
//             best_labels.cols*best_labels.rows == N &&
//             best_labels.type() == CV_32S &&
//             best_labels.isContinuous()))
//        {
//            _bestLabels.create(N, 1, CV_32S);
//            best_labels = _bestLabels.getMat();
//        }
//        _labels.create(best_labels.size(), best_labels.type());
//    }
//    int* labels = _labels.ptr<int>();
//
//    Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);
//    cv::AutoBuffer<int, 64> counters(K);
//    cv::AutoBuffer<double, 64> dists(N);
//    RNG& rng = theRNG();
//
//    if (criteria.type & TermCriteria::EPS)
//        criteria.epsilon = std::max(criteria.epsilon, 0.);
//    else
//        criteria.epsilon = FLT_EPSILON;
//    criteria.epsilon *= criteria.epsilon;
//
//    if (criteria.type & TermCriteria::COUNT)
//        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
//    else
//        criteria.maxCount = 100;
//
//    if (K == 1)
//    {
//        attempts = 1;
//        criteria.maxCount = 2;
//    }
//
//    cv::AutoBuffer<Vec2f, 64> box(dims);
//    if (!(flags & KMEANS_PP_CENTERS))
//    {
//        {
//            const float* sample = data.ptr<float>(0);
//            for (int j = 0; j < dims; j++)
//                box[j] = Vec2f(sample[j], sample[j]);
//        }
//        for (int i = 1; i < N; i++)
//        {
//            const float* sample = data.ptr<float>(i);
//            for (int j = 0; j < dims; j++)
//            {
//                float v = sample[j];
//                box[j][0] = std::min(box[j][0], v);
//                box[j][1] = std::max(box[j][1], v);
//            }
//        }
//    }
//
//    double best_compactness = DBL_MAX;
//    for (int a = 0; a < attempts; a++)
//    {
//        double compactness = 0;
//
//        for (int iter = 0; ;)
//        {
//            double max_center_shift = iter == 0 ? DBL_MAX : 0.0;
//
//            swap(centers, old_centers);
//
//            if (iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)))
//            {
//                if (flags & KMEANS_PP_CENTERS)
//                    generateCentersPP(data, centers, K, rng, SPP_TRIALS);
//                else
//                {
//                    for (int k = 0; k < K; k++)
//                        generateRandomCenter(dims, box.data(), centers.ptr<float>(k), rng);
//                }
//            }
//            else
//            {
//                // compute centers
//                centers = Scalar(0);
//                for (int k = 0; k < K; k++)
//                    counters[k] = 0;
//
//                for (int i = 0; i < N; i++)
//                {
//                    const float* sample = data.ptr<float>(i);
//                    int k = labels[i];
//                    float* center = centers.ptr<float>(k);
//                    for (int j = 0; j < dims; j++)
//                        center[j] += sample[j];
//                    counters[k]++;
//                }
//
//                for (int k = 0; k < K; k++)
//                {
//                    if (counters[k] != 0)
//                        continue;
//
//                    // if some cluster appeared to be empty then:
//                    //   1. find the biggest cluster
//                    //   2. find the farthest from the center point in the biggest cluster
//                    //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
//                    int max_k = 0;
//                    for (int k1 = 1; k1 < K; k1++)
//                    {
//                        if (counters[max_k] < counters[k1])
//                            max_k = k1;
//                    }
//
//                    double max_dist = 0;
//                    int farthest_i = -1;
//                    float* base_center = centers.ptr<float>(max_k);
//                    float* _base_center = temp.ptr<float>(); // normalized
//                    float scale = 1.f/counters[max_k];
//                    for (int j = 0; j < dims; j++)
//                        _base_center[j] = base_center[j]*scale;
//
//                    for (int i = 0; i < N; i++)
//                    {
//                        if (labels[i] != max_k)
//                            continue;
//                        const float* sample = data.ptr<float>(i);
//                        double dist = hal::normL2Sqr_(sample, _base_center, dims);
//
//                        if (max_dist <= dist)
//                        {
//                            max_dist = dist;
//                            farthest_i = i;
//                        }
//                    }
//
//                    counters[max_k]--;
//                    counters[k]++;
//                    labels[farthest_i] = k;
//
//                    const float* sample = data.ptr<float>(farthest_i);
//                    float* cur_center = centers.ptr<float>(k);
//                    for (int j = 0; j < dims; j++)
//                    {
//                        base_center[j] -= sample[j];
//                        cur_center[j] += sample[j];
//                    }
//                }
//
//                for (int k = 0; k < K; k++)
//                {
//                    float* center = centers.ptr<float>(k);
//                    CV_Assert( counters[k] != 0 );
//
//                    float scale = 1.f/counters[k];
//                    for (int j = 0; j < dims; j++)
//                        center[j] *= scale;
//
//                    if (iter > 0)
//                    {
//                        double dist = 0;
//                        const float* old_center = old_centers.ptr<float>(k);
//                        for (int j = 0; j < dims; j++)
//                        {
//                            double t = center[j] - old_center[j];
//                            dist += t*t;
//                        }
//                        max_center_shift = std::max(max_center_shift, dist);
//                    }
//                }
//            }
//
//            bool isLastIter = (++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon);
//
//            if (isLastIter)
//            {
//                // don't re-assign labels to avoid creation of empty clusters
//                parallel_for_(Range(0, N), KMeansDistanceComputer<true>(dists.data(), labels, data, centers), (double)divUp((size_t)(dims * N), CV_KMEANS_PARALLEL_GRANULARITY));
//                compactness = sum(Mat(Size(N, 1), CV_64F, &dists[0]))[0];
//                break;
//            }
//            else
//            {
//                // assign labels
//                parallel_for_(Range(0, N), KMeansDistanceComputer<false>(dists.data(), labels, data, centers), (double)divUp((size_t)(dims * N * K), CV_KMEANS_PARALLEL_GRANULARITY));
//            }
//        }
//
//        if (compactness < best_compactness)
//        {
//            best_compactness = compactness;
//            if (_centers.needed())
//            {
//                if (_centers.fixedType() && _centers.channels() == dims)
//                    centers.reshape(dims).copyTo(_centers);
//                else
//                    centers.copyTo(_centers);
//            }
//            _labels.copyTo(best_labels);
//        }
//    }
//
//    return best_compactness;
//}