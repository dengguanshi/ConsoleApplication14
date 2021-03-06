﻿
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iterator>
#include <vector>
#include <map>
#include<fstream>
#include <opencv2\imgproc\types_c.h>
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "func.h"
#include "ConsoleApplication16.cpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace cv::ml;

#define DATA_FOLDER "data/"
#define TRAIN_FOLDER "data/train_images/"
#define TEMPLATE_FOLDER "data/templates/"
#define TEST_FOLDER "data/test_image"
#define RESULT_FOLDER "data/result_image/"

// 移除扩展名，用来讲模板组织成类目
string categorizer::remove_extention(string full_name)
{
    //find_last_of找出字符最后一次出现的地方
    int last_index = full_name.find_last_of(".");
    string name = full_name.substr(0, last_index);
    return name;
}

// 构造函数
categorizer::categorizer(int _clusters)
{
    cout << "开始初始化..." << endl;
    clusters = _clusters;
    //初始化指针
    int minHessian = 400;
    featureDecter = SURF::create(minHessian);
    bowtrainer = new BOWKMeansTrainer(clusters);
    descriptorMacher = BFMatcher::create();
    bowDescriptorExtractor = new BOWImgDescriptorExtractor(featureDecter, descriptorMacher);

    // //boost库文件 遍历数据文件夹  directory_iterator(p)就是迭代器的起点，无参数的directory_iterator()就是迭代器的终点。
    boost::filesystem::directory_iterator begin_iter(TEMPLATE_FOLDER);
    boost::filesystem::directory_iterator end_iter;
    //获取该目录下的所有文件名
    for (; begin_iter != end_iter; ++begin_iter)
    {
        //文件的路径 data/templates/airplanes.jpg
        string filename = string(TEMPLATE_FOLDER) + begin_iter->path().filename().string();
        //文件夹名称 airplanes
        string sub_category = remove_extention(begin_iter->path().filename().string());
        cout << "sub_category" << endl;
        //读入模板图片
        if (begin_iter->path().filename().string() != ".DS_Store") {
            Mat image = imread(filename);
            Mat templ_image;
            //存储原图模板
            result_objects[sub_category] = image;
        }
    }
    cout << "初始化完毕..." << endl;
    //读取训练集
    make_train_set();
}

//构造训练集合
void categorizer::make_train_set()
{
    cout << "读取训练集..." << endl;
    string categor;
    //递归迭代rescursive 直接定义两个迭代器：i为迭代起点（有参数），end_iter迭代终点
    for (boost::filesystem::recursive_directory_iterator i(TRAIN_FOLDER), end_iter; i != end_iter; i++)
    {
        // level == 0即为目录，因为TRAIN__FOLDER中设置如此
        if (i.level() == 0)
        {
            // 将类目名称设置为目录的名称
            if ((i->path()).filename().string() != ".DS_Store") {
                categor = (i->path()).filename().string();
                category_name.push_back(categor);

            }
        }
        else {
            // 读取文件夹下的文件。level 1表示这是一副训练图，通过multimap容器来建立由类目名称到训练图的一对多的映射
            string filename = string(TRAIN_FOLDER) + categor + string("/") + (i->path()).filename().string();

            if ((i->path()).filename().string() != ".DS_Store") {
                Mat temp = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
                pair<string, Mat> p(categor, temp);
                //得到训练集
                train_set.insert(p);
            }

        }


    }
    categories_size = category_name.size();
    cout << "发现 " << categories_size << "种类别物体..." << endl;
}
Mat categorizer::getMyMat(Mat mymat) {


    vector<IPoint> ips1 = surf.GetAllFeatures(mymat);
    std::cout << "ips1" << endl;
    std::cout << ips1.size() << endl;
    Mat test;//行 列
    for (int t = 0; t < ips1.size(); t++) {
        Mat tempmat(1, 64, CV_32F, ips1[t].descriptor);
        test.push_back(tempmat);
        //std::cout << "进入循环" << endl;
        //float* value = ips1[t].descriptor;//读出第i行第j列像素值
        //std::cout << " this->IPoints[t].descriptor" << endl;
        //test.at<float>(0, t) = *value; //将第i行第j列像素值设置为128
    }
    std::cout << "循环外" << endl;
    cout << test.size() << endl;//[1000 x 1]
    return test;
}
Mat Mycluster(const Mat& _descriptors) {
    Mat labels, vocabulary;
    int K{ 4 }, attemps{ 100 };
    int flags = ANN::KMEANS_RANDOM_CENTERS;
    std::vector<int> best_labels;
    double compactness_measure{ 0. };
    const int myK{ 4 }, myattemps{ 100 }, max_iter_count{ 100 };
    const double epsilon{ 0.001 };



    //ANN::kmeans<float>(_descriptors, myK, best_labels, vocabulary, compactness_measure, max_iter_count, epsilon, myattemps, flags);
    return vocabulary;
}
// 训练图片feature聚类，得出词典
void categorizer::bulid_vacab()
{
    FileStorage vacab_fs(DATA_FOLDER "vocab.xml", FileStorage::READ);

    //如果之前已经生成好，就不需要重新聚类生成词典
    if (vacab_fs.isOpened())
    {
        cout << "图片已经聚类，词典已经存在.." << endl;
        vacab_fs.release();
    }
    else
    {
        Mat vocab_descriptors;
        // 对于每一幅模板，提取SURF算子，存入到vocab_descriptors中
        multimap<string, Mat> ::iterator i = train_set.begin();
        for (; i != train_set.end(); i++)
        {
            vector<KeyPoint>kp;
            Mat templ = (*i).second;
            cout << templ.type() << endl;
            templ.convertTo(templ, CV_32F);
            cout << templ.type() << endl;
            vocab_descriptors.push_back(getMyMat(templ));
            cout << "getMyMat" << endl;
            //Mat descrip;
            //featureDecter->detect(templ, kp);
            ////cout << templ << endl;
            //featureDecter->compute(templ, kp, descrip);
            //cout << "descrip" << endl;
            //cout << descrip.type() << endl;//5
            //cout << descrip.size() << endl;//[64 x 20700]
            ////push_back(Mat);在原来的Mat的最后一行后再加几行,元素为Mat时， 其类型和列的数目 必须和矩阵容器是相同的
            //vocab_descriptors.push_back(descrip);
            ////cout << descrip << endl;
            //cout << descrip.size() << endl;//[64 x 20700]
        }
        vocab_descriptors.convertTo(vocab_descriptors, CV_32F);
        //cout << vocab_descriptors << endl;
        cout << "训练图片开始聚类..." << endl;
        //将每一副图的ORB特征加入到bowTraining中去,就可以进行聚类训练了
        // 对ORB描述子进行聚类
        cout << "vocab_descriptors" << endl;
        cout << vocab_descriptors.type() << endl;//5
        cout << vocab_descriptors.size() << endl;//[64 x 21460]列 行
            
            
        vocab = bowtrainer->cluster(vocab_descriptors);
        cout << "聚类完毕，得出词典..." << endl;
        cout << "vocab" << endl;
        cout << vocab.type() << endl;//5

        cout << vocab.size() << endl;//[64 x 1000]
        //以文件格式保存词典
        FileStorage file_stor(DATA_FOLDER "vocab.xml", FileStorage::WRITE);
        file_stor << "vocabulary" << vocab;
        file_stor.release();
    }
}

//构造bag of words
void categorizer::compute_bow_image()
{
    cout << "构造bag of words..." << endl;
    FileStorage va_fs(DATA_FOLDER "vocab.xml", FileStorage::READ);
    //如果词典存在则直接读取
    if (va_fs.isOpened())
    {
        Mat temp_vacab;
        va_fs["vocabulary"] >> temp_vacab;
        bowDescriptorExtractor->setVocabulary(temp_vacab);
        va_fs.release();
    }
    else
    {
        //对每张图片的特征点，统计这张图片各个类别出现的频率，作为这张图片的bag of words
        bowDescriptorExtractor->setVocabulary(vocab);
    }

    //如果bow.txt已经存在说明之前已经训练过了，下面就不用重新构造BOW
    string bow_path = string(DATA_FOLDER) + string("bow.txt");
    boost::filesystem::ifstream read_file(bow_path);
    // //如BOW已经存在，则不需要构造
    if (read_file.is_open())
    {
        cout << "BOW 已经准备好..." << endl;
    }
    else {
        // 对于每一幅模板，提取SURF算子，存入到vocab_descriptors中
        multimap<string, Mat> ::iterator i = train_set.begin();
        for (; i != train_set.end(); i++)
        {
            vector<KeyPoint>kp;
            string cate_nam = (*i).first;
            Mat tem_image = (*i).second;
            Mat imageDescriptor;
            featureDecter->detect(tem_image, kp);
            bowDescriptorExtractor->compute(tem_image, kp, imageDescriptor);
            //push_back(Mat);在原来的Mat的最后一行后再加几行,元素为Mat时， 其类型和列的数目 必须和矩阵容器是相同的
            allsamples_bow[cate_nam].push_back(imageDescriptor);
        }
        //简单输出一个文本，为后面判断做准备
        boost::filesystem::ofstream ous(bow_path);
        ous << "flag";
        cout << "bag of words构造完毕..." << endl;
    }
}

//训练分类器

void categorizer::trainSvm()
{
    cout << "trainSvm" << endl;
    int flag = 0;
    for (int k = 0; k < categories_size; k++)
    {
        string svm_file_path = string(DATA_FOLDER) + category_name[k] + string("SVM.xml");
        cout << svm_file_path << endl;
        FileStorage svm_fil(svm_file_path, FileStorage::READ);
        //判断训练结果是否存在
        if (svm_fil.isOpened())
        {
            svm_fil.release();
            continue;
        }
        else
        {
            flag = -1;
            break;
        }
    }
    //如果训练结果已经存在则不需要重新训练
    if (flag != -1)
    {
        cout << "分类器已经训练完毕..." << endl;
    }
    else

    {
        stor_svms = new Ptr<SVM>[categories_size];

        cout << "训练分类器..." << endl;
        for (int i = 0; i < categories_size; i++)
        {
            //在Mat中，将一个Mat放在另一个Mat中的时候，就可以使用push_back，但是需要保持列相同
            //创建0*category_name.col的mat对象
            Mat tem_Samples(0, allsamples_bow.at(category_name[i]).cols, allsamples_bow.at(category_name[i]).type());
            //0行1列CV_32SC1类型的图像
            Mat responses(0, 1, CV_32SC1);
            //tem_Samples为列数相同的图像
            tem_Samples.push_back(allsamples_bow.at(category_name[i]));
            //创建行数为 训练图像的行，列数为 1，类型为 CV_32SC1 的图像，并将所有元素初始化为值 （1，1，1，1）；
            Mat posResponses(allsamples_bow.at(category_name[i]).rows, 1, CV_32SC1, Scalar::all(1));
            //responses为行数的1
            responses.push_back(posResponses);


            for (map<string, Mat>::iterator itr = allsamples_bow.begin(); itr != allsamples_bow.end(); ++itr)
            {
                if (itr->first == category_name[i]) {
                    continue;
                }
                //---训练图像拼接，遍历的图像在前面，其他图像拼接在后面
                tem_Samples.push_back(itr->second);
                //---训练图像行数总和拼接，遍历的图像在前面，其他图像拼接在后面
                Mat response(itr->second.rows, 1, CV_32SC1, Scalar::all(-1));
                responses.push_back(response);
            }
            //设置训练参数
            stor_svms[i] = SVM::create();
            //CvSVM::C_SVC C类支持向量分类机 n类分组(n \geq 2)，允许用异常值惩罚因子C进行不完全分类。
            stor_svms[i]->setType(SVM::C_SVC);
            //CvSVM::LINEAR 线性内核。没有任何向映射至高维空间，线性区分（或回归）在原始特征空间中被完成，这是最快的选择。K(x_i, x_j) = x_i^T x_j.
            stor_svms[i]->setKernel(SVM::LINEAR);
            //内核函数（POLY/ RBF/ SIGMOID）的参数\gamma。
            stor_svms[i]->setGamma(3);
            //设置/获取SVM训练时迭代终止条件，默认值是
            stor_svms[i]->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
            //用于训练/预测，均使用基类StatModel中的
            stor_svms[i]->train(tem_Samples, ROW_SAMPLE, responses);
            //存储svm
            string svm_filename = string(DATA_FOLDER) + category_name[i] + string("SVM.xml");
            cout << svm_filename.c_str() << endl;
            stor_svms[i]->save(svm_filename.c_str());
        }
        cout << "分类器训练完毕..." << endl;
    }
}


//对测试图片进行分类

void categorizer::category_By_svm()
{
    cout << "物体分类开始..." << endl;
    //输入的灰度图
    Mat gray_pic;
    //Mat threshold_image;
    string prediction_category;
    float curConfidence;

    boost::filesystem::directory_iterator begin_train(TEST_FOLDER);
    boost::filesystem::directory_iterator end_train;

    for (; begin_train != end_train; ++begin_train)
    {

        //获取该目录下的图片名
        string train_pic_name = (begin_train->path()).filename().string();
        string train_pic_path = string(TEST_FOLDER) + string("/") + (begin_train->path()).filename().string();

        //读取图片
        if ((begin_train->path()).filename().string() == ".DS_Store") {
            continue;
        }
        Mat input_pic = imread(train_pic_path);
        cvtColor(input_pic, gray_pic, CV_BGR2GRAY);

        // 提取BOW描述子
        vector<KeyPoint>kp;


        Mat newImage;
        gray_pic.convertTo(newImage, CV_32F);
        cout <<"提取BOW描述子"  << endl;
        Mat test;
        featureDecter->detect(gray_pic, kp);
        bowDescriptorExtractor->compute(gray_pic, kp, test);
     //   cout << gray_pic.size() << endl;
     //   cout << test1.size() << endl;
     //   Surf surf;
	    //Visualize v;
     //   std::cout << "Visualize外" << endl;
     //   cout << input_pic.depth() << endl;
     //   cout << input_pic.type() << endl;
     //   
	    //vector<IPoint> ips1 = surf.GetAllFeatures(newImage);
     //   std::cout << "循环外" << endl;
     //   std::cout << ips1.size() << endl;
     //   Mat test;//行 列

     //   for (int t = 0; t < ips1.size(); t++) {
     //       //Mat tempmat(1, 64, CV_32F, ips1[t].descriptor);
     //       //test.push_back(tempmat);
     //       std::cout << "进入循环" << endl;
     //       float *value = ips1[t].descriptor;//读出第i行第j列像素值
     //       std::cout << " this->IPoints[t].descriptor" << endl;
     //       test.at<float>(0, t) = *value; //将第i行第j列像素值设置为128
     //   }
     //   std::cout << "循环外" << endl;
     //    cout << test.size() << endl;//[1000 x 1]

        //Mat a;
        //test.col(0).copyTo(a.col(0));


       //// step1(i) :每一维元素的通道数
       // cout << test.step1(0) << endl;//1000
       // cout << test.step1(1) << endl;//1
       // cout << test.step1(2) << endl;//3689348814741910323
       // //    step[i] : 每一维元素的大小，单位字节
       // cout << test.step[0] << endl;//4000
       // cout << test.step[1] << endl;//4
       // cout << test.step[2] << endl;//14757395258967641292
       //  //   size[i] : 每一维元素的个数
       // cout << test.size[0] << endl;//1
       // cout << test.size[1] << endl;//1000
       // cout <<" test.size[2]" << endl;
       // cout << test.size[2] << endl;
        //    elemSize()：每个元素大小，单位字节

         //   elemSize1()：每个通道大小，单位字节


        cout << test.size() << endl;//gray_pic.size=934x933




        int sign = 0;
        float best_score = -2.0f;
        for (int i = 0; i < categories_size; i++)
        {
            string cate_na = category_name[i];
            string f_path = string(DATA_FOLDER) + cate_na + string("SVM.xml");
            FileStorage svm_fs(f_path, FileStorage::READ);
            //读取SVM.xml
            if (svm_fs.isOpened())
            {
                svm_fs.release();
                Ptr<SVM> st_svm = Algorithm::load<SVM>(f_path.c_str());
                if (sign == 0)
                {
                    cout << "进入if循环" << endl;
                    float score_Value = st_svm->predict(test, noArray(), true);
                    float class_Value = st_svm->predict(test, noArray(), false);
                    sign = (score_Value < 0.0f) == (class_Value < 0.0f) ? 1 : -1;
                }
                curConfidence = sign * st_svm->predict(test, noArray(), true);
            }
            else
            {
                if (sign == 0)
                {
                    float scoreValue = stor_svms[i]->predict(test, noArray(), true);
                    float classValue = stor_svms[i]->predict(test, noArray(), false);
                    sign = (scoreValue < 0.0f) == (classValue < 0.0f) ? 1 : -1;
                }
                curConfidence = sign * stor_svms[i]->predict(test, noArray(), true);
            }
            if (curConfidence > best_score)
            {
                best_score = curConfidence;
                prediction_category = cate_na;
            }
        }
        //将图片写入相应的文件夹下
        boost::filesystem::directory_iterator begin_iterater(RESULT_FOLDER);
        boost::filesystem::directory_iterator end_iterator;
        //获取该目录下的文件名
        for (; begin_iterater != end_iterator; ++begin_iterater)
        {

            if (begin_iterater->path().filename().string() == prediction_category)
            {
                string filename = string(RESULT_FOLDER) + prediction_category + string("/") + train_pic_name;
                imwrite(filename, input_pic);
            }
        }
        cout << "这张图属于:" << prediction_category << endl;
    }
}


