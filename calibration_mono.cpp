/**
 * @file calibration_mono.cpp
 * @brief 张正友相机标定opencv实现
 * @author GWH
 * @version 0.1
 * @date 2020-05-04 16:53:55
 */
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>

using namespace std;
using namespace cv;

class CameraCalibration {
   private:
    string data_root_path_;    // 保存棋盘格的文件夹路径
    Size board_size_;          // 标定板上每行、列的角点数
    Size2f true_square_size_;  // 实际测量得到的标定板上每个棋盘格的大小(单位：mm)

    Size image_size_;                  // 棋盘格图像的像素大小
    static int num_of_sample_images_;  // 可用的棋盘格标定图像
    vector<string> filesPath_;         // 可用的棋盘格标定图像的路径

    static Mat cameraMatrix_;                         // 相机内参数矩阵
    static Mat distCoeffs_;                           // 相机畸变系数k1、k2、p1、p2、k3
    static vector<Mat> tvecsMat_;                     // 每幅图像的旋转向量
    static vector<Mat> rvecsMat_;                     // 每幅图像的平移向量
    vector<vector<Point2f>> image_points_;            // 每一副标定板上的角点坐标
    vector<vector<Point2f>> reproject_image_points_;  // 根据标定的摄像机参数重新投影的像素坐标
    vector<vector<Point3f>> object_points_;           // 每一副标定板上角点的真值三维坐标

    // 视觉测量相关变量
    static Mat sampleImage_;                          // 带有尺子的那张图像，28张里的最后一张

   public:
    /**
     * @brief 构造函数，函数体内手动更改一下需要设定的参数，不传递参数了
     */
    CameraCalibration() {
        data_root_path_   = "/media/gwh/学习资料/课件/研究生课程/计算机视觉测量与导航/calibrate";
        board_size_       = Size(8, 6);
        true_square_size_ = Size2f(27.7667, 27.7667);

        cameraMatrix_         = Mat(3, 3, CV_64FC1, Scalar::all(0));
        distCoeffs_           = Mat(1, 5, CV_64FC1, Scalar::all(0));
        num_of_sample_images_ = 0;
    }

    /**
     * @brief 读取所有标定图像的路径，只保存左眼图像的路径
     *
     * @param rootPath 标定图像所在的文件夹
     * @param filePath 所有左眼图像的路径
     */
    vector<string> GetFileNameFromDir() {
        vector<string> filesPath;
        int len = data_root_path_.length();
        boost::filesystem::path dir(data_root_path_.c_str());
        if (boost::filesystem::exists(dir))  // 判断路径是否存在
        {
            boost::filesystem::directory_iterator itEnd;
            boost::filesystem::directory_iterator itDir(dir);
            string fileName("");
            for (; itDir != itEnd; itDir++)  // 遍历路径下所有文件
            {
                fileName = itDir->path().string();
                if (boost::filesystem::is_directory(fileName.c_str()))  // 判断文件是否是文件夹
                    continue;
                if (fileName.at(len + 1) == 'L') {
                    filesPath.push_back(fileName);
                }
            }
        }
        return std::move(filesPath);
    }

    /**
     * @brief 读取一张图像，提取角点，对角点亚像素精确化
     *
     * @param imagePath 图像路径
     */
    bool FeatureExtraction(string& imagePath, vector<cv::Point2f>& image_points) {
        vector<Point2f> image_points_buf;
        Mat image = imread(imagePath);
        if (num_of_sample_images_ == 0) {
            image_size_.width  = image.cols;
            image_size_.height = image.rows;
            cout << "image_size.width = " << image_size_.width << endl;
            cout << "image_size.height = " << image_size_.height << endl;
        }
        if (findChessboardCorners(image, board_size_, image_points_buf) == 0) {
            cout << "can't find chessboard corners!" << endl;
            return false;
        } else {
            ++num_of_sample_images_;
            Mat image_gray;
            cvtColor(image, image_gray, CV_RGB2GRAY);  // 转换为灰度图

            // 亚像素精确化
            // image_points_buf 初始的角点坐标向量，同时作为亚像素坐标位置的输出
            // Size(5,5) 搜索窗口大小的一半
            // （-1，-1）表示没有死区
            // TermCriteria 角点的迭代过程的终止条件, 可以为迭代次数和角点精度两者的组合
            cornerSubPix(image_gray, image_points_buf, Size(5, 5), Size(-1, -1),
                         TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            image_points = std::move(image_points_buf);

            // 显示角点位置
            drawChessboardCorners(image, board_size_, image_points, true);
            namedWindow("Camera Calibration", WINDOW_AUTOSIZE);
            imshow("Camera Calibration", image);
            waitKey(10);
            return true;
        }
    }

    void Calibration() {
        int i, j, t;
        // 棋盘格角点是逐行，从左边到右边存储的，世界坐标系与此对应
        for(t = 0; t<num_of_sample_images_; ++t){
            vector<Point3f> tempPoints;
            for (j = 0; j < board_size_.height; ++j) {
                for (i = 0; i < board_size_.width; ++i) {
                    Point3f realPoint;
                    // 假定标定板放在世界坐标系中z=0的平面上
                    realPoint.x = i*true_square_size_.width;
                    realPoint.y = j*true_square_size_.height;
                    realPoint.z = 0;
                    tempPoints.push_back(realPoint);
                }
            }
            object_points_.emplace_back(tempPoints);
        }
        vector<int> corner_counts; // 每一幅图像中角点的数量
        for(auto eachImagePoints:image_points_){
            corner_counts.push_back(eachImagePoints.size());
        }
        /* 开始标定 */
        // object_points_ 世界坐标系中的角点的三维坐标
        // image_points_ 每一个内角点对应的图像坐标点
        // image_size_ 图像的像素尺寸大小
        // cameraMatrix_ 输出，内参矩阵
        // distCoeffs_ 输出，畸变系数
        // rvecsMat_ 输出，旋转向量
        // tvecsMat_ 输出，位移向量
        // 0 标定时所采用的算法
        calibrateCamera(object_points_, image_points_, image_size_, cameraMatrix_, distCoeffs_, rvecsMat_, tvecsMat_,0);
        return;
    }

    void Compute() {
        /*读取图像的绝对路径*/
        filesPath_ = GetFileNameFromDir();
        cout << "Totally read " << filesPath_.size() << " images" << endl;

        /*棋盘格提取角点*/
        for (auto it = filesPath_.begin(); it != filesPath_.end(); ++it) {
            vector<Point2f> image_corner;
            if (FeatureExtraction(*it, image_corner)) {
                image_points_.push_back(image_corner);
            }else{
                it = filesPath_.erase(it);
                --it;
            }
        }
        destroyWindow("Camera Calibration");
        cout << image_points_.size() << " images successfully extract feature!" << endl;

        /*标定*/
        cout<<"Start Calibration."<<endl;
        Calibration();
        cout<<"Calibration finished!"<<endl; 
        cout<<"IntrinsicMatrix: \n"<<cameraMatrix_<<endl;
        cout<<"Distortion：\n"<<distCoeffs_<<endl;
        return;
    }

    void AccuracyValuation(){
        double err       = 0.0;  // 每幅图像的平均误差
        double total_err = 0.0;  // 所有图像的平均误差总和
        reproject_image_points_.resize(num_of_sample_images_);
        cout << "Calibration error of each image: " << endl;
        for (int i = 0; i < num_of_sample_images_; ++i) {
            projectPoints(object_points_[i], rvecsMat_[i], tvecsMat_[i], cameraMatrix_, distCoeffs_, reproject_image_points_[i]);
            // 计算新的投影点和旧的投影点之间的误差
            auto& trueImagePoint       = image_points_[i];
            auto& reprojectImagePoint  = reproject_image_points_[i];
            Mat trueImagePointMat      = Mat(1, trueImagePoint.size(), CV_32FC2);
            Mat reprojectImagePointMat = Mat(1, reprojectImagePoint.size(), CV_32FC2);
            for(size_t j = 0; j<trueImagePoint.size(); ++j){
                trueImagePointMat.at<Vec2f>(0,j) = Vec2f(trueImagePoint[j].x,trueImagePoint[j].y);
                reprojectImagePointMat.at<Vec2f>(0,j) = Vec2f(reprojectImagePoint[j].x,reprojectImagePoint[j].y);
            }
            err = norm(trueImagePointMat, reprojectImagePointMat, NORM_L2);
            err /= trueImagePoint.size();
            total_err += err / num_of_sample_images_;
            cout<<"The "<<i+1<<" image reprojection error is "<<err<<endl;
        }
        cout<<"Total reprojection error is "<<total_err<<endl;
        return;
    }

    void ShowUndistortion(){
        // 图像在x,y方向上的映射关系，必须是CV_32FC1类型的
        Mat mapx = Mat(image_size_, CV_32FC1);
        Mat mapy = Mat(image_size_, CV_32FC1);
        Mat R = Mat::eye(3,3,CV_32F);
        for(int i = 0; i< num_of_sample_images_; ++i){
            // 获得去掉畸变之后的映射关系
            cv::initUndistortRectifyMap(cameraMatrix_,distCoeffs_,R,cameraMatrix_,image_size_,CV_32FC1,mapx,mapy);
            Mat imageSource = imread(filesPath_[i]);
            Mat imageDistortion = imageSource.clone();
            Mat gap_image_3(10, imageSource.cols, CV_8UC3, cv::Scalar(255, 255, 255));
            vconcat(imageSource, gap_image_3, gap_image_3);
            cv::remap(imageSource,imageDistortion,mapx,mapy,INTER_LINEAR);
            vconcat(gap_image_3,imageDistortion,gap_image_3);
            namedWindow("Undistortion image", 0);
            imshow("Undistortion image", gap_image_3);
            waitKey(200);
        }
        cv::destroyWindow("Undistortion image");
        return;
    }

    void monoMeasurement(){
        sampleImage_ = imread(filesPath_[num_of_sample_images_-1]);
        namedWindow("monoMeasurement", WINDOW_AUTOSIZE);
        cv::setMouseCallback("monoMeasurement",on_mouse,0);
        waitKey(0);
    }

    static void on_mouse(int event,int x, int y, int flages, void* ustc){
        static char temp[16];
        char show_distance[1024];
        static char temp_1[16];
        static int count = 0;
        static Point2f firstPoint;
        static Point2f secondPoint;
        static Mat temp_image = sampleImage_.clone();
        Mat temp_image_2 = sampleImage_.clone();
        if (event == CV_EVENT_LBUTTONDOWN) {
            ++count;
            if (count == 1) {
                sprintf(temp, "(%d,%d)", x, y);
                firstPoint = Point2f(x, y);
                putText(temp_image, temp, firstPoint, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
                cv::circle(temp_image, firstPoint, 2, cv::Scalar(255, 0, 255), 2);
            } else {
                temp_image = sampleImage_.clone();
                sprintf(temp, "(%d,%d)", x, y);
                firstPoint = Point2f(x, y);
                putText(temp_image, temp, firstPoint, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
                circle(temp_image, firstPoint, 3, cv::Scalar(255, 0, 255), 3);
                putText(temp_image, temp_1, secondPoint, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
                circle(temp_image, secondPoint, 3, cv::Scalar(255, 0, 255), 3);
                // 计算距离
                double distance;
                int it = num_of_sample_images_-1;
                distance = distanceInWorld(secondPoint,firstPoint,cameraMatrix_,distCoeffs_,rvecsMat_[it],tvecsMat_[it]);
                sprintf(show_distance,"The distance of these two poins is %f mm",distance);
                Mat gap_image(42, temp_image.cols, CV_8UC3, cv::Scalar(255, 255, 255));
                putText(gap_image, show_distance, Point2f(5, 34), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0),2);
                vconcat(temp_image,gap_image, temp_image);
                line(temp_image, secondPoint, firstPoint, cv::Scalar(255, 0, 0), 3, 8, 0);
            }
            secondPoint = firstPoint;
            for(int i = 0; i<16 ; ++i){
                temp_1[i] = temp[i];
            }
            imshow("monoMeasurement", temp_image);
        }
        if (event == CV_EVENT_MOUSEMOVE) {
            temp_image_2 = temp_image.clone();
            sprintf(temp, "(%d,%d)", x, y);
            firstPoint = Point2f(x, y);
            putText(temp_image_2, temp, firstPoint, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
            cv::circle(temp_image_2, firstPoint, 2, cv::Scalar(255, 0, 255), 2);
            imshow("monoMeasurement", temp_image_2);
        }
    }

    /**
     * @brief 计算同一平面上特征点的距离
     *
     * @param p1_uv 像素点1
     * @param p2_uv 像素点2
     * @param M     相机内参
     * @param rvec  该图的旋转矩阵
     * @param tvrc  该图的平移矩阵
     * @param K     相机的畸变系数矩阵
     *
     * @return 世界坐标系下的距离(mm)
     */
    static double distanceInWorld(Point2f p1_uv, Point2f p2_uv, Mat M, Mat K, Mat rvec, Mat tvrc){
        Mat R;
        cv::Rodrigues(rvec,R);
        cout<<"测量图像的旋转矩阵：\n"<<R<<endl;
        cout<<"测量图像的平移：\n"<<tvrc<<endl;
        // 对像素点去畸变
        Mat puv_1_2(1,1,CV_64FC2,Scalar(p1_uv.x,p1_uv.y));
        Mat puv_2_2(1,1,CV_64FC2,Scalar(p2_uv.x,p2_uv.y));
        undistortPoints(puv_1_2, puv_1_2, M, K, Mat::eye(3, 3, CV_64FC1), M);
        undistortPoints(puv_2_2, puv_2_2, M, K, Mat::eye(3, 3, CV_64FC1), M);

        // 齐次像素坐标
        Mat puv_1_3 = (cv::Mat_<double>(3, 1) << puv_1_2.at<Vec2d>(0,0).val[0], puv_1_2.at<Vec2d>(0,0).val[1],1);
        Mat puv_2_3 = (cv::Mat_<double>(3, 1) << puv_2_2.at<Vec2d>(0,0).val[0], puv_2_2.at<Vec2d>(0,0).val[1],1);
        
        // 矩阵[r1,r2,t]
        Mat R_33(3,3,CV_64FC1); 
        R.col(0).copyTo(R_33.col(0));
        R.col(1).copyTo(R_33.col(1));
        tvrc.copyTo(R_33.col(2));

        // 归一化相机平面坐标
        Mat pxy_1 = M.inv() * puv_1_3;
        Mat pxy_2 = M.inv() * puv_2_3;

        // 转换为世界坐标系下的坐标
        Mat pW_1 = R_33.inv() * pxy_1;
        Mat pW_2 = R_33.inv() * pxy_2;
        pW_1 = pW_1 / pW_1.at<double>(2, 0);
        pW_2 = pW_2 / pW_2.at<double>(2, 0);

        cout<<"点1在世界坐标系下的坐标是："<<pW_1.t()<<endl;
        cout<<"点2在世界坐标系下的坐标是："<<pW_2.t()<<endl;
        double distance = norm(pW_1, pW_2, NORM_L2);
        cout<<"距离是："<<distance<<" mm"<<endl;
        return distance;
    }
};

Mat CameraCalibration::sampleImage_;
Mat CameraCalibration::cameraMatrix_;      
Mat CameraCalibration::distCoeffs_;        
int CameraCalibration::num_of_sample_images_;
vector<Mat> CameraCalibration::tvecsMat_;  
vector<Mat> CameraCalibration::rvecsMat_;  

int main(int argc, char** argv) {
    CameraCalibration cameraCalibration;
    cameraCalibration.Compute();
    cameraCalibration.AccuracyValuation();
    cameraCalibration.ShowUndistortion();
    cameraCalibration.monoMeasurement();
    return 0;
}
