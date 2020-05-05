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

using namespace std;
using namespace cv;

class CameraCalibration {
   private:
    string data_root_path_;    // 保存棋盘格的文件夹路径
    Size board_size_;          // 标定板上每行、列的角点数
    Size2f true_square_size_;  // 实际测量得到的标定板上每个棋盘格的大小(单位：mm)

    Size image_size_;           // 棋盘格图像的像素大小
    int num_of_sample_images_;  // 可用的棋盘格标定图像
    vector<string> filesPath_;  // 可用的棋盘格标定图像的路径

    Mat cameraMatrix_;                                // 相机内参数矩阵
    Mat distCoeffs_;                                  // 相机畸变系数k1、k2、p1、p2、k3
    vector<Mat> tvecsMat_;                            // 每幅图像的旋转向量
    vector<Mat> rvecsMat_;                            // 每幅图像的平移向量
    vector<vector<Point2f>> image_points_;            // 每一副标定板上的角点坐标
    vector<vector<Point2f>> reproject_image_points_;  // 根据标定的摄像机参数重新投影的像素坐标
    vector<vector<Point3f>> object_points_;           // 每一副标定板上角点的真值三维坐标

   public:
    /**
     * @brief 构造函数，函数体内手动更改一下需要设定的参数，不传递参数了
     */
    CameraCalibration() {
        data_root_path_   = "/media/gwh/学习资料/课件/研究生课程/计算机视觉测量与导航/calibrate";
        board_size_       = Size(8, 6);
        true_square_size_ = Size2f(27.7667, 27.7667);

        cameraMatrix_     = Mat(3, 3, CV_32FC1, Scalar::all(0));
        distCoeffs_       = Mat(1, 5, CV_32FC1, Scalar::all(0));
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
            waitKey(100);

            return true;
        }
    }

    void Calibration() {
        int i, j, t;
        for(t = 0; t<num_of_sample_images_; ++t){
            vector<Point3f> tempPoints;
            for(i = 0; i<board_size_.height; ++i){
                for (j = 0; j<board_size_.width; ++j){
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
            waitKey(1000);
        }
        return;
    }
};

int main(int argc, char** argv) {
    CameraCalibration cameraCalibration;
    cameraCalibration.Compute();
    cameraCalibration.AccuracyValuation();
    cameraCalibration.ShowUndistortion();
    return 0;
}
