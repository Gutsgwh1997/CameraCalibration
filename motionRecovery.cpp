#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

double distanceInWorld(Point2f p1_uv, Point2f p2_uv, Mat M, Mat R, Mat tvrc);
void ORBFeatureMatches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2, std::vector<DMatch>& matches);

int main(){
    // Read Images
    Mat image1 = imread("/media/gwh/学习资料/课件/研究生课程/计算机视觉测量与导航/calibrate/Left025.jpg");
    Mat image2 = imread("/media/gwh/学习资料/课件/研究生课程/计算机视觉测量与导航/calibrate/Left026.jpg");
    Mat gap_image(image1.rows, 20, CV_8UC3, cv::Scalar(255, 255, 255));
    hconcat(image1, gap_image, gap_image);
    hconcat(gap_image, image2,gap_image);
    namedWindow("rawImage", 0);
    imshow("rawImage", gap_image);
    // waitKey(2000);

    // Set IntrinsMatrix
    Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<697.86293,0,627.0078,0,698.835,342.57,0,0,1);
    Mat distCoeffs = (cv::Mat_<double>(1, 5) <<-0.33459478,0.154093,3.0875912e-6,-0.0008864374,-0.04151643613);
    cout << "CameraMatrix: \n" << cameraMatrix << endl;
    cout << "DistCoeffs: " << distCoeffs << endl;

    // Undistort
    Size image_size(Size(image1.cols,image1.rows));
    Mat mapx = Mat(image_size, CV_32FC1);
    Mat mapy = Mat(image_size, CV_32FC1);
    Mat R = Mat::eye(3, 3, CV_32F);
    initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
    remap(image1, image1, mapx, mapy, INTER_LINEAR);
    remap(image2, image2, mapx, mapy, INTER_LINEAR);
    Mat gap_image_1(image1.rows, 20, CV_8UC3, cv::Scalar(255, 255, 255));
    hconcat(image1, gap_image_1, gap_image_1);
    hconcat(gap_image_1, image2,gap_image_1);
    namedWindow("Undistortion image", 0);
    imshow("Undistortion image", gap_image_1);
    // waitKey(2000);

    // Feature Extract
    vector<Point2f> image_points_1;
    vector<Point2f> image_points_2;
    Size board_size = Size(8,6);
    findChessboardCorners(image1, board_size, image_points_1);
    findChessboardCorners(image2, board_size, image_points_2);
    Mat image_gray;
    cvtColor(image1, image_gray, CV_RGB2GRAY);
    cornerSubPix(image_gray, image_points_1, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    cvtColor(image2, image_gray, CV_RGB2GRAY);
    cornerSubPix(image_gray, image_points_2, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    // TODO:Feature Matching
    // 提取的是棋盘格角点，默认去畸变后的棋盘格完整，则两张图的角点天然是匹配好的
    drawChessboardCorners(image1, board_size, image_points_1, true);
    drawChessboardCorners(image2, board_size, image_points_2, true);
    Mat gap_image_3(image1.rows, 20, CV_8UC3, cv::Scalar(255, 255, 255));
    hconcat(image1, gap_image_3, gap_image_3);
    hconcat(gap_image_3, image2, gap_image_3);
    namedWindow("Matching Points", 0);
    imshow("Matching Points", gap_image_3);
    // waitKey(2000);
    // 提取ORB角点作对比
    // vector<KeyPoint> keypoints_1,keypoints_2;
    // vector<DMatch> matches;
    // ORBFeatureMatches(image1,image2,keypoints_1,keypoints_2,matches);
    // Mat img_match;
    // drawMatches(image1, keypoints_1, image2, keypoints_2, matches, img_match);
    // imshow("ORBFeature", img_match);

    // Estimate Essential Matrix
    Mat EMatrix = findEssentialMat(image_points_1, image_points_2, cameraMatrix);
    Mat w, U, Vt;
    SVD::compute(EMatrix, w, U, Vt);
    cout<<"\n-----------------------------------------"<<endl;
    cout << "SVD of Essential Matrix:\n" << w << endl;
    Mat R_12, t;
    recoverPose(EMatrix, image_points_1, image_points_2, cameraMatrix, R_12, t);
    double distance = distanceInWorld(image_points_1[0], image_points_1[1], cameraMatrix, R_12, t);
    cout << "Essential Matrix is:\n" << EMatrix << endl;
    cout << "Rotation is :\n" << R_12 << endl;
    // cout << "Translation is: \n" << 27.7667 / distance * t.t() << " mm" << endl;
    cout << "Translation is: \n" << t.t() << endl;
    cout<<"\n-----------------------------------------"<<endl;

    // Homography Matrix
    Mat homography = findHomography(image_points_1,image_points_2,RANSAC);
    cout<<"\n-----------------------------------------"<<endl;
    cout<<"Homography Matrix is:\n"<<homography<<endl;
    vector<Mat> Rs,ts;
    decomposeHomographyMat(homography, cameraMatrix, Rs, ts, noArray());
    for(int i = 0; i<Rs.size(); ++i){
        cout<<"Rotation_"<<i+1<<":\n"<<Rs[i]<<endl;
        cout<<"Translation_"<<i+1<<":\n"<<ts[i].t()<<endl;
        if (i < Rs.size() - 1) cout << endl;
    }
    cout<<"\n-----------------------------------------"<<endl;
    waitKey(0);
}

/**
 * @brief 计算两个像素点的真实坐标系下的距离
 *
 * @param p1_uv 去畸变后的像点1
 * @param p2_uv 去畸变后的像点2
 * @param M 相机矩阵
 * @param R 当前图的旋转矩阵
 * @param tvrc 当前图的平移矢量
 *
 * @return 3维坐标系下的平面距离
 */
double distanceInWorld(Point2f p1_uv, Point2f p2_uv, Mat M, Mat R, Mat tvrc) {
    // 齐次像素坐标
    Mat puv_1_3 = (cv::Mat_<double>(3, 1) << p1_uv.x, p1_uv.y, 1);
    Mat puv_2_3 = (cv::Mat_<double>(3, 1) << p2_uv.x, p2_uv.y, 1);

    // 矩阵[r1,r2,t]
    Mat R_33(3, 3, CV_64FC1);
    R.col(0).copyTo(R_33.col(0));
    R.col(1).copyTo(R_33.col(1));
    tvrc.copyTo(R_33.col(2));

    // 归一化相机平面坐标
    Mat pxy_1 = M.inv() * puv_1_3;
    Mat pxy_2 = M.inv() * puv_2_3;

    // 转换为世界坐标系下的坐标
    Mat pW_1 = R_33.inv() * pxy_1;
    Mat pW_2 = R_33.inv() * pxy_2;
    pW_1     = pW_1 / pW_1.at<double>(2, 0);
    pW_2     = pW_2 / pW_2.at<double>(2, 0);

    // cout << "点1在世界坐标系下的坐标是：" << pW_1.t() << endl;
    // cout << "点2在世界坐标系下的坐标是：" << pW_2.t() << endl;
    double distance = norm(pW_1, pW_2, NORM_L2);
    // cout << "距离是：" << distance << " mm" << endl;
    return distance;
}

void ORBFeatureMatches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2, std::vector<DMatch>& matches) {
    // 初始化
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector       = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher      = DescriptorMatcher::create("BruteForce-Hamming");
    // 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    // 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    // 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}
