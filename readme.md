### 张正友相机标定——OpenCv实现

#### 1、本机运行环境：

1. Boost 1.5.4
2. OpenCV 3.1.0

#### 2、运行指令：

mkdir build  
cd build  
cmake ..  
make -j4  

#### 3、代码修改

请手动修改CameraCalibration的构造函数：
```C++
    CameraCalibration() {
        data_root_path_   = "/media/gwh/学习资料/课件/研究生课程/计算机视觉测量与导航/calibrate";
        board_size_       = Size(8, 6);
        true_square_size_ = Size2f(27.7667, 27.7667);

        cameraMatrix_     = Mat(3, 3, CV_32FC1, Scalar::all(0));
        distCoeffs_       = Mat(1, 5, CV_32FC1, Scalar::all(0));
        num_of_sample_images_ = 0;
    }
```
1. data_root_path_：标定图像所在的文件夹路径
2. board_size_：标定板中角点的数量
3. true_square_size_：标定板实际方格的尺寸
  
注意：本程序标定的是小觅相机，左目和右目拍摄的图像放在一个文件夹中，左目图像文件名以"L"开头，右目图像文件名以"R"开头。

#### 4、标定原理
标定原理在我的博客，欢迎访问：
https://gutsgwh1997.github.io/2020/05/19/%E5%BC%A0%E6%AD%A3%E5%8F%8B%E7%9B%B8%E6%9C%BA%E6%A0%87%E5%AE%9A%E6%B3%95/
