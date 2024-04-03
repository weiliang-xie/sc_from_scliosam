#pragma once

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm> 
#include <cstdlib>
#include <memory>
#include <iostream>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

#include "tictoc.h"

using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

using SCPointType = pcl::PointXYZI; // using xyz only. but a user can exchange the original bin encoding function (i.e., max hegiht) to max intensity (for detail, refer 20 ICRA Intensity Scan Context)
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float>;



//椭球模型
class Ellipsoid{
public:
    Ellipsoid(){
        center << 0,0,0;
        axis.resize(3,3);
        cov.resize(3,3);
        axis = Matrix3d::Zero();
        cov = Matrix3d::Zero();
        eigen << 0,0,0;
        point_num = 0;
        mode = -1;
        eloid_vaild = -1;
        num_exit = 0;
    }
    int eloid_vaild;                    //椭球有效标志位 -1：椭球不存在 0：椭球无效  1：椭球有效
    Eigen::Vector3d center;             //椭球中心点
    Eigen::Matrix3d axis;               //椭球轴方向
    Eigen::Vector3d eigen;              //椭球轴长(目前 = 协方差特征值)
    int point_num;                      //用于描绘椭球的点云数
    Eigen::Matrix3d cov;                //协方差
    int mode;                           //椭球类型 -1: 无效 1：线性 2：平面型 3：立体型
    uint num_exit;                      //测试项 各个高度是否存在点云 存在：对应二进制位 == 1 不存在：对应二进制 == 0 范围0~2^9
};

class Frame_Ellipsoid
{
public:
    std::vector<Ellipsoid> ground_voxel_eloid;
    std::vector<Ellipsoid> nonground_voxel_eloid;
};



class BaseGlobalLocalization    //全局算法基类 通用
{
public:
    //data
    std::vector<Frame_Ellipsoid> frame_eloid;

    //lidar
    const double LIDAR_HEIGHT = 2.0;

    //voxel
    const double MAX_RANGE = 80.0;                                                              //体素划分的最远范围，整个点云为160x160m
    const int VOXEL_NUM_HORIZONTAL = 40;                                                        //点云水平方向上的体素数量
    const int VOXEL_NUM_VERTICAL = 40;                                                          //点云垂直方向上的体素数量
    const double VOXEL_UNIT_HORIZANTAL = MAX_RANGE * 2 / double(VOXEL_NUM_HORIZONTAL);          //体素单元水平方向长度
    const double VOXEL_UNIT_VERTICAL = MAX_RANGE * 2 / double(VOXEL_NUM_VERTICAL);              //体素单元垂直方向长度
    /*体素划分说明：以体素左下角的坐标作为体素的index，如（0，0）*/

    //threshold
    const int GROUND_HEIGHT = 0.4;                                                              //0.4为暂取值，仍待验证
    const int NUM_EXCLUDE_FIRST = 30;                                                           //前x个描述符不进行回环检测
    const int NUM_EXCLUDE_RECENT = 100;                                                          //排除相邻的x帧

    const int NUM_CANDIDATES_HASH_ID = 20;                                                      //匹配的候选id数量

    //
    std::vector<std::vector<std::vector<Eigen::Vector3d> > > DivideVoxel(pcl::PointCloud<SCPointType> & _scan_cloud);
    void BulidingEllipsoidModel(std::vector<std::vector<std::vector<Eigen::Vector3d> > > voxel_point);

    //构建椭球
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GetCovarMatrix(std::vector<Eigen::Vector3d> bin_piont);
    std::pair<Eigen::Vector3d,Eigen::Matrix3d> GetEigenvalues(Eigen::MatrixXd bin_cov);
    int ClassifyEllipsoid(Ellipsoid voxeleloid);
    bool IsEllipsoidVaild(Ellipsoid voxeleloid);

    //可视化函数
    
};