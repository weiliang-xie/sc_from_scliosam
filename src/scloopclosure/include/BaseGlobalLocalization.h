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
// #include "utility.h"

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

//评价数据收集
class Evaluate
{
public:
    //loop id 准确率
    std::vector<int> loop_id;
    std::vector<int> inquiry_id;
    std::vector<int> loop_status;

    Evaluate(){

    }
};

//椭球模型与体素特征
class Ellipsoid{
public:
    Ellipsoid(){
        center << 0,0,0;
        max_z_point << 0,0,0;
        axis.resize(3,3);
        cov.resize(3,3);
        axis = Matrix3d::Zero();
        cov = Matrix3d::Zero();
        eigen << 0,0,0;
        point_num = 0;
        mode = -1;
        eloid_vaild = -1;
        num_exit = 0;
        voxel_index = {-1, -1};
    }

    int eloid_vaild;                    //椭球有效标志位 -1：椭球不存在 0：椭球无效  1：椭球有效
    Eigen::Vector3d center;             //椭球中心点
    Eigen::Matrix3d axis;               //椭球轴方向
    Eigen::Vector3d eigen;              //椭球轴长(目前 = 协方差特征值)
    int point_num;                      //用于描绘椭球的点云数
    Eigen::Matrix3d cov;                //协方差
    int mode;                           //椭球类型 -1: 无效 1：线性 2：平面型 3：立体型
    uint num_exit;                      //测试项 各个高度是否存在点云 存在：对应二进制位 == 1 不存在：对应二进制 == 0 范围0~2^9
    Eigen::Vector3d max_z_point;        //最大高度点

    std::pair<int, int> voxel_index;    //体素索引，从0开始，first是x方向，second是y方向
    
};

//体素特征信息 
class VoxelData{
public:
    VoxelData(){
        coordinate << 1000, 1000, 1000;
        length << 0,0,0;
        is_sege_success = false;
    }
    Eigen::Vector3d coordinate;                 //体素的坐标系，即体素坐标值最小的边角点的坐标
    Eigen::Vector3d length;                     //体素尺寸
    std::vector<Eigen::Vector3d> point_cloud;   //点云数据
    bool is_sege_success;                        //体素分割标志
};

//单帧内的体素集合及相关数据
class Frame_Voxeldata{
public:
    Frame_Voxeldata(){
        min_point_z = 1000;
        max_point_z = -1000;
    }
    std::map<int, VoxelData> origin_voxel_data;                 //分割前体素信息，通过x_index * VOXEL_NUM_VERTICAL + y_index 来索引
    std::vector<VoxelData> seg_voxel_data;                      //分割后的体素数据
    double min_point_z;                                         //点云帧中z方向最小坐标
    double max_point_z;                                         //点云帧中z方向最大坐标
};


class Frame_Ellipsoid
{
public:
    std::vector<Ellipsoid> ground_voxel_eloid;
    std::vector<Ellipsoid> nonground_voxel_eloid;
};

#define SEGMENT_VOXEL_ENABLE 0                  //使能自适应分割，使用分割后的体素生成特征

class BaseGlobalLocalization    //全局算法基类 通用
{
public:                                                  
    //data
    std::vector<Frame_Voxeldata> frame_voxel;       //各帧体素信息

    //测试
    std::vector<std::pair<double, double>> frame_seg_ori_num;
    std::vector<std::pair<double, double>> frame_seg_ori_littlevoxel_num;
    std::vector<std::pair<double, double>> frame_seg_ori_validvoxel_num;

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
    const int NUM_EXCLUDE_FIRST = 30;                                                           //满足检索要求的数据库的最小数据帧数量
    const int NUM_EXCLUDE_RECENT = 100;                                                          //排除相邻的x帧

    const int NUM_CANDIDATES_HASH_ID = 200;                                                         //hash匹配的候选id数量
    const int NUM_CANDIDATES_KD_ID = 20;                                                           //kd tree匹配的候选id数量

    const int MIN_VAILD_VOXEL_POINT_NUM = 50;                                                   //有效体素的最小点云数量

    //自适应划分
    const double DIVIDE_MIN_PROJECT_DISTANCE_FROM = 1;                                          //满足分割要求的两点最小映射距离
    const double VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY = 0.1;                              //有效的分割点需要满足的距离边界的阈值要求
    const double MAX_DIVIDE_LINE_BOUNDARY_DISTANCE = 0.08;                                      //用于判断分割线附近点云存在情况的最大范围距离
    const int DIVIDE_LINE_BOUNDARY_MAX_NUM = 10;                                                //满足分割要求的分割线附近范围内最多允许存在的点云数量

    //建立描述符特征
    void DivideVoxel(pcl::PointCloud<SCPointType> & _scan_cloud);
    Frame_Ellipsoid BulidingEllipsoidModel(void);

    //构建椭球
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GetCovarMatrix(std::vector<Eigen::Vector3d> bin_piont);
    std::pair<Eigen::Vector3d,Eigen::Matrix3d> GetEigenvalues(Eigen::MatrixXd bin_cov);
    int ClassifyEllipsoid(Ellipsoid voxeleloid);
    bool IsEllipsoidVaild(Ellipsoid voxeleloid);

    //可视化函数

    //自适应划分体素
    void AdaptiveSegmentation(std::map<int, VoxelData> origin_voxel_);
    std::vector<VoxelData> project_pt(Eigen::Matrix<double, 1, 3> cur_vector, VoxelData voxel_data,  Eigen::Vector3d cur_leaf_mean, Eigen::Vector3d cur_leaf_coordinate, Eigen::Vector3d cur_leaf_length);
    std::vector<VoxelData> subdivision(Eigen::Vector3d seg_lidar_point, std::vector<Eigen::Vector3d> cur_leaf_point, Eigen::Vector3d cur_leaf_coordinate, Eigen::Vector3d cur_leaf_length);
    std::vector<VoxelData> subdivision_1(bool seg_x, bool seg_y, bool seg_z, Eigen::Vector3d seg_lidar_point, std::vector<Eigen::Vector3d> cur_leaf_point, Eigen::Vector3d cur_leaf_coordinate, Eigen::Vector3d cur_leaf_length);
    std::vector<VoxelData> subdivision_2(bool seg_x, bool seg_y, bool seg_z, Eigen::Vector3d seg_lidar_point, std::vector<Eigen::Vector3d> cur_leaf_point, Eigen::Vector3d cur_leaf_coordinate,Eigen::Vector3d cur_leaf_length);

    //kd树描述符（sc改版）
    std::vector<float> MakeAndSaveDescriptorAndKey(std::map<int, VoxelData> origin_voxel_, int frame_id);
    Eigen::MatrixXd GetMatrixDescriptor(std::map<int, VoxelData> origin_voxel_);
    Eigen::MatrixXd MakeVerticalKeyFromDescriptor( Eigen::MatrixXd &_desc );

};