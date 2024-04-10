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

#include <unordered_map>        //hash

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "tictoc.h"

#include "BaseGlobalLocalization.h"

using namespace std;
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


class EllipsoidLocalization : public BaseGlobalLocalization //全局算法基类
{
public:
    std::vector<Frame_Ellipsoid> database_frame_eloid;                   //数据库存储的椭球模型
    std::vector<Frame_Ellipsoid> cur_frame_eloid;                        //存储的查询帧椭球模型

    std::vector<std::vector<int> > database_frame_eloid_key;             //数据库中各帧椭球的对应键值 与frame_eloid的nonground_voxel_eloid对应
    std::vector<std::vector<int> > cur_frame_eloid_key;                  //查询帧各帧椭球的对应键值 与frame_eloid的nonground_voxel_eloid对应



    std::unordered_map<int, vector<int> > eloid_eigen_map;              //椭球特征值的hash

    //eloid
    void MakeDatabaseEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id);
    void MakeInquiryEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id);
    int GetEloidEigenKey(Ellipsoid eloid);
    vector<int> GetHashFrameID(int key);

    std::vector<int> DetectLoopClosureID(int frame_id);

};