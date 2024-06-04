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
#include <ceres/ceres.h>

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

#define CUSTOM_HASH_ENABLE 1    //测试用 开启自定义hash

//哈希表键值类  自定义哈希使用
struct HashKey{
    int mode;
    int cloud_exit;
    int mean_qf;    //质心的二次型
    int mean_z;
};

struct HashKeyEqual{
    bool operator()(const HashKey& lhs, const HashKey& rhs) const
    {
        // return lhs.mode == rhs.mode && lhs.cloud_exit == rhs.cloud_exit && lhs.mean_qf == rhs.mean_qf;
        // return lhs.mode == rhs.mode && lhs.cloud_exit == rhs.cloud_exit && lhs.mean_z == rhs.mean_z;
        return lhs.mode == rhs.mode && lhs.cloud_exit == rhs.cloud_exit;
        // return lhs.mean_qf == rhs.mean_qf;
    }
};
struct GetHashKey{
    std::size_t operator()(const HashKey& k) const
    {
        // return hash<int>()(k.cloud_exit) ^ hash<int>()(k.mean_qf) ^ hash<int>()(k.mode);
        // return hash<int>()(k.cloud_exit) ^ hash<int>()(k.mode) ^ hash<int>()(k.mean_z);
        return hash<int>()(k.mode) ^ hash<int>()(k.cloud_exit);
        // return hash<int>()(k.mean_qf);
    }
};

struct CostFunctor{
    const Eigen::Matrix<double, 2, 1> source_point, cand_point;
    //结构初始化，依次传入单个源点和单个候选点
    CostFunctor(Eigen::Matrix<double, 2, 1> source, Eigen::Matrix<double, 2, 1> cand) : source_point(source), cand_point(cand) {}

    template <typename T>
    bool operator()(const T* const tf_para, T *residual) const {
        const T x = tf_para[0];
        const T y = tf_para[1];
        const T theta = tf_para[2];
        Eigen::Matrix<T, 2, 2> R;
        R << cos(theta), -sin(theta), sin(theta), cos(theta);
        Eigen::Matrix<T, 2, 1> t(x, y);
        Eigen::Matrix<T, 2, 1> dis_pose = R * source_point + t - cand_point;
        // residual[0] = T(abs(dis_pose[0])) + T(abs(dis_pose[1]));
        residual[0] = T(dis_pose.norm());
        return true;
    }

};


class EllipsoidLocalization : public BaseGlobalLocalization //全局算法基类
{
public:
//evaluate
    Evaluate evaluate_data;

//base data
    std::vector<Frame_Ellipsoid> database_frame_eloid;                      //数据库存储的椭球模型
    std::vector<Frame_Ellipsoid> inquiry_frame_eloid;                           //存储的查询帧椭球模型
    std::vector<int> database_gt_id;                                        //database的对应真值id
    std::vector<int> inquiry_gt_id;                                        //inquiry的对应真值id

//hash
    std::vector<std::vector<int> > database_frame_eloid_key;             //数据库中各帧椭球的对应键值 与frame_eloid的nonground_voxel_eloid对应
    std::vector<std::vector<int> > inquiry_frame_eloid_key;                  //查询帧各帧椭球的对应键值 与frame_eloid的nonground_voxel_eloid对应

    std::unordered_map<int, vector<int> > eloid_eigen_map;              //椭球特征值的hash

    //base function
    void MakeDatabaseEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id);
    void MakeInquiryEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id);
    std::pair<int, float> Localization(int frame_id);

//检索部分
    //原hash
    void MakeDatabaseHashForm(int frame_id);        //数据集的hash表制作
    void MakeCurrentHashForm(int frame_id);         //查询集的hash表制作
    int GetEloidEigenKey(Ellipsoid eloid);
    vector<int> GetHashFrameID(int key);
    std::vector<int> GetCandidatesFrameIDwithHash(int frame_id);

    //自定义hash
    std::unordered_map<HashKey, vector<int>, GetHashKey, HashKeyEqual> custom_frame_hash;
    std::vector<std::vector<HashKey> > database_custom_frame_eloid_key; 
    std::vector<std::vector<HashKey> > cur_custom_frame_eloid_key;
    HashKey GetEloidEigenKeyCustom(Ellipsoid eloid);
    vector<int> GetHashFrameIDCustom(HashKey key);

    //kd树（sc改版）
    KeyMat database_vertical_invkeys_mat_;                //数据帧垂直方向上的键值集合
    KeyMat cur_vertical_invkeys_mat_;                     //查询帧垂直方向上的键值集合
    KeyMat vertical_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> verticalkey_tree_;

    std::vector<int> GetCandidatesFrameIDwithMatrix(int frame_id);    //返回在data数据中的索引，并非真实帧id，真实id需要通过database_gt_id来转换

    //hash + kd树 测试
    KeyMat database_vertical_invkeys_mat_after_hashfliter_;         //过滤后的数据库
    std::vector<int> after_database_gt_id;                            //after database的对应真值id


//匹配部分
    //协方差匹配
    const double MAX_COV_SIMILARITY_EUCLID  = 10;
    const double MAX_COV_SIMILARITY_COS  = 0.9;

    std::pair<int, double> LocalizationWithCov(std::vector<int> can_index);
    std::pair<double, Ellipsoid> FindNearestModel(Ellipsoid inquiry_model, std::vector<Ellipsoid> can_model);
    double GetCovSimilarityWithEuclidean(Eigen::Matrix3d inquiry_cov, Eigen::Matrix3d can_cov);
    double GetCovSimilarityWithCos(Matrix3d _sc1, Matrix3d _sc2);

//转移矩阵部分
    const int FEATRUE_POINT_NUMS = 30;                          //提取的特征点数量

    std::vector<Eigen::Matrix4d> transform_matrix;

    std::vector<Eigen::Vector3d> MakeFeaturePoint(Frame_Ellipsoid frame_eloid);
    Eigen::Matrix4d MakeFeaturePointandGetTransformMatirx(Frame_Ellipsoid frame_eloid_1, Frame_Ellipsoid frame_eloid_2);
    Eigen::Isometry2d EvaculateTFWithIso(Eigen::Matrix4d can_gt, Eigen::Matrix4d src_gt, Eigen::Matrix4d est);

};  