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
        center = {0};
        axis.resize(3,3);
        cov.resize(3,3);
        axis << 0,0,0,
                0,0,0,
                0,0,0;
        cov = axis;
        axis_length.resize(3);
        point_num = 0;
    }
    SCPointType center;                     //椭球中心点
    Eigen::MatrixXd axis;                   //椭球轴方向
    std::vector<double> axis_length;        //椭球轴长(目前 = 协方差特征值)
    int point_num;
    Eigen::MatrixXd cov;
};

class Voxel_Ellipsoid: public Ellipsoid{
public:
    Voxel_Ellipsoid(){
        valid = 0;
        mode = -1;
    }
    bool valid;         //是否为满足评判相似度的椭球模型
    int mode;           //椭球类型 1：线性 2：平面型 3：立体型
};

class Refer_Ellipsoid: public Ellipsoid{
public:
    std::vector<int> id;
};


class NDManager
{
public:
    NDManager() = default;
    void NDmakeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> & _scan_cloud);
    std::pair<int, float> NDdetectLoopClosureID( void ); // int: nearest node index, float: relative yaw  
    Eigen::MatrixXd NDmakeScancontext(pcl::PointCloud<SCPointType> & _scan_cloud);
    Eigen::MatrixXd NDmakeRingkeyFromScancontext( Eigen::MatrixXd &_desc );
    Eigen::MatrixXd NDmakeSectorkeyFromScancontext( Eigen::MatrixXd &_desc );

    int NDfastAlignUsingVkey ( MatrixXd & _vkey1, MatrixXd & _vkey2 ); 
    double NDdistDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 ); // "d" (eq 5) in the original paper (IROS 18)
    std::pair<double, int> NDdistanceBtnScanContext ( MatrixXd &_sc1, MatrixXd &_sc2 ); // "D" (eq 6) in the original paper (IROS 18)

    const Eigen::MatrixXd& NDgetConstRefRecentSCD(void);

    //体素椭球
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> NDGetCovarMatrix(std::vector<Eigen::Vector3d> bin_piont);
    Eigen::MatrixXd NDGetSingularvalue(Eigen::MatrixXd bin_cov);
    std::pair<std::vector<double>,Eigen::MatrixXd> NDGetEigenvalues(Eigen::MatrixXd bin_cov);
    std::pair<double, int> NDdistancevoxeleloid( MatrixXd &_sc1, MatrixXd &_sc2, std::vector<class Voxel_Ellipsoid> &v_eloid_cur,std::vector<class Voxel_Ellipsoid> &v_eloid_can);
    bool NDFilterVoxelellipsoid(class Voxel_Ellipsoid &voxeleloid);
    double NDDistVoxeleloid(std::vector<class Voxel_Ellipsoid> &v_eloid_cur, std::vector<class Voxel_Ellipsoid> &v_eloid_can, int num_shift);
    class Voxel_Ellipsoid NDMergeColVoxeleloid(std::vector<class Voxel_Ellipsoid> &v_eloid, int col);
    class Voxel_Ellipsoid NDMergeVoxeleloid(std::vector<class Voxel_Ellipsoid> &v_eloid, std::vector<int> v_id);    
    Eigen::Vector3d NDDistMergeVoxelellipsoid(std::vector<class Voxel_Ellipsoid> &v_eloid_cur,std::vector<class Voxel_Ellipsoid> &v_eloid_can, int num_shift);
    void NDSaveVoxelellipsoidData(std::vector<class Voxel_Ellipsoid> v_eloid_data, int id);


    template<typename _Tp>
    void print_matrix(const _Tp* data, const int rows, const int cols)
    {
    	for (int y = 0; y < rows; ++y) {
    		for (int x = 0; x < cols; ++x) {
    			fprintf(stderr, "  %f  ", static_cast<float>(data[y * cols + x]));
    		}
    		fprintf(stderr, "\n");
    	}
    	fprintf(stderr, "\n");
    }

public:
    const double LIDAR_HEIGHT = 2.0;                                                //雷达高度
    const int    ND_PC_NUM_RING = 20;                                               //圆环数量 沿径向切割
    const int    ND_PC_NUM_SECTOR = 60;                                             //扇形数量 沿方位角切割
    const double ND_PC_MAX_RADIUS = 80.0;                                           //最大检测距离
    const double ND_PC_UNIT_SECTORANGLE = 360.0 / double(ND_PC_NUM_SECTOR);         //单元方位角角度 deg
    const double ND_PC_UNIT_RINGGAP = ND_PC_MAX_RADIUS / double(ND_PC_NUM_RING);    //径向单元长度

    // tree
    const int    ND_NUM_EXCLUDE_RECENT = 30; // simply just keyframe gap (related with loopClosureFrequency in yaml), but node position distance-based exclusion is ok.    排除时间上相近的关键帧 30
    const int    ND_NUM_CANDIDATES_FROM_TREE = 20; // 10 is enough. (refer the IROS 18 paper)   //KD树的候选数量

    // loop thres
    const double ND_SEARCH_RATIO = 0.1; // for fast comparison, no Brute-force, but search 10 % is okay. // not was in the original conf paper, but improved ver.
    // const double ND_SC_DIST_THRES = 0.13; // empirically 0.1-0.2 is fine (rare false-alarms) for 20x60 polar context (but for 0.15 <, DCS or ICP fit score check (e.g., in LeGO-LOAM) should be required for robustness)
    const double ND_SC_DIST_THRES = 0.3; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15
    // const double ND_SC_DIST_THRES = 0.7; // 0.4-0.6 is good choice for using with robust kernel (e.g., Cauchy, DCS) + icp fitness threshold / if not, recommend 0.1-0.15

    // config 
    const int    TREE_MAKING_PERIOD_ = 100; // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / in the LeGO-LOAM integration, it is synchronized with the loop detection callback (which is 1Hz) so it means the tree is updated evrey 10 sec. But you can use the smaller value because it is enough fast ~ 5-50ms wrt N.
    int          tree_making_period_conter = 0;
    double       ND_VOXEL_ELIOD_DIST_THRES = 1;
    double       ND_VOXEL_ELIOD_COS_THRES = 1;


    //data
    std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;
    std::vector<int16_t> context_origin_index;  //制作描述符使用到的原始点云帧序号
    std::vector<std::pair<int,double>> loopclosure_id_and_dist;  //SC制作保存的所有回环帧id和相似度距离

    KeyMat polarcontext_invkeys_mat_;   //float的容器的容器
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;

    std::vector<std::vector<class Voxel_Ellipsoid> > cloud_voxel_eloid;   //各帧点云的体素椭球
};