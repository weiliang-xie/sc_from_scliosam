#include "utility.h"

#include "lio_sam/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

// #include <pcl/registration/ndt.h>

#include "Scancontext.h"
#include "Normaldistribute.h"
#include "Mixdescriptor.h"


using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose


void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), fstream::out);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;

        const Pose3& pose = p->value();

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}


/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT    //在点云xyz基础上自定义结构，包括 xyz intensity roll pitch yaw time
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

//注册点云类型
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

// giseop
enum class SCInputType { 
    SINGLE_SCAN_FULL, 
    SINGLE_SCAN_FEAT, 
    MULTI_SCAN_FEAT 
}; 

class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subCloud;
    ros::Subscriber subGPS;
    ros::Subscriber subLoop;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lio_sam::cloud_info cloudInfo;
    sensor_msgs::PointCloud2 cloudinfo_new;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;    //TODO 这个是每次配准获取到的转换矩阵储存指针？
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses2D; // giseop 
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudRaw; // giseop    //去畸变后的点云 即未经任何提取的 原始点云
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS; // giseop  //降采样后的原始点云
    double laserCloudRawTime;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::VoxelGrid<PointType> downSizeFilterSC; // giseop   //体素下采样结构
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    // map<int, int> loopIndexContainer; // from new to old
    multimap<int, int> SCloopIndexContainer; // from new to old // giseop 

    vector<pair<int, int>> loopIndexQueue;  //回环序号队列容器
    vector<gtsam::Pose3> loopPoseQueue;
    // vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue; // Diagonal <- Gausssian <- Base
    vector<gtsam::SharedNoiseModel> loopNoiseQueue; // giseop for polymorhpisam (Diagonal <- Gausssian <- Base)

    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    // // loop detector 
    SCManager scManager;    //SC类定义
    NDManager ndManager;    //ND类定义
    MIXManager mixManager;   //混合类定义

    // data saver
    std::fstream pgSaveStream; // pg: pose-graph 
    std::fstream pgTimeSaveStream; // pg: pose-graph 
    std::vector<std::string> edges_str;
    std::vector<std::string> vertices_str;
    // std::fstream pgVertexSaveStream;
    // std::fstream pgEdgeSaveStream;

    std::string saveSCDDirectory;
    std::string saveNodePCDDirectory;
    std::string NDsaveSCDDirectory;
    std::string NDsaveNodePCDDirectory;
    std::string MIXsaveSCDDirectory;
    std::string MIXsaveNodePCDDirectory;

    //xwl 修改添加
    string data_set_sq = "00";
    int16_t laser_cloud_frame_number;   //实时更新的当前帧id
    Eigen::MatrixXd sc_pose_origin;
    Eigen::MatrixXd sc_pose_change;
    std::vector<Eigen::Matrix4d> pose_ground_truth;
    std::vector<Eigen::Matrix4d> pose_final_change;
    pcl::PointCloud<PointTypePose>::Ptr cloudkey; 
    std::vector<int> loopclosure_gt_index;  //回环真值id队列
    ofstream prFile;                                                    //pr文件流定义
    string sc_pr_data_file = savePCDDirectory + "SC/PRcurve/sc_kitti_" + data_set_sq + "_can-20_gt-small.csv";    //SC pr数据储存地址
    string nd_pr_data_file = savePCDDirectory + "ND/PRcurve/nd_kitti_" + data_set_sq + "_ca_ve-test_can-40.csv";    //ND pr数据储存地址
    string mix_pr_data_file = savePCDDirectory + "MIX/PRcurve/mix_kitti_" + data_set_sq + "_ca_num_ve-test_can-20.csv";    //MIX pr数据储存地址



public:
    mapOptimization()
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        // subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());   //TODO 根据话题修改订阅话题名

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        const float kSCFilterSize = 0.5; // giseop
        downSizeFilterSC.setLeafSize(kSCFilterSize, kSCFilterSize, kSCFilterSize); // giseop    //设置降采样的体素大小

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

        // giseop
        // create directory and remove old files;
        // savePCDDirectory = std::getenv("HOME") + savePCDDirectory; // rather use global path 
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());

        saveSCDDirectory = savePCDDirectory + "SC/SCDs/"; // SCD: scan context descriptor 
        unused = system((std::string("exec rm -r ") + saveSCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveSCDDirectory).c_str());

        NDsaveSCDDirectory = savePCDDirectory + "ND/SCDs/"; // SCD: scan context descriptor 
        unused = system((std::string("exec rm -r ") + NDsaveSCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + NDsaveSCDDirectory).c_str());

        MIXsaveSCDDirectory = savePCDDirectory + "MIX/SCDs/"; // SCD: scan context descriptor 
        unused = system((std::string("exec rm -r ") + MIXsaveSCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + MIXsaveSCDDirectory).c_str());

        saveNodePCDDirectory = savePCDDirectory + "SC/Scans/";
        unused = system((std::string("exec rm -r ") + saveNodePCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveNodePCDDirectory).c_str());

        NDsaveNodePCDDirectory = savePCDDirectory + "ND/Scans/";
        unused = system((std::string("exec rm -r ") + NDsaveNodePCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + NDsaveNodePCDDirectory).c_str());

        MIXsaveNodePCDDirectory = savePCDDirectory + "MIX/Scans/";
        unused = system((std::string("exec rm -r ") + MIXsaveNodePCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + MIXsaveNodePCDDirectory).c_str());

        pgSaveStream = std::fstream(savePCDDirectory + "singlesession_posegraph.g2o", std::fstream::out);
        pgTimeSaveStream = std::fstream(savePCDDirectory + "times.txt", std::fstream::out); pgTimeSaveStream.precision(dbl::max_digits10);
        // pgVertexSaveStream = std::fstream(savePCDDirectory + "singlesession_vertex.g2o", std::fstream::out);
        // pgEdgeSaveStream = std::fstream(savePCDDirectory + "singlesession_edge.g2o", std::fstream::out);

        laser_cloud_frame_number = 0;

        getposegroundtruth();   //获取序列的pose真值
        getloopclosuregt();     //获取回环真值

    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses2D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudRaw.reset(new pcl::PointCloud<PointType>()); // giseop
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>()); // giseop

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        cloudkey.reset(new pcl::PointCloud<PointTypePose>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
    }

    void writeVertex(const int _node_idx, const gtsam::Pose3& _initPose)
    {
        gtsam::Point3 t = _initPose.translation();
        gtsam::Rot3 R = _initPose.rotation();

        std::string curVertexInfo {
            "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
            + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
            + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
            + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

        // pgVertexSaveStream << curVertexInfo << std::endl;
        vertices_str.emplace_back(curVertexInfo);
    }
    
    void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose)  //将变量转换成字符串并以一定格式存入edges_str容器中
    {
        gtsam::Point3 t = _relPose.translation();
        gtsam::Rot3 R = _relPose.rotation();

        std::string curEdgeInfo {
            "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
            + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
            + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
            + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

        // pgEdgeSaveStream << curEdgeInfo << std::endl;
        edges_str.emplace_back(curEdgeInfo);    //与push_back()基本一致
    }

    //加载保存的描述矩阵数据函数 输入矩阵数据文件夹 返回矩阵数据
    template<typename T>
    T test_load_csv_descriptor(const std::string & path) 
    {
        std::ifstream in;
        in.open(path);
        std::string line;
        std::vector<double> values;
        uint rows = 0;
        while (std::getline(in, line))
        {
            std::stringstream ss(line);

	            string token;			// 接收缓冲区
	            while (getline(ss, token, ' '))	// 以split为分隔符
	            {
	            	double val = std::stod(token);
                    values.push_back(val);
	            }
            ++rows;
        }
        return Eigen::Map<const Eigen::Matrix<
            typename T::Scalar, 
            T::RowsAtCompileTime, 
            T::ColsAtCompileTime,
            RowMajor>>(values.data(), rows, values.size() / rows);
    }

    //加载保存的椭球数据函数 输入椭球数据文件夹 返回体素椭球模型vector 与SaveVoxelellipsoidData()对应
    std::vector<class Voxel_Ellipsoid>  test_load_csv_voxel_eloid(const std::string & path) 
    {
        std::ifstream in;
        in.open(path);
        std::string line;
        std::vector<double> values;
        std::vector<class Voxel_Ellipsoid> cloud_eloid;
        uint rows = 0;
        while (std::getline(in, line))
        {
            if(line == "ID,valid,num,a,b,c,a-mean,b-mean,c-mean,a-max,b-max,c-max,mode,a-axis-x,a-axis-y,a-axis-z,b-axis-x,b-axis-y,b-axis-z,c-axis-x,c-axis-y,c-axis-z,")
            {   
                cout << "first line" << endl;
                continue;
            }

            std::stringstream ss(line);
	        string token;			// 接收缓冲区
            int cnt = 0;
            class Voxel_Ellipsoid bin_eloid;
	        while (getline(ss, token, ','))
	        {
                // cout << token << endl;
                double val = std::stod(token);
                switch (cnt)
                {
                    case 0:
                        break;
                    case 1:
                        bin_eloid.valid = (bool)val;
                        break;
                    case 2:
                        bin_eloid.point_num = (int)val;
                        break;
                    case 3:
                        bin_eloid.axis_length[0] = val;
                        break;
                    case 4:
                        bin_eloid.axis_length[1] = val;
                        break;
                    case 5:
                        bin_eloid.axis_length[2] = val;
                        break;
                    case 6:
                        bin_eloid.center.x = (float)val;
                        break;
                    case 7:
                        bin_eloid.center.y = (float)val;
                        break;
                    case 8:
                        bin_eloid.center.z = (float)val;
                        break;
                    case 9:
                        bin_eloid.max_h_center.x = (float)val;
                        break;
                    case 10:
                        bin_eloid.max_h_center.y = (float)val;
                        break;
                    case 11:
                        bin_eloid.max_h_center.z = (float)val;
                        break;
                    case 12:
                        bin_eloid.mode = (int)val;
                        break;
                    case 13:
                        break;
                }
                if (bin_eloid.valid == 0 && cnt == 1)
                    break;
                cnt++;
	        }
            // cout << "cnt: " << cnt << endl;
            ++rows;
            cloud_eloid.push_back(bin_eloid);
        }
        return cloud_eloid;
    }

    void testcode()
    {
        int cur = 1567;
        int can = 122;
        // MatrixXd cur_des = test_load_csv_descriptor<MatrixXd>("/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMND/SCDs copy/001567.scd");
        // MatrixXd can_des = test_load_csv_descriptor<MatrixXd>("/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMND/SCDs copy/000122.scd");

        std::vector<class Voxel_Ellipsoid>  cur_eloid = test_load_csv_voxel_eloid("/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMVoxelEllipsoid copy/CloudData/001567.csv");
        std::vector<class Voxel_Ellipsoid>  can_eloid = test_load_csv_voxel_eloid("/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMVoxelEllipsoid copy/CloudData/000122.csv");
        
        cout << "cur_eloid num:" << cur_eloid.size() << endl;
        cout << "can_eloid num:" << can_eloid.size() << endl;


        // Eigen::Vector3d a =  ndManager.NDDistMergeVoxelellipsoid(cur_eloid,can_eloid,1);
        Eigen::Vector3d pre_center_vector = ndManager.MatchKeyVoxelEllipsoid(cur_eloid,can_eloid);
        // double b = ndManager.NDDistVoxeleloid(cur_eloid,can_eloid,1);
        cout << "avg center:" << endl;
        cout << pre_center_vector << endl;
        Eigen::Vector3d gt_center_vector = {pose_ground_truth[cur](0,3) - pose_ground_truth[can](0,3),
                                            pose_ground_truth[cur](1,3) - pose_ground_truth[can](1,3),
                                            pose_ground_truth[cur](2,3) - pose_ground_truth[can](2,3)};
        
        double cos_vector = (1 - ((pre_center_vector.dot(gt_center_vector)) / (pre_center_vector.norm() * gt_center_vector.norm())));
        cout << "gt center vector: " << endl;
        cout << gt_center_vector << endl;
        cout << "cos error: " << cos_vector << endl;
    }





    

    //获取pose真值
    void getposegroundtruth()
    {

        const char *fileName = "/home/jtcx/data_set/kitti/data_odometry/dataset/poses/00.txt";
    
        std::ifstream fileStream;
        std::string buf;
        float temp;
        int k;
        fileStream.open(fileName, std::ios::in);//ios::in 表示以只读的方式读取文件
    
        if (fileStream.fail())//文件打开失败:返回0
        {
            cout << "open pose file fail!" << endl;
        } else//文件存在
        {
            while (getline(fileStream, buf, '\n'))//读取一行
            {
                k = 0;
                // j = 0;
                // l = 0;
                int index = 0;
                Eigen::Matrix4d tmpV;

                for (uint i = 0; i < buf.size(); i++) {
                    if (buf[i] == ' ' || buf[i] == '\n' || i == buf.size() - 1) {
                        if(i == buf.size() - 1) i++;
                        temp = stod(buf.substr(k, i-k));

                        tmpV((index/4),(index%4)) = temp;
                        index++;
                        if(index == 12-1)
                        {
                            tmpV(3,0) = 0.0;
                            tmpV(3,1) = 0.0;
                            tmpV(3,2) = 0.0;
                            tmpV(3,3) = 1.0;
                        }

                        k = i + 1;
                    }
                }
                // cout << "matrix: \r\n" << tmpV << endl;
                pose_ground_truth.push_back(tmpV);
                tmpV.setZero();
            }
            fileStream.close();
        }

    }

    /*xwl 位姿转换 
    输入描述符的原始点云序号和旋转偏移量（弧度）
    输出位姿矩阵*/
    void changeSCpose(int16_t cloud_frame_index, float yawdata)
    {
        Eigen::Matrix4d last_offset_pose;
        //制作旋转偏移矩阵
        Eigen::Matrix4d rotate_offset;
        rotate_offset << cos(yawdata),-sin(yawdata),0.0,0.0,
                         sin(yawdata),cos(yawdata),0.0,0.0,
                         0.0,0.0,1.0,1.0;
        cout << "xwl rotate offset matrix: \r\n" << rotate_offset << endl;
        last_offset_pose = rotate_offset * pose_ground_truth[cloud_frame_index];
        cout << "xwl change final matrix: \r\n" << last_offset_pose << endl;
        pose_final_change.push_back(last_offset_pose);
    }

    //计算回环gt 通过位置距离判断，与非前后50帧内的距离小于10m则为回环
    void getloopclosuregt(void)
    {
        
        for(int i = 0; i < pose_ground_truth.size(); i++)
        {
            int cur_index = i;
            // cout << "cur index: " << cur_index << endl;
            if(cur_index < 50) continue;  //前50帧不进行回环判断
            Eigen::Matrix4d pose_matrix;
            double distance = 0;
            double cur_x, cur_y;

            // pose_matrix = *it;
            cur_x = pose_ground_truth[i](0,3);
            cur_y = pose_ground_truth[i](2,3);


            for(int i = 0; i < cur_index - 50; i++)
            {
                double his_x, his_y;
                his_x = pose_ground_truth[i](0,3);
                his_y = pose_ground_truth[i](2,3);

                // distance = sqrt((his_x-cur_x)*(his_x-cur_x) + (his_y-cur_y)*(his_y-cur_y) + (his_z-cur_z)*(his_z-cur_z));
                distance = sqrt((his_x-cur_x)*(his_x-cur_x) + (his_y-cur_y)*(his_y-cur_y));
                // cout << "cur and his distance: " << distance << endl;
                if (distance <= 5)
                {
                    loopclosure_gt_index.push_back(cur_index);
                    break; 
                }
            }
        }
        cout << "loop closure num: " << loopclosure_gt_index.size() << endl;
        //打印真值ID
        // for(auto &gt_data : loopclosure_gt_index)
        // {
        //     cout << gt_data << " ";
        // }
        // cout <<endl;
        cout << "finish get loop closure gt" << endl;
    }




    /*回环PR计算
      输入： 预测点云帧id和最小距离
      输出： PR值*/
    std::vector<std::pair<double,double>> makeprcurvedata(std::vector<std::pair<int,double>> & loopclosure_id_and_dist)
    {
        std::vector<std::pair<double,double>> pr_data_queue;
        int tp,fp,fn,pre_loop_num;
        double presession, recall;
        double value;
        double min_dist = 100000;
        double max_dist = 0.00001;
        tp = 0;
        fp = 0;
        fn = 0;
        pre_loop_num = 0;

        //寻找dist最大值 最小值
        for(auto pre_pair = loopclosure_id_and_dist.begin(); pre_pair != loopclosure_id_and_dist.end(); ++pre_pair)
        {
            min_dist = pre_pair->second < min_dist ? pre_pair->second : min_dist;
            max_dist = pre_pair->second > max_dist ? pre_pair->second : max_dist;
        }

        cout << "predict loop closure num is        " << loopclosure_id_and_dist.size() << endl;
        // cout << "predict origin loop closure num is " << scManager.context_origin_index.size() << endl;


        cout << "min distance is: " << min_dist << "    max distance is: " << max_dist <<endl;

        //将dist划分出50个阈值
        for(value = min_dist + (max_dist-min_dist)/50; value <= max_dist; value += (max_dist-min_dist)/50)
        {
            cout << "value is: " << value << endl;
            for(auto pre_pair = loopclosure_id_and_dist.begin(); pre_pair != loopclosure_id_and_dist.end(); ++pre_pair)
            {
                if(pre_pair->second <= value)
                {
                    pre_loop_num++;
                    // cout << "pre_pair_index: " << pre_pair->first << " is loop frame" << endl;
                    for(auto gt_it = loopclosure_gt_index.begin(); gt_it != loopclosure_gt_index.end(); ++gt_it)
                    {
                        if(*gt_it == pre_pair->first)
                        {
                            tp++;
                            break;
                        }
                            
                        if(gt_it == loopclosure_gt_index.end() - 1)
                            fp++;
                    }                    
                } 
            }

            fn = loopclosure_gt_index.size() - tp;

            cout << "tp: " << tp << "   fp: " << fp << endl;
            cout << "loop closure pre num:  " << pre_loop_num << endl;

            presession = (static_cast<double>(tp)/(tp + fp));
            recall = (static_cast<double>(tp)/(tp + fn));
            tp = 0;
            fp = 0;
            pre_loop_num = 0;

            std::pair<double,double> pr_data = {recall,presession};
            pr_data_queue.push_back(pr_data);

            cout << "presession: " << presession << "   recall: " << recall << endl;

        }
            cout << "all cloud frame num:   " << laser_cloud_frame_number << endl;
            cout << "loop closure gt num:   " << loopclosure_gt_index.size() << endl; 

        return pr_data_queue;

    }

    void saveprcurvedata(const std::string pr_data_file_name, std::vector<std::pair<double,double>> pr_data_queue)
    {
        if(pr_data_queue.empty())
            return;
        //写入csv文件
        prFile.open(pr_data_file_name, ios::out);
        // 写入标题行
        prFile << "recall" << ',' << "presession" << endl;

        for(auto prdata = pr_data_queue.begin(); prdata != pr_data_queue.end(); ++prdata)
        {
            prFile << prdata->first << "," << prdata->second << endl;
        }
        prFile.close();        
    }

    void makeandsaveprcurve()
    {
        std::vector<std::pair<double,double>> sc_pr_data_queue;
        std::vector<std::pair<double,double>> nd_pr_data_queue;
        std::vector<std::pair<double,double>> mix_pr_data_queue;

        if(scManager.loopclosure_id_and_dist.empty() != 1)
            sc_pr_data_queue = makeprcurvedata(scManager.loopclosure_id_and_dist);
        if(ndManager.loopclosure_id_and_dist.empty() != 1)
            nd_pr_data_queue = makeprcurvedata(ndManager.loopclosure_id_and_dist);
        if(mixManager.loopclosure_id_and_dist.empty() != 1)
            mix_pr_data_queue = makeprcurvedata(mixManager.loopclosure_id_and_dist);


        saveprcurvedata(sc_pr_data_file, sc_pr_data_queue);
        saveprcurvedata(nd_pr_data_file, nd_pr_data_queue);
        saveprcurvedata(mix_pr_data_file, mix_pr_data_queue);
    }







    //更新路径
    void updataposepath(Eigen::Matrix4d pose)
    {
        PointTypePose pose_truth;
        //平移

        pose_truth.x = pose(0,3);
        pose_truth.y = pose(2,3);
        pose_truth.z = -pose(1,3);  
        // cout << "pose truth x: " << pose_truth.x << 
        //         "pose truth y: " << pose_truth.y << 
        //         "pose truth z: " << pose_truth.z << endl;
        pose_truth.pitch = 0.0;
        pose_truth.roll = 0.0;
        pose_truth.yaw = 0.0;
        pose_truth.time = timeLaserInfoCur;

        //旋转
        // rotationMatrixToEulerAngles();


        updatePath(pose_truth);
    }


    // void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)    //ros订阅信息接收回调函数 订阅特征提取cpp发送的点云数据
    void laserCloudInfoHandler(const sensor_msgs::PointCloud2ConstPtr& msg)    //ros订阅信息接收回调函数 订阅特征提取cpp发送的点云数据
    {
        // cout << "xwl receive point cloud data" << endl;
        // extract time stamp
        timeLaserInfoStamp = msg->header.stamp;   //点云时间戳
        timeLaserInfoCur = msg->header.stamp.toSec(); //转换成秒为单位

        // extract info and feature cloud
        cloudinfo_new = *msg;
        //TODO 需要提取消息中的点云数据
        pcl::fromROSMsg(*msg, *laserCloudRaw); // giseop    //*将接收到的点云转换成laserCloudRaw
        laserCloudRawTime = cloudinfo_new.header.stamp.toSec(); // giseop save node time
        laser_cloud_frame_number++;
        // cout << "xwl receive time is " << laserCloudRawTime << "    frame number is " << laser_cloud_frame_number << endl;

        // static tf    发布odom坐标系
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, msg->header.stamp, mapFrame, odometryFrame));

        //更新路径
        updataposepath(pose_ground_truth[laser_cloud_frame_number]);

        std::lock_guard<std::mutex> lock(mtx);  //开启互斥锁 lock_guard互斥锁的一种写法，在构造函数内加锁，析构函数中自动解锁

        static double timeLastProcessing = -1;
        // if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)    //判断接收点云的间隔有没有大于映射间隔
        {
            // std::cout << "xwl enter deal function" << std::endl;
            timeLastProcessing = timeLaserInfoCur;
            
            // downsampleCurrentScan();    //降采样处理

            // SCsaveKeyFramesAndFactor();   //制作SC描述符并更新位姿信息

            NDsaveKeyFramesAndFactor();     //制作ND描述符

            // MIXsaveKeyFramesAndFactor();    //制作MIX描述符

            // performSCLoopClosure();      //求解SC回环状态

            NDperformSCLoopClosure();      //求解ND回环状态

            // MIXperformSCLoopClosure();      //求解MIX回环状态

            publishFrames();             //发布路径

        }

        laserCloudRaw->points.clear();
        // if(laserCloudRaw->points.empty())
        //     cout << "laser cloud is empty!" << endl;    //有效        

        //接收完数据后进行pr计算    仅限于kitti数据集的00 02 05 08
        int16_t sq_num = data_set_sq == "00" ? 4541 : (data_set_sq == "02" ? 4661 : (data_set_sq == "05" ? 2761 : 4071));
        if(laser_cloud_frame_number == sq_num)
            makeandsaveprcurve();
    }


    //使用转换矩阵转换点云
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }


    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }


    void loopClosureThread()    //回环检测函数进程
    {
        // if (loopClosureEnableFlag == false)     //判断是否开启回环检测使能
        //     return;

        ros::Rate rate(loopClosureFrequency);   //设定回环线程执行频率
        while (ros::ok())
        {
            rate.sleep();
            // performRSLoopClosure();     //TODO 是否为lio-sam原始的icp回环检测 如果是 是否屏蔽了? 如何屏蔽？ 
            // performSCLoopClosure(); // giseop   
            visualizeLoopClosure();
        }
    }

    void visualizeLoopClosure()     //回环闭合可视化函数
    {
        visualization_msgs::MarkerArray markerArray;    //maker队列
        // loop nodes   //点可视化配置
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges 线可视化配置
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1; markerEdge.scale.y = 0.1; markerEdge.scale.z = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = SCloopIndexContainer.begin(); it != SCloopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = pose_ground_truth[key_cur](0,3);
            p.y = pose_ground_truth[key_cur](2,3);
            p.z = 0;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = pose_ground_truth[key_pre](0,3);
            p.y = pose_ground_truth[key_pre](2,3);
            p.z = 0;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }



    void performSCLoopClosure()     //执行回环闭合 检测是否有回环 并对回环进行配准校正位姿并保存
    {
        // if (cloudKeyPoses3D->points.empty() == true)    //points是储存所有点数据的数组 判断点云是否为空
        //     return;

        // find keys
        // cout << "   xwl enter perform sc loop closure" << endl;

        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
        int loopKeyCur = laser_cloud_frame_number;   //获取当前获取的实时帧id  变量失效
        int loopKeyPre = detectResult.first;                //获取存在回环的历史帧的id 若不存在则返回-1
        float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
        if( loopKeyPre == -1 /* No loop found */)
            return;

        // std::cout << "[SC] SC loop found! between " << laser_cloud_frame_number << " and " << loopKeyPre << "." << std::endl; // giseop

        //xwl 描述符匹配旋转位姿转换对比


        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        // {
            loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, loopKeyPre); // giseop 
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

            int base_key = 0;
            loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, base_key); // giseop 
            loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key); // giseop    //获取转换后的点云

            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)    //? 这里判断点云内的点数量是否足够？
                return;
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)   //判断是否有节点订阅历史关键帧话题 getNumSubscribers返回是否有节点订阅
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);   //发布历史关键点云话题 输入 话题名 点云指针 当前ros时间 点云坐标系
        // }


        

        // ICP Settings
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);  //输入当前实时帧和回环找到的匹配帧
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);  //?ICP配准没有用上初始位姿？
        // giseop 
        // TODO icp align with initial 

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore) {
            std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this SC loop." << std::endl;
            return;
        } else {
            std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this SC loop." << std::endl;
        }

        // publish corrected cloud
        if (pubIcpKeyFrames.getNumSubscribers() != 0)   //判断是否有节点订阅ICP配准转换后的点云
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());  //将当前帧用ICP配准后的转换矩阵转换
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);    //发布转换后的实时帧
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;   //定义一个三维仿射变换矩阵
        correctionLidarFrame = icp.getFinalTransformation();    //获取配准后的转换矩阵 getFinalTransformation()返回的是4x4矩阵

        // // transform from world origin to wrong pose
        // Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // // transform from world origin to corrected pose
        // Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        // pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        // gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

        // gtsam::Vector Vector6(6);
        // float noiseScore = icp.getFitnessScore();
        // Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        // noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // giseop 
        pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);    //从变换矩阵中获取转换值
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

        // giseop, robust kernel for a SC loop
        float robustNoiseScore = 0.5; // constant is ok...
        gtsam::Vector robustNoiseVector6(6); 
        robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        noiseModel::Base::shared_ptr robustConstraintNoise; 
        robustConstraintNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
        ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        // Add pose constraint
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));  //
        loopNoiseQueue.push_back(robustConstraintNoise);
        mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        SCloopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap  //将当前帧与历史帧中匹配好的对应帧同时储存
    } // performSCLoopClosure

    void NDperformSCLoopClosure()     //执行ND回环闭合 检测是否有回环 并对回环进行配准校正位姿并保存
    {
        // if (cloudKeyPoses3D->points.empty() == true)    //points是储存所有点数据的数组 判断点云是否为空
        //     return;

        // find keys
        // cout << "   xwl enter perform sc loop closure" << endl;

        auto detectResult = ndManager.NDdetectLoopClosureID(); // first: nn index, second: yaw diff 
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;   //获取当前获取的实时帧id  变量失效
        int loopKeyPre = detectResult.first;                //获取存在回环的历史帧的id 若不存在则返回-1
        float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
        if( loopKeyPre == -1 /* No loop found */)
            return;

        // std::cout << "[ND] SC loop found! between " << laser_cloud_frame_number << " and " << loopKeyPre << "." << std::endl; // giseop

        //xwl 描述符匹配旋转位姿转换对比


        // // extract cloud
        // pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        // pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        // // {
            // loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, loopKeyPre); // giseop 
            // loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

            // int base_key = 0;
            // loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, base_key); // giseop 
            // loopFindNearKeyframesWithRespectTo(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum, base_key); // giseop    //获取转换后的点云

            // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)    //? 这里判断点云内的点数量是否足够？
            //     return;
            // if (pubHistoryKeyFrames.getNumSubscribers() != 0)   //判断是否有节点订阅历史关键帧话题 getNumSubscribers返回是否有节点订阅
            //     publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);   //发布历史关键点云话题 输入 话题名 点云指针 当前ros时间 点云坐标系
        // }


        

        // ICP Settings
        // static pcl::IterativeClosestPoint<PointType, PointType> icp;
        // icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
        // icp.setMaximumIterations(100);
        // icp.setTransformationEpsilon(1e-6);
        // icp.setEuclideanFitnessEpsilon(1e-6);
        // icp.setRANSACIterations(0);

        // Align clouds
        // icp.setInputSource(cureKeyframeCloud);  //输入当前实时帧和回环找到的匹配帧
        // icp.setInputTarget(prevKeyframeCloud);
        // pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // icp.align(*unused_result);  //?ICP配准没有用上初始位姿？
        // // giseop 
        // // TODO icp align with initial 

        // if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore) {
        //     std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this SC loop." << std::endl;
        //     return;
        // } else {
        //     std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this SC loop." << std::endl;
        // }

        // publish corrected cloud
        // if (pubIcpKeyFrames.getNumSubscribers() != 0)   //判断是否有节点订阅ICP配准转换后的点云
        // {
        //     pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        //     pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());  //将当前帧用ICP配准后的转换矩阵转换
        //     publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);    //发布转换后的实时帧
        // }

        // Get pose transformation
        // float x, y, z, roll, pitch, yaw;
        // Eigen::Affine3f correctionLidarFrame;   //定义一个三维仿射变换矩阵
        // correctionLidarFrame = icp.getFinalTransformation();    //获取配准后的转换矩阵 getFinalTransformation()返回的是4x4矩阵

        // // transform from world origin to wrong pose
        // Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // // transform from world origin to corrected pose
        // Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        // pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        // gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);

        // gtsam::Vector Vector6(6);
        // float noiseScore = icp.getFitnessScore();
        // Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        // noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // giseop 
        // pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);    //从变换矩阵中获取转换值
        // gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

        // giseop, robust kernel for a SC loop
        // float robustNoiseScore = 0.5; // constant is ok...
        // gtsam::Vector robustNoiseVector6(6); 
        // robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        // noiseModel::Base::shared_ptr robustConstraintNoise; 
        // robustConstraintNoise = gtsam::noiseModel::Robust::Create(
        //     gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
        //     gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
        // ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        // Add pose constraint
        // mtx.lock();
        // loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        // loopPoseQueue.push_back(poseFrom.between(poseTo));  //
        // loopNoiseQueue.push_back(robustConstraintNoise);
        // mtx.unlock();

        // add loop constriant
        // loopIndexContainer[loopKeyCur] = loopKeyPre;
        // loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap  //将当前帧与历史帧中匹配好的对应帧同时储存
    } // performSCLoopClosure

    void MIXperformSCLoopClosure()     //执行MIX回环闭合 检测是否有回环 并对回环进行配准校正位姿并保存
    {
        // if (cloudKeyPoses3D->points.empty() == true)    //points是储存所有点数据的数组 判断点云是否为空
        //     return;

        // find keys
        // cout << "   xwl enter perform sc loop closure" << endl;

        auto detectResult = mixManager.MIXdetectLoopClosureID(); // first: nn index, second: yaw diff 
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;   //获取当前获取的实时帧id  变量失效
        int loopKeyPre = detectResult.first;                //获取存在回环的历史帧的id 若不存在则返回-1
        float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
        if( loopKeyPre == -1 /* No loop found */)
            return;

        // std::cout << "[MIX] SC loop found! between " << laser_cloud_frame_number << " and " << loopKeyPre << "." << std::endl; // giseop
    }

    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void loopFindNearKeyframesWithRespectTo(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum, const int _wrt_key)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[_wrt_key]); //使用转换矩阵转换点云 返回转换后的点云指针
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[_wrt_key]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes    //降采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }


    void downsampleCurrentScan()    //当前扫描降采样处理
    {
        //xwl 取消点云的降采样
        // giseop
        // laserCloudRawDS->clear();       //清空点云数据函数
        // downSizeFilterSC.setInputCloud(laserCloudRaw);  //三种点云类型全部都进行降采样处理
        // downSizeFilterSC.filter(*laserCloudRawDS);        

        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();        

    }


    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;   //TODO 如何获取相邻帧的变化位姿 
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); //获取欧拉角和平移量

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold) //判断各个位姿量是否满足大于判断阈值，若小于不保存关键帧
            return false;

        return true;
    }

    void SCsaveKeyFramesAndFactor()   //回环相关函数 TODO 求解描述符并保存矩阵和点云
    {
        if (saveFrame() == false)   //判断是否满足保存关键帧的要求
            return;

        const SCInputType sc_input_type = SCInputType::SINGLE_SCAN_FULL; // change this 

        //这里对输入类型进行判断后分类处理，但目前已经将分类条件写死了
        if( sc_input_type == SCInputType::SINGLE_SCAN_FULL ) {
            // std::cout << "xwl make and save scan context and keys" << std::endl;
            pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudRaw,  *thisRawCloudKeyFrame);  //复制点云
            scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);    //使用点云进行描述符制作
            scManager.context_origin_index.push_back(laser_cloud_frame_number);          //保存原始点云帧序号
        }  

        // save sc data
        const auto& curr_scd = scManager.getConstRefRecentSCD();    //获取当前的SC描述矩阵
        std::string curr_scd_node_idx = padZeros(scManager.polarcontexts_.size() - 1);  //记下当前矩阵序号（string） 用于保存文件的命名

        saveSCD(saveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);   //命名并保存矩阵


        // save keyframe cloud as file giseop
        bool saveRawCloud { true }; //这里是选择保存哪种类型的点云
        pcl::PointCloud<PointType>::Ptr thisKeyFrameCloud(new pcl::PointCloud<PointType>());
        if(saveRawCloud) { 
            *thisKeyFrameCloud += *laserCloudRaw;
        } else {
            // *thisKeyFrameCloud += *thisCornerKeyFrame;
            // *thisKeyFrameCloud += *thisSurfKeyFrame;
        }
        pcl::io::savePCDFileBinary(saveNodePCDDirectory + curr_scd_node_idx + ".pcd", *thisKeyFrameCloud);
        pgTimeSaveStream << laserCloudRawTime << std::endl;
    }

    void NDsaveKeyFramesAndFactor()     //ND描述符制作
    {
        if (saveFrame() == false)   //判断是否满足保存关键帧的要求
            return;

        // std::cout << "[ND] make and save ND scan context and keys" << std::endl;

        pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudRaw,  *thisRawCloudKeyFrame);  //复制点云
        ndManager.NDmakeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);    //使用点云进行描述符制作
        ndManager.context_origin_index.push_back(laser_cloud_frame_number);          //保存原始点云帧序号

        // save sc data
        const auto& curr_scd = ndManager.NDgetConstRefRecentSCD();    //获取当前的SC描述矩阵
        std::string curr_scd_node_idx = padZeros(ndManager.polarcontexts_.size() - 1);  //记下当前矩阵序号（string） 用于保存文件的命名

        saveSCD(NDsaveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);   //命名并保存矩阵


        // save keyframe cloud as file giseop
        bool saveRawCloud { true }; //这里是选择保存哪种类型的点云
        pcl::PointCloud<PointType>::Ptr thisKeyFrameCloud(new pcl::PointCloud<PointType>());
        if(saveRawCloud) { 
            *thisKeyFrameCloud += *laserCloudRaw;
        } else {
            // *thisKeyFrameCloud += *thisCornerKeyFrame;
            // *thisKeyFrameCloud += *thisSurfKeyFrame;
        }
        pcl::io::savePCDFileBinary(NDsaveNodePCDDirectory + curr_scd_node_idx + ".pcd", *thisKeyFrameCloud);
        pgTimeSaveStream << laserCloudRawTime << std::endl;        
    }

    void MIXsaveKeyFramesAndFactor()     //MIX描述符制作
    {
        if (saveFrame() == false)   //判断是否满足保存关键帧的要求
            return;

        // std::cout << "[MIX] make and save MIX scan context and keys" << std::endl;

        pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudRaw,  *thisRawCloudKeyFrame);  //复制点云
        mixManager.MIXmakeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);    //使用点云进行描述符制作
        mixManager.context_origin_index.push_back(laser_cloud_frame_number);          //保存原始点云帧序号

        // save sc data
        const auto& curr_scd = mixManager.MIXgetConstRefRecentSCD();    //获取当前的SC描述矩阵
        std::string curr_scd_node_idx = padZeros(mixManager.polarcontexts_.size() - 1);  //记下当前矩阵序号（string） 用于保存文件的命名

        saveSCD(MIXsaveSCDDirectory + curr_scd_node_idx + ".scd", curr_scd);   //命名并保存矩阵


        // save keyframe cloud as file giseop
        bool saveRawCloud { true }; //这里是选择保存哪种类型的点云
        pcl::PointCloud<PointType>::Ptr thisKeyFrameCloud(new pcl::PointCloud<PointType>());
        if(saveRawCloud) { 
            *thisKeyFrameCloud += *laserCloudRaw;
        } else {
            // *thisKeyFrameCloud += *thisCornerKeyFrame;
            // *thisKeyFrameCloud += *thisSurfKeyFrame;
        }
        pcl::io::savePCDFileBinary(MIXsaveNodePCDDirectory + curr_scd_node_idx + ".pcd", *thisKeyFrameCloud);
        pgTimeSaveStream << laserCloudRawTime << std::endl;        
    }



    void updatePath(const PointTypePose& pose_in)   //发布路径更新路径
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);   //将欧拉角转换成四元数
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishFrames()
    {
        // if (cloudKeyPoses3D->points.empty())
        //     return;
        // publish key poses
        // publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        // publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        // if (pubRecentKeyFrame.getNumSubscribers() != 0)
        // {
        //     pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        //     PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
        //     *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
        //     *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
        //     publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        // }
        // publish registered high-res raw cloud
        // if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        // {
        //     pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        //     pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
        //     PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
        //     *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
        //     publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        // }
        // publish path
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    mapOptimization MO;

    // Eigen::Vector3d m1,m2,m3;
    // m1 << 1.2, 2.5, 5.6;
    // m2 << -3.6, 9.2, 0.5;
    // m3 << 4.3, 1.3, 9.4;
    // std::vector<Eigen::Vector3d> piont;
    // piont.push_back(m1);
    // piont.push_back(m2);
    // piont.push_back(m3);

    // Eigen::MatrixXd cov;
    // cov = MO.ndManager.NDGetCovarMatrix(piont);
    // MO.ndManager.NDGetSingularvalue(cov);

    // MO.testcode();

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");
    
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);       //开启回环检测线程
    // std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopthread.join();  //阻塞等待回环检测线程结束释放内存
    // visualizeMapThread.join();

    return 0;
} 
