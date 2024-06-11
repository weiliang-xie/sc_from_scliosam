#include "Ellipsoiddescriptor.h"

#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <functional>

#include <numeric>


using namespace std;

std::vector<Eigen::MatrixXd> GetSingularvalue(Eigen::MatrixXd bin_cov);
bool feature_point_cmp(Eigen::Vector3d point1, Eigen::Vector3d point2);
Eigen::Matrix4d GetTransformMatrix(vector<Eigen::Vector3d> feature_point_1, vector<Eigen::Vector3d> feature_point_2);
Eigen::Matrix4d GetTransformMatrixwithCERE(std::vector<Eigen::Matrix3Xd> source_feature_point, std::vector<Eigen::Matrix3Xd> cand_feature_point, const double yaw = 0);


//特征数据库描述符制作函数  输入点云数据 点云的帧id（ 从0开始算 ）
void EllipsoidLocalization::MakeDatabaseEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id)
{
    DivideVoxel(_scan_cloud);
    database_frame_eloid.push_back(BulidingEllipsoidModel());
    database_gt_id.push_back(frame_id);     //储存真值
    // cout << "[EL]  Make Dataset Descriptor  Dataset Frame Make Eloid  ground eloid num is: " << database_frame_eloid.back().ground_voxel_eloid.size() << endl;
    // cout << "[EL]  Make Dataset Descriptor  Dataset Frame Make Eloid  non ground eloid num is: " << database_frame_eloid.back().nonground_voxel_eloid.size() << endl;
    // cout << endl;

    //sc改版 建立描述符和键值
    std::vector<float> vertical_invkey_vec = MakeAndSaveDescriptorAndKey(frame_voxel.back().origin_voxel_data, frame_id);
    database_vertical_invkeys_mat_.push_back(vertical_invkey_vec);

    MakeDatabaseHashForm(frame_id);

}

//当前帧描述符制作函数  输入点云数据 点云的帧id（ 从0开始算 ）
void EllipsoidLocalization::MakeInquiryEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id)
{
    DivideVoxel(_scan_cloud);
    inquiry_frame_eloid.push_back(BulidingEllipsoidModel());
    inquiry_gt_id.push_back(frame_id);
    // cout << "[EL]  Cur Frame Make Eloid  ground eloid num is: " << inquiry_frame_eloid.back().ground_voxel_eloid.size() << endl;
    // cout << "[EL]  Cur Frame Make Eloid  non ground eloid num is: " << inquiry_frame_eloid.back().nonground_voxel_eloid.size() << endl;
    // cout << endl;

    //sc改版 建立描述符和键值
    std::vector<float> vertical_invkey_vec = MakeAndSaveDescriptorAndKey(frame_voxel.back().origin_voxel_data, frame_id);
    cur_vertical_invkeys_mat_.push_back(vertical_invkey_vec);

    MakeCurrentHashForm(frame_id);         //制作current哈希表
}


//定位函数
std::pair<int, float> EllipsoidLocalization::Localization(int frame_id)
{
    //获取候选索引
    std::vector<int> can_index = GetCandidatesFrameIDwithMatrix(frame_id);
    std::pair<int, double> max_similar_index_dist =  LocalizationWithCov(can_index);

    cout << "[EL]  Nearest distance: " << max_similar_index_dist.second << " btn " << inquiry_gt_id.back() << " and " << database_gt_id[max_similar_index_dist.first] << "." << endl;


    return make_pair(max_similar_index_dist.first, max_similar_index_dist.second);
}



/*---------------------------------------------------------------------------检索函数---------------------------------------------------------------------------*/

/*-------------------------原哈希部分-------------------------*/

//保存帧索引，保存key值
void EllipsoidLocalization::MakeDatabaseHashForm(int frame_id)
{   
    //获取数据索引 填入hash表
    int frame_index = database_frame_eloid.size();
    //填入hash
#if CUSTOM_HASH_ENABLE
    std::vector<HashKey> cur_frame_eigen_key;   
#else
    std::vector<int> cur_frame_eigen_key;
#endif
    for(auto non_gd_it : database_frame_eloid.back().nonground_voxel_eloid)
    {
        // cout << non_gd_it.num_exit << " " << non_gd_it.center[2] << ",";
#if CUSTOM_HASH_ENABLE
        HashKey eigen_key = GetEloidEigenKeyCustom(non_gd_it);
#else
        int eigen_key = GetEloidEigenKey(non_gd_it);
#endif
        // if(eigen_key != -1)
        {
            //判断id_bin内是否已有相同id
            std::vector<int> his_frame_id = GetHashFrameIDCustom(eigen_key);
            if(his_frame_id.empty() == 0)
            {
                for(auto his_id_it = his_frame_id.begin(); his_id_it != his_frame_id.end(); ++his_id_it)
                {
                    if(*his_id_it == frame_index)
                        break;
                    else if(his_id_it == his_frame_id.end() - 1)
                    {
#if CUSTOM_HASH_ENABLE
                        custom_frame_hash[eigen_key].push_back(frame_index);         //制作hash
#else
                        eloid_eigen_map[eigen_key].push_back(frame_index);         //制作hash
#endif
                    }
                }
            }
            else
            {
#if CUSTOM_HASH_ENABLE
                custom_frame_hash[eigen_key].push_back(frame_index);         //制作hash
#else
                eloid_eigen_map[eigen_key].push_back(frame_index);         //制作hash
#endif
            }

            cur_frame_eigen_key.push_back(eigen_key);
        }
    }
    // cout << endl;
#if CUSTOM_HASH_ENABLE
    database_custom_frame_eloid_key.push_back(cur_frame_eigen_key);
#else
    database_frame_eloid_key.push_back(cur_frame_eigen_key);
#endif
}

//只保存key值 不填入数据
void EllipsoidLocalization::MakeCurrentHashForm(int frame_id)
{
    //填入hash
#if CUSTOM_HASH_ENABLE
    std::vector<HashKey> cur_frame_eigen_key;   
#else
    std::vector<int> cur_frame_eigen_key;
#endif
    for(auto non_gd_it : inquiry_frame_eloid.back().nonground_voxel_eloid)
    {
        // cout << non_gd_it.num_exit << " " << non_gd_it.center[2] << ",";
#if CUSTOM_HASH_ENABLE
        HashKey eigen_key = GetEloidEigenKeyCustom(non_gd_it);
#else 
        int eigen_key = GetEloidEigenKey(non_gd_it);
#endif
        // if(eigen_key != -1)
        {
            cur_frame_eigen_key.push_back(eigen_key);
        }
    }
    // cout << endl;
    cur_custom_frame_eloid_key.push_back(cur_frame_eigen_key);    
}

//获取特征值的hash键值 返回 键值
int EllipsoidLocalization::GetEloidEigenKey(Ellipsoid eloid)
{
    int result = -1;


    //不同高度的点云存在二值化 + mode 候选准确率 < 0.8
    // if(eloid.mode != -1)
    //     result = (int)(eloid.num_exit + eloid.mode * 1000);

    //均值二次型 候选准确率 0.23
    Eigen::MatrixXd result_;
    Eigen::Matrix<double, 3, 1> eloid_mean_;
    eloid_mean_.col(0) = eloid.center;
    if(eloid.mode != -1)
    {
        result_ = (eloid_mean_.transpose() * eloid.cov.inverse() * eloid_mean_);
        result = (int)result_(0,0);
    }

    //均值高度 mean_z  候选准确率 0.22
    // if(eloid.mode != -1)
    //     result = (int)(eloid.center[2] * 100);


    // cout << "[EL]  Make Key  hash key is : " << result << endl;

    return result;
}

//输入键值检索hash表内的帧ID(vector<int>格式) 输出：如果有找到数据，则输出vector<int> 如果没找到，输出空的vector
vector<int> EllipsoidLocalization::GetHashFrameID(int key)
{
    auto eigen_it = eloid_eigen_map.find(key);
    if(eigen_it != eloid_eigen_map.end())
        return eigen_it->second;
    return std::vector<int>();
}

//回环检测函数 暂时先返回候选ID 获取候选值，返回真实候选id 函数格式不对，有待修改 
std::vector<int> EllipsoidLocalization::GetCandidatesFrameIDwithHash(int frame_id)
{
    static int max_frame_id = database_gt_id.size();
    int loop_id { -1 }; 
    //是否满足数量要求
    if( (int)database_frame_eloid.size() < NUM_EXCLUDE_FIRST + 1) //储存的描述符数量是否足够
    {
        // std::pair<int, float> result {loop_id, 0.0};
        return std::vector<int>(); // Early return         
    }

#if CUSTOM_HASH_ENABLE
    auto curr_key = cur_custom_frame_eloid_key.back();         //查询帧key值
#else 
    auto curr_key = inquiry_frame_eloid_key.back();         //查询帧key值
#endif
    vector<int> vote_num(max_frame_id, 0);          //存放投票数量的结构体，用于进一步判断
    for(auto it : curr_key)
    {
        std::vector<int> can_id_queue;
        //前后寻找最近距离的key 这里就要排除紧邻的id
        auto key_it = it;
        // for(int eigen_key_shift = 0; eigen_key_shift < 80; eigen_key_shift++) 
        {
            // key_it +=  ((-1) ^ eigen_key_shift) * eigen_key_shift;
            can_id_queue = GetHashFrameIDCustom(key_it);
            if(can_id_queue.empty() == 0)
            {
                //ID投票
                for(auto can_id_it : can_id_queue)
                {
                    if(can_id_it < max_frame_id)
                        vote_num[can_id_it]++;
                    // cout << "[EL]  vote id: " << can_id_it << " one ticket" << endl;
                }
                // cout << endl;
            }
        }
    }
    // cout << "[EL]  finish vote all the frame" << endl;

    //获取候选Frame index
    std::vector<int> can_index;
    std::vector<int> can_vote_num;
    for(int i = 0; i < vote_num.size(); i++)
    {
        auto max_vote_it = std::max_element(vote_num.begin(), vote_num.end());  //寻找投票ID中的最大值

        //判断是否为回环检测区域内
        int max_id = std::distance(vote_num.begin(), max_vote_it);
        if(max_id < max_frame_id)
        {
            can_index.push_back(max_id);
            can_vote_num.push_back(*max_vote_it);
        }
        //删除最大值
        *max_vote_it = -1;
        if(can_index.size() == NUM_CANDIDATES_HASH_ID)
            break;
    }


    //打印测试
    cout << "[EL]  current frame id: " << frame_id << "  ";
    cout << "[EL]  candidates id:";
    for(int i = 0; i < can_index.size(); i++)
    {
        // cout << " " << database_gt_id[can_index[i]]<< " " << can_vote_num[i];
    }
    cout << endl;

    return can_index;
}







/*-------------------------自定义哈希部分-------------------------*/

//获取特征值的hash键值 返回 键值
HashKey EllipsoidLocalization::GetEloidEigenKeyCustom(Ellipsoid eloid)
{
    HashKey result;

    //混合Key 二次型+mode 候选准确率 0.32  高度点云存在二值+mode 0.73    高度点云存在二值+mode+均值z 0.62    mode+均值z 0.35
    //最高点二次型+mode 0.31   最高点二次型 0.25
    Eigen::MatrixXd quadratic_form;
    Eigen::Matrix<double, 3, 1> eloid_mean_;
    eloid_mean_.col(0) = eloid.max_z_point - eloid.center;
    if(eloid.mode != -1)
    {
        // quadratic_form = (eloid_mean_.transpose() * eloid.cov.inverse() * eloid_mean_ * 1000);
        // result.mean_qf = (int)quadratic_form(0,0);
    
        result.cloud_exit = eloid.num_exit;
        result.mode = eloid.mode;
        // result.mean_z = (int)(eloid.center[2] * 100);
    }

    // cout << "[EL]  Make Key  hash key is : " << result.cloud_exit << endl;
    // cout << "[EL]  Make Key  hash key is : " << result.mean_qf << endl;
    // cout << "[EL]  Make Key  hash key is : " << result.mode << endl;
    // cout << "[EL]  Make Key  hash key is : " << result.mean_z << endl;

    return result;
}

//输入键值检索hash表内的帧ID(vector<int>格式) 输出：如果有找到数据，则输出vector<int> 如果没找到，输出空的vector
vector<int> EllipsoidLocalization::GetHashFrameIDCustom(HashKey key)
{
    auto eigen_it = custom_frame_hash.find(key);
    if(eigen_it != custom_frame_hash.end())
        return eigen_it->second;
    return std::vector<int>();
}





/*-------------------------SC改版描述符部分-------------------------*/

//返回在data数据中的索引，并非真实帧id，真实id需要通过database_gt_id来转换
std::vector<int> EllipsoidLocalization::GetCandidatesFrameIDwithMatrix(int frame_id)
{
    //是否满足数量要求
    if( (int)database_vertical_invkeys_mat_.size() < NUM_EXCLUDE_FIRST + 1) //储存的描述符数量是否足够
    {
        // std::pair<int, float> result {loop_id, 0.0};
        return std::vector<int>(); // Early return         
    }

    auto curr_key = cur_vertical_invkeys_mat_.back();
    
    static int struct_tree_enable = 0;
    // if(!struct_tree_enable)
    {
        vertical_invkeys_to_search_.clear();
        vertical_invkeys_to_search_.assign(database_vertical_invkeys_mat_.begin(),database_vertical_invkeys_mat_.end());
        // vertical_invkeys_to_search_.assign(database_vertical_invkeys_mat_after_hashfliter_.begin(),database_vertical_invkeys_mat_after_hashfliter_.end());
        verticalkey_tree_.reset();
        verticalkey_tree_ = std::make_unique<InvKeyTree>(VOXEL_NUM_VERTICAL, vertical_invkeys_to_search_, 10);
        struct_tree_enable = 1;
    }

    std::vector<size_t> can_index(NUM_CANDIDATES_KD_ID);
    std::vector<float> out_dist_sqr(NUM_CANDIDATES_KD_ID);

    nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_KD_ID);
    knnsearch_result.init(&can_index[0], &out_dist_sqr[0]);
    verticalkey_tree_->index->findNeighbors(knnsearch_result, &curr_key[0], nanoflann::SearchParams(10));

    vector<int> result_(can_index.begin(),can_index.end());

    //候选id打印
    // cout << "[EL]  current frame id: " << frame_id << "  ";
    // cout << "[EL]  candidates id:";
    // for(int i = 0; i < result_.size(); i++)
    // {
    //     cout << "  " << database_gt_id[result_[i]];
    // }
    // cout << endl;

    return result_;
}

/*---------------------------------------------------------------------------匹配函数---------------------------------------------------------------------------*/


/*-------------------------匹配部分 协方差矩阵对比-------------------------*/
//返回值是  索引+相似度 返回的索引，并非真实帧id，真实id需要通过database_gt_id来转换
std::pair<int, double> EllipsoidLocalization::LocalizationWithCov(std::vector<int> can_index)
{
    int max_overlap = 0;
    int max_overlap_index = 0;

    Frame_Ellipsoid cur_eloid_model = inquiry_frame_eloid.back();
    for(auto can_index_it : can_index)
    {
        double loop_eloid_num = 0;
        std::vector<Ellipsoid> can_nonground_eloid_model = database_frame_eloid[can_index_it].nonground_voxel_eloid;
        //非地面模型
        for(auto cur_nonground_model_it : cur_eloid_model.nonground_voxel_eloid)
        {
            std::pair<double, Ellipsoid> min_dist_model = FindNearestModel(cur_nonground_model_it, can_nonground_eloid_model);
            if(min_dist_model.first != 1000)
            {
                //欧式距离
                // double cov_dist = GetCovSimilarityWithEuclidean(cur_nonground_model_it.cov, min_dist_model.second.cov);
                // if(cov_dist < MAX_COV_SIMILARITY_EUCLID)
                // {
                //     loop_eloid_num++;
                // }

                //余弦距离
                double cov_cos = GetCovSimilarityWithCos(cur_nonground_model_it.cov, min_dist_model.second.cov);
                if(cov_cos < MAX_COV_SIMILARITY_COS)
                    loop_eloid_num++;

            }
        }
        double overlap = loop_eloid_num / cur_eloid_model.nonground_voxel_eloid.size();

        if(overlap > max_overlap)
        {
            max_overlap = overlap;
            max_overlap_index = can_index_it;
        }
    }

    return make_pair(max_overlap_index, max_overlap);
}

//寻找质心相差最近的模型
std::pair<double, Ellipsoid> EllipsoidLocalization::FindNearestModel(Ellipsoid inquiry_model, std::vector<Ellipsoid> can_model)
{
    double min_dist = 1000;
    Ellipsoid min_model;

    for(auto can_model_it : can_model)
    {
        double dist = (can_model_it.center - inquiry_model.center).norm();
        if(dist < min_dist)
        {   
            min_dist = dist;
            min_model = can_model_it;
        } 

    }

    return std::make_pair(min_dist, min_model);
}

//欧氏距离
double EllipsoidLocalization::GetCovSimilarityWithEuclidean(Eigen::Matrix3d inquiry_cov, Eigen::Matrix3d can_cov)
{
    //欧氏距离
    double cov_dist = (inquiry_cov - can_cov).norm();

    return cov_dist;
}

double EllipsoidLocalization::GetCovSimilarityWithCos(Matrix3d _sc1, Matrix3d _sc2)
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for ( int col_idx = 0; col_idx < _sc1.cols(); col_idx++ )
    {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);
        
        if( (col_sc1.norm() == 0) | (col_sc2.norm() == 0) )
            continue; // don't count this sector pair. 

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }
    
    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;

}


/*---------------------------------------------------------------------------获取转移矩阵---------------------------------------------------------------------------*/
//匹配点刚性转置获取转移矩阵 旋转基底为参数1特征点所在的坐标系 已弃用
Eigen::Matrix4d EllipsoidLocalization::MakeFeaturePointandGetTransformMatirx(Frame_Ellipsoid frame_eloid_1, Frame_Ellipsoid frame_eloid_2)
{
    std::vector<Eigen::Vector3d> feature_point_1 = MakeFeaturePoint(frame_eloid_1);
    std::vector<Eigen::Vector3d> feature_point_2 = MakeFeaturePoint(frame_eloid_2);

    return GetTransformMatrix(feature_point_1, feature_point_2);
    // return GetTransformMatrixwithCERE(feature_point_1, feature_point_2);
}

Eigen::Matrix4d GetTransformMatrix(std::vector<Eigen::Vector3d> feature_point_1, std::vector<Eigen::Vector3d> feature_point_2)
{
    //打印特征点集
    // cout << "GetFeaturePoint_1   make feature point: " << endl;
    // for(auto feature_p : feature_point_1)
    // {
    //     cout << feature_p.transpose() << ","; 
    // }cout << endl;
    // cout << "GetFeaturePoint_2   make feature point: " << endl;
    // for(auto feature_p : feature_point_2)
    // {
    //     cout << feature_p.transpose() << ","; 
    // }cout << endl;

    //判断特征点数量
    int point_num_;
    if(feature_point_1.size() != feature_point_2.size())
    {
        cout << "feature point num incorperate" << endl;
        Eigen::Matrix4d result = Eigen::Matrix4d::Zero();
        return result;
    }else{
        point_num_ = feature_point_1.size();
        cout << "feature point num: " << point_num_ << endl;
    }

    //求特征点中心
    Eigen::Vector3d feature_point_1_mean_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d feature_point_2_mean_ = Eigen::Vector3d::Zero();
    for(int i = 0; i < point_num_; i++)
    {
        feature_point_1_mean_ += feature_point_1[i];
        feature_point_2_mean_ += feature_point_2[i];
    }
    feature_point_1_mean_ /= point_num_;
    feature_point_2_mean_ /= point_num_;
    // cout << "feature point 1 mean: " << feature_point_1_mean_.transpose() << endl;
    // cout << "feature point 2 mean: " << feature_point_2_mean_.transpose() << endl;

    //求S = XY
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    for(int i = 0; i < point_num_; i++)
    {
        S += (feature_point_1[i] - feature_point_1_mean_) * (feature_point_2[i] - feature_point_2_mean_).transpose();
    }

    // cout << "S is: " << S << endl;


    //奇异值分解
    Eigen::MatrixXd S_ = S;
    std::vector<Eigen::MatrixXd> svd_matrix = GetSingularvalue(S_);
    Eigen::Matrix3d U = svd_matrix[0];
    Eigen::Matrix3d V = svd_matrix[1];

    // cout << "U is: " << U << endl;
    // cout << "V is: " << V << endl;


    //求旋转矩阵
    Eigen::Matrix3d M = Eigen::Matrix3d::Identity();
    M(2,2) =  (V * U.transpose()).determinant();

    Eigen::Matrix3d rotate_matrix = V * M * U.transpose();

    // cout << "rotate matrix is: " << rotate_matrix << endl;

    Eigen::Vector3d translate_vector = feature_point_2_mean_ - rotate_matrix * feature_point_1_mean_;

    // cout << "translate vector is: " << translate_vector.transpose() << endl;

    //组装转移矩阵
    Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Zero();
    transform_matrix.block(0,0,3,3) = rotate_matrix;
    transform_matrix(0,3) = translate_vector[0];
    transform_matrix(1,3) = translate_vector[1];
    transform_matrix(2,3) = translate_vector[2];
    transform_matrix(3,3) = 1;

    cout << "transform matrix is: " << endl << transform_matrix << endl;

    return transform_matrix;
}

Eigen::Matrix4d GetTransformMatrixwithCERE(std::vector<Eigen::Matrix3Xd> source_feature_point, std::vector<Eigen::Matrix3Xd> cand_feature_point, const double yaw)
{
    //使用迭代优化方法实现位姿的求解
    //获取初值
    double tf_para[3] = {10,10,0};
    // ceres::Problem problem;

    // for (int i = 0; i < source_feature_point.size(); i++)
    // {
    //     source_feature_point[i][2] = 1;
    //     cand_feature_point[i][2] = 1;
    //     Eigen::Matrix<double, 2, 1> source_fp_ = source_feature_point[i].segment(0,2);
    //     Eigen::Matrix<double, 2, 1> cand_fp_ = cand_feature_point[i].segment(0,2);
    //     problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CostFunctor, 1, 3>(new CostFunctor(source_fp_, cand_fp_)), nullptr, tf_para);
    // }

    ceres::FirstOrderFunction* function = new ceres::AutoDiffFirstOrderFunction<CostFunctor, 3>(new CostFunctor(source_feature_point, cand_feature_point));

    ceres::GradientProblem problem(function);

    //获取初值
    double min_cost = 100000;
    for (size_t i = 0; i < source_feature_point.size() - 1; i++)
    {
        double cost[1] = {0};

        // Eigen::Matrix3d init_t;
        
        Eigen::Matrix<double, 2, Eigen::Dynamic> src_pt_; // src
        Eigen::Matrix<double, 2, Eigen::Dynamic> cand_pt_; // tgt
        src_pt_.resize(2, 2);
        cand_pt_.resize(2, 2);

        //高点
        src_pt_.col(0) = source_feature_point[i].col(0).segment(0,2);
        cand_pt_.col(0) = cand_feature_point[i].col(0).segment(0,2);
        //低点
        src_pt_.col(1) = source_feature_point[i].col(1).segment(0,2);
        cand_pt_.col(1) = cand_feature_point[i].col(1).segment(0,2);

        Eigen::Matrix3d init_t = Eigen::umeyama(src_pt_, cand_pt_, false);      //svd分解求解

        double tf_data_init_[3];
        tf_data_init_[0] = init_t(0,2);
        tf_data_init_[1] = init_t(1,2);
        tf_data_init_[2] = std::atan2(init_t(1, 0), init_t(0, 0));

        // cout << "[opintimize] init trans x y and rotate yaw = ";
        // for (auto i : tf_data_init_)
        //     cout << i << " ";
        // cout << endl;

        problem.Evaluate(tf_data_init_, cost, nullptr);

        // cout << "cost: " << cost[0] << endl;

        if(cost[0] < min_cost)
        {
            memcpy(tf_para,tf_data_init_,sizeof(tf_data_init_));     
            min_cost = cost[0];
        } 
                        
    }



    // cout << "min cost parameter is: " << tf_para[0] << " " << tf_para[1] << " " << tf_para[2] << " " << endl;
    

    //配置求解器，这里有很多options的选项，自查
    ceres::GradientProblemSolver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;  //增量方程如何求解,QR分解的方法
    options.minimizer_progress_to_stdout = false; //不输出到命令行
    options.max_num_iterations = 30;

    ceres::GradientProblemSolver::Summary summary;// 优化信息
    ceres::Solve(options, problem, tf_para, &summary); // 开始优化

    // cout << summary.BriefReport() << endl;
    // cout << "[opintimize] estimated trans x y and rotate yaw = ";
    // for (auto i : tf_para)
    //     cout << i << " ";
    // cout << endl;
    

    Eigen::Isometry2d tf_est;
    tf_est.setIdentity();
    tf_est.rotate(tf_para[2]);
    tf_est.pretranslate(Eigen::Matrix<double, 2, 1>(tf_para[0], tf_para[1]));

    // cout << "[opintimize] isometry transform: " << endl << tf_est.matrix() << endl;

    Eigen::Matrix4d tf_return = Eigen::Matrix4d::Identity();
    tf_return.block<2,2>(0,0) = tf_est.matrix().block<2,2>(0,0);
    tf_return.block<2,1>(0,3) = tf_est.matrix().block<2,1>(0,2);

    // cout << "[opintimize] tr return matrix: " << endl << tf_return << endl;

    return tf_return;

}

//奇异值分解 输入 协方差  输出 U V
std::vector<Eigen::MatrixXd> GetSingularvalue(Eigen::MatrixXd bin_cov)
{
	const int rows = bin_cov.rows();
	const int cols = bin_cov.cols();

	std::vector<double> vec_;
	for (int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j){
            vec_.insert(vec_.begin() + i * cols + j, bin_cov(i, j));
        }
	}
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);
 
	// fprintf(stderr, "source matrix:\n");
	// std::cout << m << std::endl;
 
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeFullV | Eigen::ComputeFullU); // ComputeThinU | ComputeThinV //奇异值分解
	Eigen::MatrixXd singular_values = svd.singularValues(); //奇异值
	Eigen::MatrixXd left_singular_vectors = svd.matrixU();  //左奇异值向量 U
	Eigen::MatrixXd right_singular_vectors = svd.matrixV(); //右奇异值向量 V

    std::vector<Eigen::MatrixXd> result;
    result.push_back(left_singular_vectors);
    result.push_back(right_singular_vectors);
 
	return result;
}


bool feature_point_cmp(Eigen::Vector3d point1, Eigen::Vector3d point2)
{
    if(point1[2] > point2[2])
        return true;
    else
        return false;
} 

//制作带有对应关系的特征点集
std::vector<Eigen::Vector3d> EllipsoidLocalization::MakeFeaturePoint(Frame_Ellipsoid frame_eloid)
{
    //选取高度最高的前n个mean作为特征匹配点
    vector<Eigen::Vector3d> max_z_center;
    for(auto eloid_it : frame_eloid.nonground_voxel_eloid)
    {
        if(max_z_center.size() < FEATRUE_POINT_NUMS)
        {
            max_z_center.push_back(eloid_it.center);
        }else
        {
            sort(max_z_center.begin(), max_z_center.end(), feature_point_cmp);

            if(max_z_center[FEATRUE_POINT_NUMS - 1][2] < eloid_it.center[2])
            {
                max_z_center[FEATRUE_POINT_NUMS - 1] = eloid_it.center;
            }
        }
    }

    sort(max_z_center.begin(), max_z_center.end(), feature_point_cmp);

    // cout << "max center z:";
    // for(auto max_z : max_z_center)
    // {
    //     cout << " " << max_z[2];
    // }
    // cout << endl;

    return max_z_center;
}


//新转移矩阵评价函数 返回gt值与estimate值之间的转移矩阵
Eigen::Isometry2d EllipsoidLocalization::EvaculateTFWithIso(Eigen::Matrix4d can_gt, Eigen::Matrix4d src_gt, Eigen::Matrix4d est)
{
    //est处理 三维->二维
    Eigen::Isometry2d tf_cantosrc_est2;
    tf_cantosrc_est2.setIdentity();
    Eigen::Vector3d z0_est(0, 0, 1);
    Eigen::Vector3d z1_est = est.block<3, 1>(0, 2);
    Eigen::Vector3d ax_est = z0_est.cross(z1_est).normalized();
    double ang_est = acos(z0_est.dot(z1_est));              //求解三维旋转矩阵z列与z0之间的夹角，生成后面的旋转向量模块矩阵
    Eigen::AngleAxisd d_rot_est(-ang_est, ax_est);          //旋转向量模块 初始化 -ang是旋转角，ax是旋转轴 用于将三维矩阵转换成二维矩阵 

    Eigen::Matrix3d R_rectified_est = d_rot_est.matrix() * est.topLeftCorner<3, 3>();  // only top 2x2 useful 去除垂直方向上的旋转

    tf_cantosrc_est2.rotate(std::atan2(R_rectified_est(1, 0), R_rectified_est(0, 0)));
    tf_cantosrc_est2.pretranslate(Eigen::Vector2d(est.col(3).segment(0, 2)));  // only xy

    // std::cout << "[evaculate]  T delta est 2d:\n" << tf_cantosrc_est2.matrix() << std::endl;  // Note T_delta is not comparable to this

    //gt处理 三维->二维
    Eigen::Isometry3d can_gt_iso, src_gt_iso;
    can_gt_iso.matrix() = can_gt;
    src_gt_iso.matrix() = src_gt;
    Eigen::Isometry2d tf_cantosrc_gt2;
    tf_cantosrc_gt2.setIdentity();

    Eigen::Isometry3d tf_cantosrc_gt3 = can_gt_iso.inverse() * src_gt_iso;
    Eigen::Vector3d z0(0, 0, 1);
    Eigen::Vector3d z1 = tf_cantosrc_gt3.matrix().block<3, 1>(0, 2);
    Eigen::Vector3d ax = z0.cross(z1).normalized();
    double ang = acos(z0.dot(z1));        //求解三维旋转矩阵z列与z0之间的夹角，生成后面的旋转向量模块矩阵
    Eigen::AngleAxisd d_rot(-ang, ax);    //旋转向量模块 初始化 -ang是旋转角，ax是旋转轴 用于将三维矩阵转换成二维矩阵

    Eigen::Matrix3d R_rectified = d_rot.matrix() * tf_cantosrc_gt3.matrix().topLeftCorner<3, 3>();  // only top 2x2 useful 去除垂直方向上的旋转

    tf_cantosrc_gt2.rotate(std::atan2(R_rectified(1, 0), R_rectified(0, 0)));
    tf_cantosrc_gt2.pretranslate(Eigen::Vector2d(tf_cantosrc_gt3.translation().segment(0, 2)));  // only xy

    // std::cout << "[evaculate]  T delta gt 2d:\n" << tf_cantosrc_gt2.matrix() << std::endl;  // Note T_delta is not comparable to this

    Eigen::Isometry2d T_gt_est = tf_cantosrc_gt2.inverse() * tf_cantosrc_est2;
    // std::cout << "[evaculate]  T gt to est 2d:\n" << T_gt_est.matrix() << std::endl; 

    return T_gt_est;    //这是gt值与estimate值之间的旋转平移矩阵，越小越接近
}