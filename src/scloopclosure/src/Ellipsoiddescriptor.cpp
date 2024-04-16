#include "Ellipsoiddescriptor.h"

#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include <numeric>


using namespace std;

//特征数据库描述符制作函数  输入点云数据 点云的帧id（ 从0开始算 ）
void EllipsoidLocalization::MakeDatabaseEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id)
{
    DivideVoxel(_scan_cloud);
    database_frame_eloid.push_back(BulidingEllipsoidModel());
    cout << "Make Dataset Descriptor  Dataset Frame Make Eloid  ground eloid num is: " << database_frame_eloid.back().ground_voxel_eloid.size() << endl;
    cout << "Make Dataset Descriptor  Dataset Frame Make Eloid  non ground eloid num is: " << database_frame_eloid.back().nonground_voxel_eloid.size() << endl;
    cout << endl;

    //sc改版 建立描述符和键值
    std::vector<float> vertical_invkey_vec = MakeAndSaveDescriptorAndKey(frame_voxel.back().origin_voxel_data, frame_id);
    database_vertical_invkeys_mat_.push_back(vertical_invkey_vec);
    database_true_frame_id.push_back(frame_id);

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
                    if(*his_id_it == frame_id)
                        break;
                    else if(his_id_it == his_frame_id.end() - 1)
                    {
#if CUSTOM_HASH_ENABLE
                        custom_frame_hash[eigen_key].push_back(frame_id);         //制作hash
#else
                        eloid_eigen_map[eigen_key].push_back(frame_id);         //制作hash
#endif
                    }
                }
            }
            else
            {
#if CUSTOM_HASH_ENABLE
                custom_frame_hash[eigen_key].push_back(frame_id);         //制作hash
#else
                eloid_eigen_map[eigen_key].push_back(frame_id);         //制作hash
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

//当前帧描述符制作函数  输入点云数据 点云的帧id（ 从0开始算 ）
void EllipsoidLocalization::MakeInquiryEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id)
{
    DivideVoxel(_scan_cloud);
    cur_frame_eloid.push_back(BulidingEllipsoidModel());
    // cout << "Cur Frame Make Eloid  ground eloid num is: " << cur_frame_eloid.back().ground_voxel_eloid.size() << endl;
    // cout << "Cur Frame Make Eloid  non ground eloid num is: " << cur_frame_eloid.back().nonground_voxel_eloid.size() << endl;
    cout << endl;

    //sc改版 建立描述符和键值
    std::vector<float> vertical_invkey_vec = MakeAndSaveDescriptorAndKey(frame_voxel.back().origin_voxel_data, frame_id);
    cur_vertical_invkeys_mat_.push_back(vertical_invkey_vec);

    //填入hash
#if CUSTOM_HASH_ENABLE
    std::vector<HashKey> cur_frame_eigen_key;   
#else
    std::vector<int> cur_frame_eigen_key;
#endif
    for(auto non_gd_it : cur_frame_eloid.back().nonground_voxel_eloid)
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


//定位函数
std::pair<int, float> EllipsoidLocalization::Localization(std::vector<int> can_id)
{
    
}





/*-------------------------原哈希部分-------------------------*/

//获取特征值的hash键值 返回 键值
int EllipsoidLocalization::GetEloidEigenKey(Ellipsoid eloid)
{
    int result = -1;

    // if(eloid.mode == 1)
    // {
    //     result = (int)(((eloid.eigen[0] - eloid.eigen[1]) / eloid.eigen[0] + round(eloid.center[2])) * 100000);
    // }
    // else if(eloid.mode == 2)
    // {
    //     result = (int)(((eloid.eigen[1] - eloid.eigen[2]) / eloid.eigen[0] + round(eloid.center[2])) * 100000);
    // }
    // else if(eloid.mode == 3)
    // {
    //     result = (int)(((double)eloid.mode + eloid.eigen[2] / eloid.eigen[0] + round(eloid.center[2])) * 100000);
    // }

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


    // cout << "Make Key  hash key is : " << result << endl;

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

//回环检测函数 暂时先返回候选ID 获取候选值，函数格式不对，有待修改
std::vector<int> EllipsoidLocalization::DetectLoopClosureID(int frame_id)
{
    static int max_frame_id = 4541;
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
    auto curr_key = cur_frame_eloid_key.back();         //查询帧key值
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
                    // cout << "vote id: " << can_id_it << " one ticket" << endl;
                }
                // cout << endl;
            }
        }
    }
    // cout << "finish vote all the frame" << endl;

    //获取候选Frame ID
    std::vector<int> can_id;
    std::vector<int> can_vote_num;
    for(int i = 0; i < vote_num.size(); i++)
    {
        auto max_vote_it = std::max_element(vote_num.begin(), vote_num.end());  //寻找投票ID中的最大值

        //判断是否为回环检测区域内
        int max_id = std::distance(vote_num.begin(), max_vote_it);
        if(max_id < max_frame_id)
        {
            can_id.push_back(max_id);
            can_vote_num.push_back(*max_vote_it);
        }
        //删除最大值
        *max_vote_it = -1;
        if(can_id.size() == NUM_CANDIDATES_HASH_ID)
            break;
    }


    //打印测试
    cout << "current frame id: " << frame_id << "  ";
    cout << "candidates id:";
    for(int i = 0; i < can_id.size(); i++)
    {
        cout << "  " << can_id[i]<< " " << can_vote_num[i];
    }
    cout << endl;


    std::pair<int, float> result {loop_id, 0};
    return can_id;
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
        quadratic_form = (eloid_mean_.transpose() * eloid.cov.inverse() * eloid_mean_ * 1000);
        result.mean_qf = (int)quadratic_form(0,0);
    
        result.cloud_exit = eloid.num_exit;
        // result.mode = eloid.mode;
        // result.mean_z = (int)(eloid.center[2] * 100);
    }

    // cout << "Make Key  hash key is : " << result.cloud_exit << endl;
    // cout << "Make Key  hash key is : " << result.mean_qf << endl;
    // cout << "Make Key  hash key is : " << result.mode << endl;
    // cout << "Make Key  hash key is : " << result.mean_z << endl;

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

std::vector<int> EllipsoidLocalization::GetCandidatesFrameID(int frame_id)
{
    //是否满足数量要求
    if( (int)database_vertical_invkeys_mat_.size() < NUM_EXCLUDE_FIRST + 1) //储存的描述符数量是否足够
    {
        // std::pair<int, float> result {loop_id, 0.0};
        return std::vector<int>(); // Early return         
    }

    auto curr_key = cur_vertical_invkeys_mat_.back();
    
    static int struct_tree_enable = 0;
    if(!struct_tree_enable)
    {
        vertical_invkeys_to_search_.clear();
        vertical_invkeys_to_search_.assign(database_vertical_invkeys_mat_.begin(),database_vertical_invkeys_mat_.end());
        verticalkey_tree_.reset();
        verticalkey_tree_ = std::make_unique<InvKeyTree>(VOXEL_NUM_VERTICAL, vertical_invkeys_to_search_, 10);
        struct_tree_enable = 1;
    }

    std::vector<size_t> can_index(NUM_CANDIDATES_HASH_ID);
    std::vector<float> out_dist_sqr(NUM_CANDIDATES_HASH_ID);

    nanoflann::KNNResultSet<float> knnsearch_result(NUM_CANDIDATES_HASH_ID);
    knnsearch_result.init(&can_index[0], &out_dist_sqr[0]);
    verticalkey_tree_->index->findNeighbors(knnsearch_result, &curr_key[0], nanoflann::SearchParams(10));

    vector<int> result_;

    for(auto can_index_it : can_index)
    {
        result_.push_back(database_true_frame_id[can_index_it]);
    }

    cout << "current frame id: " << frame_id << "  ";
    cout << "candidates id:";
    for(int i = 0; i < result_.size(); i++)
    {
        cout << "  " << result_[i];
    }
    cout << endl;
    return result_;
}



