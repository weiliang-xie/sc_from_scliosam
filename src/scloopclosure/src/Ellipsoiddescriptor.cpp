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


    //填入hash
    std::vector<int> cur_frame_eigen_key;
    for(auto non_gd_it : database_frame_eloid.back().nonground_voxel_eloid)
    {
        // cout << non_gd_it.num_exit << " " << non_gd_it.center[2] << ",";
        int eigen_key = GetEloidEigenKey(non_gd_it);
        if(eigen_key != -1)
        {
            //判断id_bin内是否已有相同id
            std::vector<int> his_frame_id = GetHashFrameID(eigen_key);
            if(his_frame_id.empty() == 0)
            {
                for(auto his_id_it = his_frame_id.begin(); his_id_it != his_frame_id.end(); ++his_id_it)
                {
                    if(*his_id_it == frame_id)
                        break;
                    else if(his_id_it == his_frame_id.end() - 1)
                    {
                        eloid_eigen_map[eigen_key].push_back(frame_id);         //制作hash
                    }
                }
            }
            else
            {
                eloid_eigen_map[eigen_key].push_back(frame_id);         //制作hash
            }

            cur_frame_eigen_key.push_back(eigen_key);
        }
    }
    // cout << endl;
    database_frame_eloid_key.push_back(cur_frame_eigen_key);
}

//当前帧描述符制作函数  输入点云数据 点云的帧id（ 从0开始算 ）
void EllipsoidLocalization::MakeInquiryEllipsoidDescriptor(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id)
{
    DivideVoxel(_scan_cloud);
    cur_frame_eloid.push_back(BulidingEllipsoidModel());
    cout << "Cur Frame Make Eloid  ground eloid num is: " << cur_frame_eloid.back().ground_voxel_eloid.size() << endl;
    cout << "Cur Frame Make Eloid  non ground eloid num is: " << cur_frame_eloid.back().nonground_voxel_eloid.size() << endl;
    cout << endl;

    //填入hash
    std::vector<int> cur_frame_eigen_key;
    for(auto non_gd_it : cur_frame_eloid.back().nonground_voxel_eloid)
    {
        // cout << non_gd_it.num_exit << " " << non_gd_it.center[2] << ",";
        int eigen_key = GetEloidEigenKey(non_gd_it);
        if(eigen_key != -1)
        {
            cur_frame_eigen_key.push_back(eigen_key);
        }
    }
    // cout << endl;
    cur_frame_eloid_key.push_back(cur_frame_eigen_key);
}

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

    if(eloid.mode != -1)
        result = (int)(eloid.num_exit + eloid.mode * 1000);

    // cout << "hash key: " << result << endl;

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

//回环检测函数 暂时先返回候选ID
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
    
    auto curr_key = cur_frame_eloid_key.back();         //查询帧key值
    vector<int> vote_num(max_frame_id, 0);          //存放投票数量的结构体，用于进一步判断
    for(auto it : curr_key)
    {
        std::vector<int> can_id_queue;
        //前后寻找最近距离的key 这里就要排除紧邻的id
        int key_it = it;
        // for(int eigen_key_shift = 0; eigen_key_shift < 80; eigen_key_shift++) 
        {
            // key_it +=  ((-1) ^ eigen_key_shift) * eigen_key_shift;
            can_id_queue = GetHashFrameID(key_it);
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
