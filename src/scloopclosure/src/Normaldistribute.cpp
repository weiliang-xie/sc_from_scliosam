#include "Normaldistribute.h"
#include "Scancontext.h"

#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include <numeric>

using namespace std;

void ClouDistrubutionVisualization(std::vector<Eigen::Vector3d> leaf_cloud, Eigen::MatrixXd axis_all, Eigen::Vector3d center, int index);
extern Eigen::Matrix4d GetTransformMatrix(vector<Eigen::Vector3d> feature_point_1, vector<Eigen::Vector3d> feature_point_2);
extern Eigen::Matrix4d GetTransformMatrixwithCERE(std::vector<Eigen::Matrix3Xd> source_feature_point, std::vector<Eigen::Matrix3Xd> cand_feature_point, const double yaw = 0);


void NDManager::NDmakeAndSaveInquiryScancontextAndKeys(pcl::PointCloud<SCPointType> & _scan_cloud, int frame_id)
{
    TicToc t_making_desc;
    t_making_desc.tic();
    // cout << "[ND] make descriptor matrix and key" << endl;
    std::pair<MatrixXd, MatrixXd> sc = NDmakeScancontext(_scan_cloud); // v1     //制作NDSC描述矩阵

    step_timecost[0] += t_making_desc.toc();
    // printf("[Make descriptor] Time cost: %7.5fs\r\n", t_making_desc.toc());

    Eigen::MatrixXd ringkey = NDmakeRingkeyFromScancontext( sc.first ); //制作ring键值 每行平均值
    Eigen::MatrixXd sectorkey = NDmakeSectorkeyFromScancontext( sc.first ); //制作sector键值 每列平均值
    std::vector<float> polarcontext_invkey_vec = eig2stdvec( ringkey ); //将ring键值传入vector容器(以数组的形式)

    inquiry_polarcontexts_.push_back( sc.first );     //保存描述矩阵
    polarcontext_invkeys_.push_back( ringkey ); //保存ring键值
    polarcontext_vkeys_.push_back( sectorkey ); //保存sector键值
    inquiry_polarcontext_invkeys_mat_.push_back( polarcontext_invkey_vec ); //保存vector格式的ring键值 
    // inquiry_gt_id.push_back(frame_id); 

    //制作点云分布键值
    // std::vector<int> pt_distribute_key = NDmakeDistributekeyFromScancontext(sc.second);

    //建立点云分布特征哈希数据库
    // MakeDatabaseHashForm(frame_id, pt_distribute_key);
}


std::pair<MatrixXd, MatrixXd> NDManager::NDmakeScancontext(pcl::PointCloud<SCPointType> & _scan_cloud)
{

    // main
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(ND_PC_NUM_RING, ND_PC_NUM_SECTOR);  //创建空（无点云）的SC矩阵
    MatrixXd desc_second = NO_POINT * MatrixXd::Ones(ND_PC_NUM_RING, ND_PC_NUM_SECTOR);  //创建空（无点云）的SC矩阵

    int num_pts_scan_down = _scan_cloud.points.size();
    std::vector<std::vector<std::vector<Eigen::Vector3d> > > nd_uint_point_queue
                                                            (ND_PC_NUM_RING,std::vector<std::vector<Eigen::Vector3d> >(ND_PC_NUM_SECTOR,std::vector<Eigen::Vector3d>(0)));  //单元内的点的数组

    SCPointType pt;
    float azim_angle, azim_range; // wihtin 2d plane
    int ring_idx, sector_idx;
    double min_high_pt_ = 10, max_high_pt_ = -10;

    for(int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        pt.x = _scan_cloud.points[pt_idx].x;                 //循环获取帧内各点
        pt.y = _scan_cloud.points[pt_idx].y;
        pt.z = _scan_cloud.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).

        // xyz to ring, sector
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);   //求点到传感器中心的距离
        azim_angle = xy2theta(pt.x, pt.y);              //输出theta角        

        if( azim_range > ND_PC_MAX_RADIUS )                //判断点距离传感器距离是否超出最大值
            continue;
        
        //寻找最低、最高点
        if(pt.z < min_high_pt_)
            min_high_pt_ = pt.z;
        if(pt.z > max_high_pt_)
            max_high_pt_ = pt.z;
        
        // cout << "[ND] computer bin index" << endl;
        ring_idx = std::max( std::min( ND_PC_NUM_RING, int(ceil( (azim_range / ND_PC_MAX_RADIUS) * ND_PC_NUM_RING )) ), 1 );    //从1开始
        sector_idx = std::max( std::min( ND_PC_NUM_SECTOR, int(ceil( (azim_angle / 360.0) * ND_PC_NUM_SECTOR )) ), 1 );
        
        // cout << "printf ring index: " << ring_idx << " sector index: " << sector_idx << endl;
        Eigen::Vector3d piont_data = {pt.x,pt.y,pt.z};
        nd_uint_point_queue[ring_idx - 1][sector_idx - 1].push_back(piont_data);
    }

    // cout << "[ND] finish point classcify" << endl;


    //点云处理
    std::vector<class Voxel_Ellipsoid> cloud_voxel_eloid_(0);

    //求点云分布的区间高度
    double pt_distri_block = (max_high_pt_ - min_high_pt_) / PT_DISTRIBUTE_BLOCK_NUM;

    ring_idx = -1;
    // sector_idx = -1;
    //顺序遍历体素
    for(auto &ring_it : nd_uint_point_queue)
    {
        sector_idx = -1;
        ring_idx++;
        for(auto &bin_it : ring_it)
        {
            sector_idx++;
            class Voxel_Ellipsoid bin_eloid;
            if(bin_it.size()){           
                std::pair<Eigen::MatrixXd,Eigen::MatrixXd> bin_mean_cov_;
                bin_eloid.point_num = bin_it.size();
                Eigen::Vector3d center_ = {0,0,0};

                //为降低耗时 简版获取均值 点云分布 最大高度
                int pt_distri_block_arry[PT_DISTRIBUTE_BLOCK_NUM] {};
                for(auto bin_pt_ : bin_it)
                {
                    center_[0] += bin_pt_[0];
                    center_[1] += bin_pt_[1];
                    center_[2] += bin_pt_[2];
                    if(bin_pt_[2] > bin_eloid.max_high_z)
                        bin_eloid.max_high_z = bin_pt_[2];
                    
                    pt_distri_block_arry[(int)floor((bin_pt_[2] - min_high_pt_) / pt_distri_block)]++;      //TODO 可以考虑过滤
                }
                center_ /= bin_eloid.point_num;
                bin_eloid.center.x = center_[0];
                bin_eloid.center.y = center_[1];
                bin_eloid.center.z = center_[2];

                for(int i = 0; i < PT_DISTRIBUTE_BLOCK_NUM; i++)
                {
                    if(pt_distri_block_arry[i] != 0)
                        bin_eloid.pt_distri += pow(2,i);
                }

                //打印层次计数值
                // cout << "[ND]   [make discriptor]  bin distribute num in layer: ";
                // for(auto num_it : pt_distri_block_arry)
                //     cout << num_it << ",";
                // cout << endl; 

                // cout << "[ND]   [make discriptor]  bin distribute data: " << bin_eloid.pt_distri << endl;

                bin_mean_cov_ = NDGetCovarMatrix(bin_it);

                //构建点云的体素椭球
                std::pair<std::vector<double>,Eigen::MatrixXd> bin_eigen_;
                //均值填充
                // bin_eloid.center.x = bin_mean_cov_.first(0,0);
                // bin_eloid.center.y = bin_mean_cov_.first(0,1);
                // bin_eloid.center.z = bin_mean_cov_.first(0,2);
                bin_eloid.cov = bin_mean_cov_.second;
                bin_eloid.num = bin_it.size();

                bin_eigen_ = NDGetEigenvalues(bin_eloid.cov);
                bin_eloid.axis_length = bin_eigen_.first;
                // bin_eloid.axis = bin_eigen_.second;

                //测试 最大高度
                // for(int i = 0; i < bin_it.size(); i++)
                // {
                //     if(bin_it[i][2] > bin_eloid.max_h_center.z)
                //     {
                //         bin_eloid.max_h_center.x = bin_it[i][0];
                //         bin_eloid.max_h_center.y = bin_it[i][1];
                //         bin_eloid.max_h_center.z = bin_it[i][2];
                //     }
                // }
                desc_second(ring_idx,sector_idx) = bin_eloid.pt_distri;
                if(!NDFilterVoxelellipsoid(bin_eloid)) 
                {   
                    //体素椭球为无效模型 筛选小于一定数量的椭球
                    // bin_it.clear();
                    //不够点云数， 填入最高的高度值
                    // desc(ring_idx,sector_idx) = bin_eloid.max_high_z;
                }else{   

                    //计算椭球扁平程度
                    double flat_ratio = 0;
                    if(bin_eloid.axis_length.size() != 3){
                        cout << "The singular value is error !!!" << endl;
                    }else{
                        if(bin_eloid.axis_length[2] != 0)
                            flat_ratio = bin_eloid.axis_length[2] / bin_eloid.axis_length[0];
                        else flat_ratio = 0;
                    }

                    //填充描述矩阵
                    desc(ring_idx,sector_idx) = flat_ratio;
                    // desc(ring_idx,sector_idx) = atan2(double(bin_eloid.center.z), double((ring_idx+1) * ND_PC_UNIT_RINGGAP)) * 100;


                    // //测试，可视化体素内的点云分布  
                    // Eigen::Vector3d center_data;
                    // center_data[0] = bin_eloid.center.x;
                    // center_data[1] = bin_eloid.center.y;
                    // center_data[2] = bin_eloid.center.z;
                    // int index = ring_idx * 60 + sector_idx;
                    // ClouDistrubutionVisualization(bin_it,bin_eloid.axis,center_data, index);
                }
            }

            cloud_voxel_eloid_.push_back(bin_eloid);    //储存单个体素椭球模型

        }
    }
    std::vector<Eigen::Matrix3Xd> feature_p_ = NDGetFeaturePoint(cloud_voxel_eloid_);
    cloud_feature_set.push_back(feature_p_);

    //传入序列 储存当帧点云体素椭球
    cloud_voxel_eloid.push_back(cloud_voxel_eloid_);
    // NDSaveVoxelellipsoidData(cloud_voxel_eloid_, cloud_voxel_eloid.size()-1); // 保存当帧体素椭球数据

    //将无点的网格描述值 = 0
    for ( int row_idx = 0; row_idx < desc.rows(); row_idx++ )
        for ( int col_idx = 0; col_idx < desc.cols(); col_idx++ )
        {
            if( desc(row_idx, col_idx) == NO_POINT )
                desc(row_idx, col_idx) = 0;
            if( desc_second(row_idx, col_idx) == NO_POINT )
                desc_second(row_idx, col_idx) = 0;
        }

    // cout << "[ND]  finish make ND descriptor" << endl;

    // t_making_desc.toc("PolarContext making");

    //测试 列临值求平均
    Eigen::MatrixXd avg_desc = MatrixXd::Zero(ND_PC_NUM_RING, ND_PC_NUM_SECTOR);
    for(int i = 1; i < ND_PC_NUM_RING - 1; i++)
    {
        for(int j = 0; j < ND_PC_NUM_SECTOR; j++)
        {
            avg_desc(i,j) = (desc(i-1,j) + desc(i,j) + desc(i+1,j)) / 3;
        }
    }
    avg_desc.row(0) = desc.row(0);
    avg_desc.row(ND_PC_NUM_RING - 1) = desc.row(ND_PC_NUM_RING - 1);

    // 打印矩阵
    // cout <<"[ND] [make descriptor] distribute matrix: " << endl;
    // cout << desc.cast<float>() << endl;

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> result = {desc, desc_second};
    return result;
}

//制作ring key
MatrixXd NDManager::NDmakeRingkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: rowwise mean vector
    */
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for ( int row_idx = 0; row_idx < _desc.rows(); row_idx++ )
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();    //求解每行的平均值
    }

    return invariant_key;
} // NDManager::makeRingkeyFromScancontext

//制作distribute key  提取每行的分布键值（将各行按序号分区，各个高度层次上若有分区内有一个体素存在点云，则该层次下整个分区存在点云，汇总各个分区的点云分布，得到键值）
std::vector<int> NDManager::NDmakeDistributekeyFromScancontext( Eigen::MatrixXd _desc )
{
    vector<int> key;
    Eigen::MatrixXd desc_ = _desc;
    for ( int row_idx = 0; row_idx < desc_.rows(); row_idx++ )
    {
        int distri_key_block_arry[PT_DISTRIBUTE_BLOCK_NUM] = {};
        for(int block_idx = 0; block_idx < DISTRIBUTE_KEY_BLOCK_NUM; block_idx++)
        {
            int layer = PT_DISTRIBUTE_BLOCK_NUM - 1;
            while(layer >= 0)
            {
                bool layer_finish_enable = 0;
                for(int i = 0; i < ND_PC_NUM_SECTOR / DISTRIBUTE_KEY_BLOCK_NUM; i++)
                {
                    if(desc_(row_idx, block_idx * DISTRIBUTE_KEY_BLOCK_NUM + i) >= pow(2,layer))
                    {
                        desc_(row_idx, block_idx * DISTRIBUTE_KEY_BLOCK_NUM + i) -= pow(2,layer);

                        if(layer_finish_enable == 0)
                        {
                            distri_key_block_arry[layer] += 1;
                            layer_finish_enable = 1;
                        }
                    }    
                }
                layer--;
            }            
        }

        //求解每行的点云分布键值
        int key_ = 0;
        for(int i = 0; i < PT_DISTRIBUTE_BLOCK_NUM; i++)
        {
            key_ += pow(DISTRIBUTE_KEY_BLOCK_NUM + 1,i) * distri_key_block_arry[i];
        }
        key.push_back(key_);    //求解每行的点云分布

        // //打印层次计数值
        // cout << "[ND]   [make discriptor]  distribute key num in layer: ";
        // for(auto num_it : distri_key_block_arry)
        //     cout << num_it << ",";
        // cout << endl;        
        // cout << "[ND]   [make discriptor]  bin distribute key: " << key_ << endl;

    }

    // //打印键值
    // cout << "[ND]   [make discriptor]  distribute key: ";
    // for(auto key_it : key)
    //     cout << key_it << ",";
    // cout << endl;

    return key;
} 

//保存帧索引，保存key值
void NDManager::MakeDatabaseHashForm(int frame_id, vector<int> distribute_key)
{   
    //填入hash
    for(auto eigen_key : distribute_key)
    {
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
                        distribute_map[eigen_key].push_back(frame_id);         //制作hash
                    }
                }
            }
            else
            {
                distribute_map[eigen_key].push_back(frame_id);         //制作hash
            }

        }
    }
    //保存当前帧的key值
    distribute_frame_key.push_back(distribute_key);
}

//输入键值检索hash表内的帧ID(vector<int>格式) 输出：如果有找到数据，则输出vector<int> 如果没找到，输出空的vector
std::vector<int> NDManager::GetHashFrameIDCustom(int key)
{
    auto eigen_it = distribute_map.find(key);
    if(eigen_it != distribute_map.end())
        return eigen_it->second;
    return std::vector<int>();
}

//回环检测 提取hash中投票最多的id
std::vector<size_t> NDManager::SearchCandidateIDwithHash(std::vector<int> cur_key)
{
    static int max_frame_id = distribute_frame_key.size();
    int loop_id { -1 }; 
    //是否满足数量要求


    vector<int> vote_num(max_frame_id, 0);          //存放投票数量的结构体，用于进一步判断
    for(auto it : cur_key)
    {
        std::vector<int> can_id_queue;
        //前后寻找最近距离的key 这里就要排除紧邻的id
        // for(int eigen_key_shift = 0; eigen_key_shift < 80; eigen_key_shift++) 
        {
            // key_it +=  ((-1) ^ eigen_key_shift) * eigen_key_shift;
            can_id_queue = GetHashFrameIDCustom(it);
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
    std::vector<size_t> can_index;
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
        if(can_index.size() == ND_NUM_CANDIDATES_FROM_TREE)
            break;
    }


    //打印测试
    cout << "[ND]  candidates id:";
    for(int i = 0; i < can_index.size(); i++)
    {
        cout<< " " << can_index[i];
    }
    cout << endl;

    return can_index;    
}

//制作sector key
MatrixXd NDManager::NDmakeSectorkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: columnwise mean vector
    */
    Eigen::MatrixXd variant_key(1, _desc.cols());
    for ( int col_idx = 0; col_idx < _desc.cols(); col_idx++ )
    {
        Eigen::MatrixXd curr_col = _desc.col(col_idx);
        variant_key(0, col_idx) = curr_col.mean();
    }

    return variant_key;
} // NDManager::makeSectorkeyFromScancontext

//返回database index 并非真实帧id
std::pair<int, float> NDManager::NDdetectLoopClosureID ( void )
{
    int loop_id { -1 }; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")

    // cout << "enter descriptor detect" << endl;

    /* 
     * step 1: candidates from ringkey tree_
     */
    if( (int)inquiry_polarcontext_invkeys_mat_.size() < ND_NUM_EXCLUDE_RECENT + 1) //储存的描述符数量是否足够
    {
        std::pair<int, float> result {loop_id, 0.0};
        cout << "[ND] descriptor number is not enough" << endl;
        return result; // Early return 
    }

    auto cur_key = inquiry_polarcontext_invkeys_mat_.back(); // current observation (query)    //取出vector格式的ring键值
    auto curr_desc = inquiry_polarcontexts_.back(); // current observation (query)              //取出描述矩阵,最近的
    auto curr_veloid = cloud_voxel_eloid.back();                                        //取出体素椭球序列,最近的


    // tree_ reconstruction (not mandatory to make everytime) //重建kd树 10秒重建一次树
    if( tree_making_period_conter % 10 == 0) // to save computation cost
    {
        TicToc t_tree_construction;
        t_tree_construction.tic();

        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.assign( inquiry_polarcontext_invkeys_mat_.begin(), inquiry_polarcontext_invkeys_mat_.end() - ND_NUM_EXCLUDE_RECENT ) ; //去除最近的50个描述符

        //复位polarcontext_tree_指针，并开辟一段动态内存用于存放 同时构造出来的KDTreeVectorOfVectorsAdaptor类
        polarcontext_tree_.reset(); 
        polarcontext_tree_ = std::make_unique<InvKeyTree>(ND_PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */ );  //开辟内存同时构造InvKeyTree类 并在构造函数内完成重建树
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
        // t_tree_construction.toc("Tree construction");

        // cout << "[Tree construction] Time cost: " << t_tree_construction.toc("Tree construction") << endl;
        step_timecost[1] += t_tree_construction.toc();
        // printf("[Tree construction] Time cost: %7.5fs\r\n", t_tree_construction.toc());
    }
    tree_making_period_conter += 1;
        
    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    // knn search
    std::vector<size_t> candidate_indexes( ND_NUM_CANDIDATES_FROM_TREE ); 
    std::vector<float> out_dists_sqr( ND_NUM_CANDIDATES_FROM_TREE );

    TicToc t_tree_search;
    t_tree_search.tic();

    nanoflann::KNNResultSet<float> knnsearch_result( ND_NUM_CANDIDATES_FROM_TREE );
    knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );
    polarcontext_tree_->index->findNeighbors( knnsearch_result, &cur_key[0] /* query */, nanoflann::SearchParams(10) );    //传入当前描述符 kd树搜索
    // t_tree_search.toc("Tree search");
        
    step_timecost[2] += t_tree_search.toc();
    // printf("[Tree search] Time cost: %7.5fs\r\n", t_tree_search.toc());

    // 测试 自制带真值的候选帧集 结果 与sc一模一样
    // int gt_loop_id = -1;
    // for(int i = 0; i < loopclosure_gt_index_copy.size(); i++)
    // {
    //     if(loopclosure_gt_index_copy[i].first == int(inquiry_polarcontexts_.size() - 1))
    //     {
    //         gt_loop_id = loopclosure_gt_index_copy[i].second;
    //         break;
    //     }
    // }

    // if(gt_loop_id >= 0)
    // {
    //     // candidate_indexes.clear();
    //     // candidate_indexes.assign(ND_NUM_CANDIDATES_FROM_TREE, 0);
    //     for(int i = 0; i < candidate_indexes.size(); i++)
    //     {
    //         if(candidate_indexes[i] == gt_loop_id)
    //             break;
    //         if(i == candidate_indexes.size() - 1)
    //         {
    //             candidate_indexes[0] = gt_loop_id;
    //             cout << "cur id: " << inquiry_polarcontexts_.size() - 1 << " can id replace: " << gt_loop_id << endl;
    //         }
    //     }
    //     // for(int i = 1; i < candidate_indexes.size(); i++)
    //     // {
    //     //     if(inquiry_polarcontexts_.size() - ND_NUM_EXCLUDE_RECENT - i == 0)
    //     //         break;  
    //     //     candidate_indexes[i] = inquiry_polarcontexts_.size() - ND_NUM_EXCLUDE_RECENT - i;
    //     // }
    // }

    //点云分布 hash vote 回环检测
    // candidate_indexes = SearchCandidateIDwithHash(distribute_frame_key.back());


    /* 
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    TicToc t_calc_dist;   
    for ( int candidate_iter_idx = 0; candidate_iter_idx < ND_NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++ )
    {
        MatrixXd polarcontext_candidate = inquiry_polarcontexts_[ candidate_indexes[candidate_iter_idx] ];
        // std::vector<class Voxel_Ellipsoid> voxel_eloid_candidate = cloud_voxel_eloid[ candidate_indexes[candidate_iter_idx] ];

        //测试 调整矩阵
        // curr_desc = NDAdjustDescriptor(NDGetGtTranslateVector(candidate_indexes[candidate_iter_idx], inquiry_polarcontexts_.size() - 1), curr_desc);

        std::pair<double, int> sc_dist_result = NDdistanceBtnScanContext( curr_desc, polarcontext_candidate );    //返回相似度值和列向量偏移量
        // cur_frame_id = polarcontexts_.size() - 1;
        // can_frame_id = candidate_indexes[candidate_iter_idx];
        // std::pair<double, int> sc_dist_result = NDdistancevoxeleloid(curr_desc, polarcontext_candidate, curr_veloid, voxel_eloid_candidate);
        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if( candidate_dist < min_dist ) //获取到最相似的描述符的相似度值 偏移量 和 描述符id
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }
    // t_calc_dist.toc("Distance calc");    

    //储存回环帧的id和相似度距离
    std::pair<int,float> data{(inquiry_polarcontexts_.size() - 1),min_dist};      
    loopclosure_id_and_dist.push_back(data);

    // cout << "[ND]   [descriptor match] the best loop id: " << nn_idx << " min distance: " << min_dist << endl;

    /* 
     * loop threshold check
     */
    if(min_dist <= ND_SC_DIST_THRES)
    {
        loop_id = nn_idx; 

        std::cout.precision(3); 
        cout << "[ND]  [Loop found] Nearest distance: " << min_dist << " btn " << inquiry_polarcontexts_.size() - 1 << " and " << nn_idx << "." << endl;
        // cout << "[ND]  [Loop found] yaw diff: " << nn_align * ND_PC_UNIT_SECTORANGLE << " deg." << endl;
    }

    // To do: return also nn_align (i.e., yaw diff)
    float yaw_diff_rad = deg2rad(nn_align * ND_PC_UNIT_SECTORANGLE);
    std::pair<int, float> result {loop_id, min_dist};   //返回达到回环阈值的描述符id和相似度

    step_timecost[3] += t_calc_dist.toc();

    // printf("[Descriptor match] Time cost: %7.5fs\r\n", t_calc_dist.toc());


    return result;

} // NDManager::NDdetectLoopClosureID

//测试 调整描述矩阵 匹配与候选相差距离大于体素一半时，对矩阵调整相差方向上的描述值
Eigen::Vector2d NDManager::NDGetGtTranslateVector(int can_id, int src_id)
{
    Eigen::Matrix4d can_pose = pose_ground_truth_copy[can_id];
    Eigen::Matrix4d src_pose = pose_ground_truth_copy[src_id];

    Eigen::Matrix4d trans_ = can_pose * src_pose.inverse();

    Eigen::Vector2d translate_ = trans_.col(3).segment(0,2);

    cout << "[TEST]     [get translate] toward is: " << translate_.transpose() << endl;
    return translate_;
}
Eigen::MatrixXd NDManager::NDAdjustDescriptor(Eigen::Vector2d can_toward, Eigen::MatrixXd src_matrix)
{
    Eigen::MatrixXd result;
    result = src_matrix;
    //小于体素的一半，不考虑调整
    if(can_toward.norm() < ND_PC_UNIT_RINGGAP / 2.0)
        return result;

    int ring_idx, sector_idx;
    double azim_range, azim_angle;
    azim_angle = xy2theta(can_toward[0], can_toward[1]);              //输出theta角  
    sector_idx = std::max( std::min( ND_PC_NUM_SECTOR, int(ceil( (azim_angle / 360.0) * ND_PC_NUM_SECTOR )) ), 1 );

    int i = 0;
    while(i <= 4)
    {
        cout << (sector_idx - 1 + i + ND_PC_NUM_SECTOR) % ND_PC_NUM_SECTOR << endl;
        cout << (sector_idx - 1 + i + (ND_PC_NUM_SECTOR / 2)) % ND_PC_NUM_SECTOR << endl;

        Eigen::VectorXd src_toward_col = src_matrix.col((sector_idx - 1 + i + ND_PC_NUM_SECTOR) % ND_PC_NUM_SECTOR);
        Eigen::VectorXd src_back_col = src_matrix.col((sector_idx - 1 + i + (ND_PC_NUM_SECTOR / 2)) % ND_PC_NUM_SECTOR);

        Eigen::VectorXd new_src_toward_col, new_src_back_col;
        new_src_toward_col.resize(ND_PC_NUM_RING);
        new_src_back_col.resize(ND_PC_NUM_RING);
        new_src_toward_col.segment(0,15) = src_toward_col.segment(1,15);
        new_src_back_col[0] = src_toward_col[0];
        new_src_toward_col[15] = src_toward_col[15];
        new_src_back_col.segment(1,15) = src_back_col.segment(0,15);
 
        result.col((sector_idx - 1 + i + ND_PC_NUM_SECTOR) % ND_PC_NUM_SECTOR).segment(0,ND_PC_NUM_RING) = new_src_toward_col.segment(0,ND_PC_NUM_RING);
        result.col((sector_idx - 1 + i + ND_PC_NUM_SECTOR / 2) % ND_PC_NUM_SECTOR).segment(0,ND_PC_NUM_RING) = new_src_back_col.segment(0,ND_PC_NUM_RING);

        //来回处理两边的
        if(i == 0)
            i++;
        else{
            if(i < 0)
            {   
                i *= -1;
                i++;
            }
            else
                i *= -1;
        }
    }

    cout << "[TEST]     [get adjustmatrix] finish! " << endl;
    return result;
}

double NDManager::NDdistDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;

    // //与行偏移配合，去除0行和最远1行，测试（全0版）
    // _sc1.row(0).setZero(); _sc2.row(0).setZero();
    // _sc1.row(_sc1.rows() - 1).setZero(); _sc2.row(_sc2.rows() - 1).setZero();

    for ( int col_idx = 0; col_idx < _sc1.cols(); col_idx++ )
    {
        VectorXd col_sc1 = _sc1.col(col_idx);
        VectorXd col_sc2 = _sc2.col(col_idx);
        
        if( (col_sc1.norm() == 0) | (col_sc2.norm() == 0) )
            continue; // don't count this sector pair. 

        // //加权弱化远距离编码值 测试
        // double uint_weight = 1.0 / double((col_sc1.size() + 1) * col_sc1.size() / 2);
        // for(int i = 0; i < col_sc1.size(); i++)
        // {
        //     col_sc1[i] *= (col_sc1.size() - i) * uint_weight;
        //     col_sc2[i] *= (col_sc1.size() - i) * uint_weight;
        // }

        double sector_similarity = col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());    //余弦
        // double sector_similarity = (col_sc1 - col_sc2).norm();                               //欧式距离

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }
    
    if(num_eff_cols == 0) num_eff_cols = _sc1.cols();
    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;
    // return sc_sim;

} // distDirectSC


int NDManager::NDfastAlignUsingVkey( MatrixXd & _vkey1, MatrixXd & _vkey2)
{
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    for ( int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++ )
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

        double cur_diff_norm = vkey_diff.norm();
        if( cur_diff_norm < min_veky_diff_norm )
        {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;

} // fastAlignUsingVkey

//SC原版列位移匹配函数
std::pair<double, int> NDManager::NDdistanceBtnScanContext( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    // 1. fast align using variant key (not in original IROS18)
    MatrixXd vkey_sc1 = NDmakeSectorkeyFromScancontext( _sc1 );   //计算列的均值并以行的形式返回整个矩阵的列向量均值
    MatrixXd vkey_sc2 = NDmakeSectorkeyFromScancontext( _sc2 );
    int argmin_vkey_shift = NDfastAlignUsingVkey( vkey_sc1, vkey_sc2 );   //将列均值向右移动并使用F-范数进行对比，返回偏移值

    const int SEARCH_RADIUS = round( 0.5 * ND_SEARCH_RATIO * _sc1.cols() ); // a half of search range  //搜索范围
    std::vector<int> shift_idx_search_space { argmin_vkey_shift };
    for ( int ii = 1; ii < SEARCH_RADIUS + 1; ii++ )
    {
        shift_idx_search_space.push_back( (argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols() );
        shift_idx_search_space.push_back( (argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols() );
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());    //在vector中填入以获取到的偏移量为中心，前后范围为搜索范围的待匹配偏移量

    // 2. fast columnwise diff 
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for ( int num_shift: shift_idx_search_space )
    {
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);  //列位移函数

        // //行偏移 测试
        // double cur_sc_dist = 10000000;
        // for(int i = -1; i < 2; i++)
        // {
        //     MatrixXd sc2_shifted_ringshifted = ringshift(sc2_shifted, i);
        //     //行偏移对应清空，防止出现跨越的情况
        //     if(i == 1)
        //         sc2_shifted_ringshifted.row(0) = sc2_shifted.row(0);
        //     else if(i == -1)
        //         sc2_shifted_ringshifted.row(sc2_shifted.rows() - 1) = sc2_shifted.row(sc2_shifted.rows() - 1);

        //     double cur_sc_dist_ringshifted = NDdistDirectSC( _sc1, sc2_shifted_ringshifted ); //计算相似度，计算各个列向量的余弦距离的和的平均值（去除行列式为0的部分）
        //     if( cur_sc_dist_ringshifted < cur_sc_dist )
        //     {
        //         cur_sc_dist = cur_sc_dist_ringshifted;
        //     }

        // }

        double cur_sc_dist = NDdistDirectSC( _sc1, sc2_shifted ); //计算相似度，计算各个列向量的余弦距离的和的平均值（去除行列式为0的部分）
        if( cur_sc_dist < min_sc_dist )
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext

const Eigen::MatrixXd& NDManager::NDgetConstRefRecentSCD(void)
{
    return inquiry_polarcontexts_.back();
}



//求解协方差 输入 点云集合  输出 均值-协方差对
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> NDManager::NDGetCovarMatrix(std::vector<Eigen::Vector3d> bin_piont)
{
	// reference: https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
	const int rows = bin_piont.size();
    const int cols = bin_piont[0].size();

    // cout << "rows: " << rows << " cols: " << cols << endl;
 
	std::vector<double> vec_;
	for (int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j){
		    vec_.insert(vec_.begin() + i * cols + j, bin_piont[i](j));     //将传入的数据按先后样本排列填入vector<float>
        }
	}   
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(vec_.data(), rows, cols);   //将vector<float>的数据映射至Map模板 可选参数 Eigen::RowMajor 按行存储
    
	// fprintf(stderr, "source matrix:\n");
	// std::cout << m << std::endl;
 
	// fprintf(stdout, "\nEigen implement:\n");
	const int nsamples = rows;
 
	Eigen::MatrixXd mean = m.colwise().mean();      //求样本均值
	// std::cout << "print mean: " << std::endl << mean << std::endl;
 
	Eigen::MatrixXd tmp(rows, cols);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			tmp(y, x) = m(y, x) - mean(0, x);
		}
	}
	//std::cout << "tmp: " << std::endl << tmp << std::endl;
 
	Eigen::MatrixXd covar = (tmp.adjoint() * tmp) / float(nsamples - 1);    //求协方差矩阵
	// std::cout << "print covariance matrix: " << std::endl << covar << std::endl << std::endl;
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> result = {mean,covar};
 
	return result;
}
 
//奇异值分解 输入 协方差  输出 奇异值
Eigen::MatrixXd NDManager::NDGetSingularvalue(Eigen::MatrixXd bin_cov)
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
 
	// fprintf(stderr, "singular values:\n");
	// print_matrix(singular_values.data(), singular_values.rows(), singular_values.cols());
	// fprintf(stderr, "left singular vectors:\n");
	// print_matrix(left_singular_vectors.data(), left_singular_vectors.rows(), left_singular_vectors.cols());
	// fprintf(stderr, "right singular vecotrs:\n");
	// print_matrix(right_singular_vectors.data(), right_singular_vectors.rows(), right_singular_vectors.cols());
 
	return singular_values;
}

//特征值分解 返回值为降序 输入 协方差  输出 特征值-特征向量对
std::pair<std::vector<double>,Eigen::MatrixXd> NDManager::NDGetEigenvalues(Eigen::MatrixXd bin_cov)
{
    // cout << "Here is a 3x3 matrix, bin_cov:" << endl << bin_cov << endl << endl;
    EigenSolver<Matrix3d> es(bin_cov);
	
	Matrix3d D = es.pseudoEigenvalueMatrix();
	Matrix3d V = es.pseudoEigenvectors();
	// cout << "The pseudo-eigenvalue matrix D is:" << endl << D << endl;
	// cout << "The pseudo-eigenvector matrix V is:" << endl << V << endl;
	// cout << "Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;

    //特征值 特征向量 排序
    for(int i = 0; i < D.cols(); i++)
    {
        for(int j = 0; j < D.cols()-1; j++)
        {
            if(D(j,j) < D(j+1,j+1))
            {
                double mid_value;
                Eigen::Vector3d mid_col;
                //D
                mid_value = D(j,j);
                D(j,j) = D(j+1,j+1);
                D(j+1,j+1) = mid_value;
                //V
                mid_col = V.col(j);
                V.col(j) = V.col(j+1);
                V.col(j+1) = mid_col;
            }
        }
    }
	// cout << "The pseudo-eigenvector matrix V is:" << endl << V << endl;
	// cout << "Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;

    //特征值赋值到vector
    std::vector<double> eigen_value;
    eigen_value.resize(D.cols());
    for(int i = 0; i < D.cols(); i++) { eigen_value[i] = D(i,i); }
    // std::cout << "eigen value: " << eigen_value[0] << " " << eigen_value[1] << " " << eigen_value[2] << endl;

    //将特征值和特征向量用pair形式返回
    std::pair<std::vector<double>,Eigen::MatrixXd> result = {eigen_value,V};

    return result;
}

//体素椭球筛选 形状筛选
//模型有效条件：点云数量大于20 
bool NDManager::NDFilterVoxelellipsoid(class Voxel_Ellipsoid &voxeleloid)
{
    if(voxeleloid.point_num > 10)
    {
        voxeleloid.valid = 1;
        double a = sqrt(voxeleloid.axis_length[0]);
        double b = sqrt(voxeleloid.axis_length[1]);
        double c = sqrt(voxeleloid.axis_length[2]);
        voxeleloid.mode = (((a-b) > (b-c)) && ((a-b) > c)) ? 1 : ((((b-c) > (a-b)) && ((b-c) > c)) ? 2 : 3);
    }else{
        voxeleloid.valid = 0;
    }

    return voxeleloid.valid;
}

//体素重合度计算 体素偏移一一对应版本 求出列偏移和环偏移后将体素偏移后一一对应
double NDManager::NDDistVoxeleloid(std::vector<class Voxel_Ellipsoid> &v_eloid_cur, std::vector<class Voxel_Ellipsoid> &v_eloid_can, int num_shift_col, int num_shift_row)
{
    double num_overlap_eloid = 0;
    double all_valid_eloid = 0;
    int can_id, cur_id;
    // cout << "shift num is:" << num_shift << endl;
    for(int i = 0; i < ND_PC_NUM_RING * ND_PC_NUM_SECTOR; i++)
    {
        //偏移候选体素椭球id
        can_id = (i % ND_PC_NUM_SECTOR + num_shift_col) % ND_PC_NUM_SECTOR + (i - i % ND_PC_NUM_SECTOR);
        cur_id = i;

        if(v_eloid_cur[cur_id].valid == 1 || v_eloid_can[can_id].valid == 1)
        {
            double dist = 0;
            double dist_mid = 0;
            double cos_col_ve = 0;
            double cos_col_ve_1 = 0;
            double cos_col_ve_2 = 0;
            if(v_eloid_cur[cur_id].valid == 1 && v_eloid_can[can_id].valid == 1 && v_eloid_cur[cur_id].mode == v_eloid_can[can_id].mode)
            {
                for(int ii = 0; ii < v_eloid_cur[cur_id].axis_length.size(); ii++)
                {
                    // cout << "cur id: " << cur_id << " cur length: " << v_eloid_cur[cur_id].axis_length[ii] 
                    //      << " can id: " << can_id << " can length: " << v_eloid_can[can_id].axis_length[ii] << endl;
                    dist += (v_eloid_cur[cur_id].axis_length[ii] - v_eloid_can[can_id].axis_length[ii]) * (v_eloid_cur[cur_id].axis_length[ii] - v_eloid_can[can_id].axis_length[ii]);
                }
                dist = sqrt(dist);
                // cout << "voxel ellipsoid cur id: " << cur_id << "  can id: " << can_id << "  distance: " << dist << endl;

                dist_mid = sqrt((v_eloid_cur[cur_id].center.x - v_eloid_can[can_id].center.x) * (v_eloid_cur[cur_id].center.x - v_eloid_can[can_id].center.x) +
                    (v_eloid_cur[cur_id].center.y - v_eloid_can[can_id].center.y) * (v_eloid_cur[cur_id].center.y - v_eloid_can[can_id].center.y) +
                    (v_eloid_cur[cur_id].center.z - v_eloid_can[can_id].center.z) * (v_eloid_cur[cur_id].center.z - v_eloid_can[can_id].center.z));

                for(int iii = 0; iii < v_eloid_cur[cur_id].axis.cols(); iii++){
                        VectorXd cur_col_axis =  v_eloid_cur[cur_id].axis.col(iii);
                        VectorXd can_col_axis =  v_eloid_can[can_id].axis.col(iii);

                        if(cur_col_axis.norm() == 0 || can_col_axis.norm() == 0)
                            continue; // don't count this sector pair.

                        cos_col_ve_1 = (1 - ((cur_col_axis.dot(can_col_axis)) / (cur_col_axis.norm() * can_col_axis.norm())));
                        can_col_axis *= (-1);
                        cos_col_ve_2 = (1 - ((cur_col_axis.dot(can_col_axis)) / (cur_col_axis.norm() * can_col_axis.norm())));
                        cos_col_ve += (cos_col_ve_1 > cos_col_ve_2 ? cos_col_ve_1 : cos_col_ve_2);
                }

                cos_col_ve /= 3;
                if(dist < ND_VOXEL_ELIOD_DIST_THRES && cos_col_ve > ND_VOXEL_ELIOD_COS_THRES)
                    num_overlap_eloid++;
            }

            all_valid_eloid++;
        }
    }
    // cout << "all valid voxel ellipsoid num: " << all_valid_eloid << "  overlap voxel ellipsoid num: " << num_overlap_eloid << endl;

    return (double)(1- (num_overlap_eloid / all_valid_eloid));
    
}

//体素重合度计算 椭球偏移落点版本 求出初始位姿后将椭球转移后判断落点于哪个体素再进行匹配
double NDManager::NDDistVoxeleloidPlace(std::vector<class Voxel_Ellipsoid> &v_eloid_cur, std::vector<class Voxel_Ellipsoid> &v_eloid_can, int num_shift_col, Eigen::Vector3d translat)
{
    double num_overlap_eloid = 0;
    double all_valid_eloid = 0;
    int can_id, cur_id;
    // cout << "col shift num is:" << num_shift_col << endl;

    //计算转移矩阵
    Eigen::Matrix4d transform;
    transform = GetTransformMatrixCombine(num_shift_col,translat);

    // 直接使用真值
    // transform = pose_ground_truth_copy[cur_frame_id] * pose_ground_truth_copy[can_frame_id].inverse();

    // cout << "transform: " << endl << transform << endl;

    for(int i = 0; i < ND_PC_NUM_RING * ND_PC_NUM_SECTOR; i++)
    {
        can_id = i;
        // cout << "can id: " << can_id << endl;
        if(v_eloid_can[can_id].valid == 1)
        {
            // // 不使用转移矩阵
            // cur_id = (i % ND_PC_NUM_SECTOR - num_shift_col + ND_PC_NUM_SECTOR) % ND_PC_NUM_SECTOR + (i - i % ND_PC_NUM_SECTOR);

            //寻找cur_id
            Eigen::Vector4d can_eloid_center;
            Eigen::Vector4d can_eloid_center_shift;
            //检索对应的查询点云体素id
            can_eloid_center[0] = v_eloid_can[can_id].center.x;
            can_eloid_center[1] = v_eloid_can[can_id].center.y;
            can_eloid_center[2] = v_eloid_can[can_id].center.z;
            can_eloid_center[3] = 1;
            can_eloid_center_shift = transform * can_eloid_center;
            // cout << "can id is: " << can_id << endl;
            // cout << "before shift can center is: " << endl << can_eloid_center << endl;
            // cout << "after shift can center is: " << endl << can_eloid_center_shift << endl;
 
            // xyz to ring, sector
            double azim_range = sqrt(can_eloid_center_shift[0] * can_eloid_center_shift[0] + can_eloid_center_shift[1] * can_eloid_center_shift[1]);   //求点到传感器中心的距离
            double azim_angle = xy2theta(can_eloid_center_shift[0], can_eloid_center_shift[1]);              //输出theta角        
            if( azim_range > ND_PC_MAX_RADIUS )     //判断点距离传感器距离是否超出最大值
            {            
                cur_id = -1;    //直接不予考虑
                continue;
            }            
            int ring_idx = std::max( std::min( ND_PC_NUM_RING, int(ceil( (azim_range / ND_PC_MAX_RADIUS) * ND_PC_NUM_RING )) ), 1 );    //从1开始
            int sector_idx = std::max( std::min( ND_PC_NUM_SECTOR, int(ceil( (azim_angle / 360.0) * ND_PC_NUM_SECTOR )) ), 1 );

            cur_id = (ring_idx - 1) * ND_PC_NUM_SECTOR + (sector_idx - 1);
            // cout << "cur id is: " << cur_id << endl;


            double dist = 0;
            double dist_mid = 0;
            double cos_col_ve = 0;

            // if(cur_id != -1 && v_eloid_cur[cur_id].valid == 1 && v_eloid_cur[cur_id].mode == v_eloid_can[can_id].mode)  //体素内不存在有效模型，直接丢弃
            if(cur_id != -1 && v_eloid_cur[cur_id].valid == 1)  //体素内不存在有效模型，直接丢弃
            {
                //重新调整候选椭球模型
                Voxel_Ellipsoid can_eloid_shift;
                can_eloid_shift.center.x = can_eloid_center_shift[0];
                can_eloid_shift.center.y = can_eloid_center_shift[1];
                can_eloid_shift.center.z = can_eloid_center_shift[2];

                //利用转移矩阵转移候选点云的椭球轴
                Eigen::MatrixXd _can_eloid_axis;
                _can_eloid_axis.resize(4,3);
                _can_eloid_axis.block<3,3>(0,0) = v_eloid_can[can_id].axis;
                _can_eloid_axis.row(3) = Eigen::Vector3d::Ones();
                _can_eloid_axis = transform * _can_eloid_axis;
                can_eloid_shift.axis = _can_eloid_axis.block<3,3>(0,0);

                for(int ii = 0; ii < v_eloid_cur[cur_id].axis_length.size(); ii++)
                {
                    // cout << "cur id: " << cur_id << " cur length: " << v_eloid_cur[cur_id].axis_length[ii] 
                    //      << " can id: " << can_id << " can length: " << v_eloid_can[can_id].axis_length[ii] << endl;
                    // dist += (v_eloid_cur[cur_id].axis_length[ii] - v_eloid_can[can_id].axis_length[ii]) * (v_eloid_cur[cur_id].axis_length[ii] - v_eloid_can[can_id].axis_length[ii]);
                }
                // dist = sqrt(dist);
                // cout << "voxel ellipsoid cur id: " << cur_id << "  can id: " << can_id << " length distance: " << dist << endl;

                dist_mid = sqrt((v_eloid_cur[cur_id].center.x - can_eloid_shift.center.x) * (v_eloid_cur[cur_id].center.x - can_eloid_shift.center.x) +
                    (v_eloid_cur[cur_id].center.y - can_eloid_shift.center.y) * (v_eloid_cur[cur_id].center.y - can_eloid_shift.center.y) +
                    (v_eloid_cur[cur_id].center.z - can_eloid_shift.center.z) * (v_eloid_cur[cur_id].center.z - can_eloid_shift.center.z));

                // cout << "voxel ellipsoid cur id: " << cur_id << "  can id: " << can_id << " center distance: " << dist_mid << endl;
                

                for(int iii = 0; iii < v_eloid_cur[cur_id].axis.cols(); iii++){
                        VectorXd cur_col_axis =  v_eloid_cur[cur_id].axis.col(iii);
                        VectorXd can_col_axis =  can_eloid_shift.axis.col(iii);

                        if(cur_col_axis.norm() == 0 || can_col_axis.norm() == 0)
                            continue; // don't count this sector pair.

                        cos_col_ve += abs(((cur_col_axis.dot(can_col_axis)) / (cur_col_axis.norm() * can_col_axis.norm())));
                }
                cos_col_ve /= 3;
                // cout << "voxel ellipsoid cur id: " << cur_id << "  can id: " << can_id << "  cosine: " << cos_col_ve << endl;


                if(dist_mid < ND_VOXEL_ELIOD_DIST_THRES && cos_col_ve > ND_VOXEL_ELIOD_COS_THRES)            //中心距离 + cosine
                // if(dist < ND_VOXEL_ELIOD_DIST_THRES && dist_mid > ND_VOXEL_ELIOD_DIST_THRES)             //中心距离 + 长度差
                // if(dist_mid < ND_VOXEL_ELIOD_DIST_THRES)                                                    //中心距离
                // if(dist < ND_VOXEL_ELIOD_DIST_THRES)                                                       //长度差
                    num_overlap_eloid++;
            }

            all_valid_eloid++;
        }
    }
    // cout << "all valid voxel ellipsoid num: " << all_valid_eloid << "  overlap voxel ellipsoid num: " << num_overlap_eloid << endl;

    if(all_valid_eloid == 0)
        return 1;
    return (double)(1- (num_overlap_eloid / all_valid_eloid));
    
}


//体素椭球列合并
class Voxel_Ellipsoid NDManager::NDMergeColVoxeleloid(std::vector<class Voxel_Ellipsoid> &v_eloid, int col)
{  
    class Voxel_Ellipsoid merge_eloid;
    int Nm = 0;
    Eigen::Matrix3d pm,Em;
    pm << 0,0,0,
          0,0,0,
          0,0,0;
    Em = pm;

    for(int i = 0; i < ND_PC_NUM_RING; i++)
    {
        class Voxel_Ellipsoid added_eloid;
        Eigen::Matrix3d added_mean;
        added_mean << 0,0,0,
                      0,0,0,
                      0,0,0;
        added_eloid = v_eloid[i * ND_PC_NUM_SECTOR + col];   

        if(added_eloid.valid == 1)
        {
            //数量
            Nm += added_eloid.point_num;
            //均值 
            added_mean(0,0) = added_eloid.center.x;
            added_mean(1,0) = added_eloid.center.y;
            added_mean(2,0) = added_eloid.center.z;
            pm += added_eloid.point_num * added_mean;
            //协方差
            Em += added_eloid.point_num * (added_eloid.cov + added_mean * added_mean.transpose());

            // cout << "Nm: " << Nm << endl
            //      << "pm: " << endl
            //      << pm << endl
            //      << "Em: " << endl
            //      << Em << endl;
        }
    }

    //均值协方差计算
    pm = pm / Nm;
    Em = (Em / Nm) - (pm * pm.transpose());

    //最后进行均值转换 椭球剩余值填充
    std::pair<std::vector<double>,Eigen::MatrixXd> merge_eigen_;
    merge_eloid.point_num = Nm; 
    merge_eloid.center.x = pm(0,0);
    merge_eloid.center.y = pm(1,0);
    merge_eloid.center.z = pm(2,0);
    merge_eloid.valid = 1;
    merge_eloid.cov = Em;
    merge_eigen_ = NDGetEigenvalues(merge_eloid.cov);
    merge_eloid.axis_length = merge_eigen_.first;
    merge_eloid.axis = merge_eigen_.second;

    // 打印测试
    // cout << "TEST : ";
    // cout << "eloid num: " << merge_eloid.point_num << endl
        //  << "center: " << merge_eloid.center.x << ',' <<  merge_eloid.center.y << ',' << merge_eloid.center.z << endl
        //  << "cov: " << endl
        //  << merge_eloid.cov << endl
        //  << "axis length: " << merge_eloid.axis_length[0] << ',' << merge_eloid.axis_length[1] << ',' << merge_eloid.axis_length[2] << endl
        //  << "axis: " << endl 
        //  << merge_eloid.axis << endl
        //  << endl;

    return merge_eloid;
}

//体素椭球合并
class Voxel_Ellipsoid NDManager::NDMergeVoxeleloid(std::vector<class Voxel_Ellipsoid> &v_eloid, std::vector<int> v_id)
{  
    class Voxel_Ellipsoid merge_eloid;
    int Nm = 0;
    Eigen::Matrix3d pm,Em;
    pm << 0,0,0,
          0,0,0,
          0,0,0;
    Em = pm;

    for(int i = 0; i < v_id.size(); i++)
    {
        class Voxel_Ellipsoid added_eloid;
        Eigen::Matrix3d added_mean;
        added_mean << 0,0,0,
                      0,0,0,
                      0,0,0;
        added_eloid = v_eloid[v_id[i]];   

        if(added_eloid.valid == 1)
        {
            //数量
            Nm += added_eloid.point_num;
            //均值 
            added_mean(0,0) = added_eloid.center.x;
            added_mean(1,0) = added_eloid.center.y;
            added_mean(2,0) = added_eloid.center.z;
            pm += added_eloid.point_num * added_mean;
            //协方差
            Em += added_eloid.point_num * (added_eloid.cov + added_mean * added_mean.transpose());

            // cout << "Nm: " << Nm << endl
            //      << "pm: " << endl
            //      << pm << endl
            //      << "Em: " << endl
            //      << Em << endl;
        }
    }

    //均值协方差计算
    pm = pm / Nm;
    Em = (Em / Nm) - (pm * pm.transpose());

    //最后进行均值转换 椭球剩余值填充
    std::pair<std::vector<double>,Eigen::MatrixXd> merge_eigen_;
    merge_eloid.point_num = Nm; 
    merge_eloid.center.x = pm(0,0);
    merge_eloid.center.y = pm(1,0);
    merge_eloid.center.z = pm(2,0);
    merge_eloid.valid = 1;
    merge_eloid.cov = Em;
    merge_eigen_ = NDGetEigenvalues(merge_eloid.cov);
    merge_eloid.axis_length = merge_eigen_.first;
    merge_eloid.axis = merge_eigen_.second;

    // 打印测试
    // cout << "TEST : ";
    cout << "eloid num: " << merge_eloid.point_num << endl
         << "center: " << merge_eloid.center.x << ',' <<  merge_eloid.center.y << ',' << merge_eloid.center.z << endl
        //  << "cov: " << endl
        //  << merge_eloid.cov << endl
         << "axis length: " << merge_eloid.axis_length[0] << ',' << merge_eloid.axis_length[1] << ',' << merge_eloid.axis_length[2] << endl
        //  << "axis: " << endl 
        //  << merge_eloid.axis <<endl
        ;

    return merge_eloid;
}

/*想法测试
    提取点云体素内投影面积远小于体素面积的椭球模型，作为关键椭球匹配，获取查询点云和候选点云的平移矩阵*/

//获取最接近z轴的方向 返回除最接近z轴的方向对应的特征值以外的两个特征值（降序）
std::pair<double,double> GetNearestZaxis(class Voxel_Ellipsoid v_eloid)
{
    double cos_ve = 0;
    double cos_ve_1 = 0;
    double cos_ve_2 = 0;
    double cos_ve_max = -1;
    int max_axis = 0;
    for(int iii = 0; iii < v_eloid.axis.cols(); iii++){
        VectorXd cur_axis =  v_eloid.axis.col(iii);
        VectorXd can_axis =  v_eloid.axis.col(iii);
        if(cur_axis.norm() == 0 || can_axis.norm() == 0)
            continue; // don't count this sector pair.

        cos_ve_1 = ((cur_axis.dot(can_axis)) / (cur_axis.norm() * can_axis.norm()));
        can_axis *= (-1);
        cos_ve_2 = ((cur_axis.dot(can_axis)) / (cur_axis.norm() * can_axis.norm()));
        cos_ve = (cos_ve_1 > cos_ve_2 ? cos_ve_1 : cos_ve_2);
        if(cos_ve > cos_ve_max)
        {
            cos_ve_max = cos_ve;
            max_axis = iii; 
        }
    }
    std::pair<double,double> result;
    if(max_axis ==  0)
    {
        result.first = v_eloid.axis_length[1];
        result.second = v_eloid.axis_length[2];
    }else if(max_axis ==  1)
    {
        result.first = v_eloid.axis_length[0];
        result.second = v_eloid.axis_length[2];        
    }else{
        result.first = v_eloid.axis_length[0];
        result.second = v_eloid.axis_length[1];       
    }

    return result;
}

//关键椭球判断函数
int IsKeyVoxelEllipsoid(int key_id, std::vector<class Voxel_Ellipsoid> &v_eloid)
{
    if(v_eloid[key_id].point_num > 100)
    {
        // if(v_eloid[key_id].max_h_center.z > v_eloid[key_id+1].max_h_center.z * 1.5
        //     && v_eloid[key_id].max_h_center.z > v_eloid[(key_id-1+60)%60].max_h_center.z * 1.5
        //     && v_eloid[key_id].max_h_center.z > v_eloid[key_id-60].max_h_center.z * 1.5
        //     && v_eloid[key_id].max_h_center.z > v_eloid[key_id+60].max_h_center.z * 1.5)
        if(v_eloid[key_id].max_h_center.z > v_eloid[key_id-60].max_h_center.z * 1.5
            && v_eloid[key_id].max_h_center.z > v_eloid[key_id+60].max_h_center.z * 1.5)        
        {
            return 1;
            // cout << "333" << endl;
        }
    }

    return 0;
}

//提取关键椭球
std::vector<class Voxel_Ellipsoid> NDManager::GetKeyVoxelEllipsoid(std::vector<class Voxel_Ellipsoid> &v_eloid)
{
    std::vector<class Voxel_Ellipsoid> key_eloid;
    double pi = 3.14;
    for(int i = 60; i < (ND_PC_NUM_RING - 10) * ND_PC_NUM_SECTOR; i++)
    {
        //体素面积
        double voxel_area = (pi * (ceil((double)i / ND_PC_NUM_SECTOR) * ND_PC_UNIT_RINGGAP * ceil((double)i / ND_PC_NUM_SECTOR) * ND_PC_UNIT_RINGGAP
                             - floor((double)i / ND_PC_NUM_SECTOR) * ND_PC_UNIT_RINGGAP * floor((double)i / ND_PC_NUM_SECTOR) * ND_PC_UNIT_RINGGAP)) / ND_PC_NUM_SECTOR;

        std::pair<double,double> far_z_eigen = GetNearestZaxis(v_eloid[i]);

        double eloid_area = pi * far_z_eigen.first * far_z_eigen.second;

        // cout << "voxel area: "  << voxel_area << "  eloid area: "  << eloid_area << endl;
        // if(eloid_area < (voxel_area * 0.1) && v_eloid[i].point_num > 100)
        if(IsKeyVoxelEllipsoid(i,v_eloid))
        {
            cout << "voxel id: " << i << endl;
            cout << "eloid num: " << v_eloid[i].point_num << endl
             << "max center: " << v_eloid[i].max_h_center.x << ',' <<  v_eloid[i].max_h_center.y << ',' << v_eloid[i].max_h_center.z << endl
            //  << "cov: " << endl
            //  << v_eloid[i].cov << endl
            //  << "axis length: " << v_eloid[i].axis_length[0] << ',' << v_eloid[i].axis_length[1] << ',' << v_eloid[i].axis_length[2] << endl
            //  << "axis: " << endl 
            //  << v_eloid[i].axis <<endl
            ;

            // cout << "voxel area: "  << voxel_area << endl;
            // cout << "eloid area: "  << eloid_area << endl;
            
            key_eloid.push_back(v_eloid[i]);
        }

    }
    cout << "key eloid num: "  << key_eloid.size() << endl;
    return key_eloid;
}

//匹配关键椭球
Eigen::Vector3d NDManager::MatchKeyVoxelEllipsoid(std::vector<class Voxel_Ellipsoid> &v_eloid_cur, std::vector<class Voxel_Ellipsoid> &v_eloid_can)
{
    cout << "CUR KEY ELOID" << endl;
    std::vector<class Voxel_Ellipsoid> cur_key_eloid = GetKeyVoxelEllipsoid(v_eloid_cur);
    cout << "CAN KEY ELOID" << endl;
    std::vector<class Voxel_Ellipsoid> can_key_eloid = GetKeyVoxelEllipsoid(v_eloid_can);
    std::vector<int> min_can_key_id;

    //匹配椭球模型
    Eigen::Vector3d avg_center;
    avg_center.setZero();
    for(int i = 0; i < cur_key_eloid.size(); i++)
    {
        double min_dist_center = 10000;
        int min_index = -1;
        for(int j = 0; j < can_key_eloid.size(); j++)
        {
            
            double dist_center = sqrt((cur_key_eloid[i].max_h_center.x - can_key_eloid[j].max_h_center.x) * (cur_key_eloid[i].max_h_center.x - can_key_eloid[j].max_h_center.x) +
                                (cur_key_eloid[i].max_h_center.y - can_key_eloid[j].max_h_center.y) * (cur_key_eloid[i].max_h_center.y - can_key_eloid[j].max_h_center.y) +
                                (cur_key_eloid[i].max_h_center.z - can_key_eloid[j].max_h_center.z) * (cur_key_eloid[i].max_h_center.z - can_key_eloid[j].max_h_center.z)); 
            if(dist_center < min_dist_center)
            {
                min_dist_center = dist_center;
                min_index = j;
            }
        }
        if(min_dist_center > 10)
            min_index = -1;

        cout << "min index: " << min_index << endl;
        min_can_key_id.push_back(min_index);
    }

    cout << "min can key id num: " << min_can_key_id.size() << endl;
    
    double avg_cnt = 0;
    //计算平均平移矩阵
    for(int i = 0; i < cur_key_eloid.size(); i++)
    {
        if(min_can_key_id[i] == -1)
            continue;
        
        avg_center[0] += cur_key_eloid[i].max_h_center.x - can_key_eloid[min_can_key_id[i]].max_h_center.x; 
        avg_center[1] += cur_key_eloid[i].max_h_center.y - can_key_eloid[min_can_key_id[i]].max_h_center.y; 
        avg_center[2] += cur_key_eloid[i].max_h_center.z - can_key_eloid[min_can_key_id[i]].max_h_center.z; 
        avg_cnt ++;

    }
    avg_center /= avg_cnt;

    return avg_center;
}

//列体素椭球筛选合并 相似度计算
Eigen::Vector3d NDManager::NDDistMergeVoxelellipsoid(std::vector<class Voxel_Ellipsoid> &v_eloid_cur, std::vector<class Voxel_Ellipsoid> &v_eloid_can, int num_shift)
{
    double num_overlap_eloid = 0;
    double all_valid_eloid = 0;
    int can_col, cur_col;
    // cout << "shift num is:" << num_shift << endl;
    Eigen::Vector3d avg_center = {0,0,0};
    SCPointType avg_center_1;
    int valid_center_num = 0;
    for(int i = 0; i < ND_PC_NUM_SECTOR; i++)
    {
        class Voxel_Ellipsoid cur_col_eloid,can_col_eloid;

        //偏移候选体素椭球列
        cur_col = i;
        can_col = (i + num_shift) % ND_PC_NUM_SECTOR;

        cout << "TEST cur cloud" << "  col: " << cur_col << endl;
        cur_col_eloid = NDMergeColVoxeleloid(v_eloid_cur,cur_col);
        cout << "TEST can cloud" << "  col: " << can_col << endl;
        can_col_eloid = NDMergeColVoxeleloid(v_eloid_can,can_col);

        //计算体素椭球三轴和中心相似度
        double dist_length = 0;
        double dist_mid = 0;
        double cos_col_ve = 0;
        double cos_col_ve_1 = 0;
        double cos_col_ve_2 = 0;
        if(cur_col_eloid.valid == 1 && can_col_eloid.valid == 1)    //判断是否为可靠椭球
        {
            for(int ii = 0; ii < cur_col_eloid.axis_length.size(); ii++)
            {
                // cout << "cur id: " << cur_col << " cur length: " << cur_col_eloid.axis_length[ii] 
                //      << " can id: " << can_col << " can length: " << can_col_eloid.axis_length[ii] << endl;
                dist_length += (cur_col_eloid.axis_length[ii] - can_col_eloid.axis_length[ii]) * (cur_col_eloid.axis_length[ii] - can_col_eloid.axis_length[ii]);
            }
            dist_length = sqrt(dist_length);

            dist_mid = sqrt((cur_col_eloid.center.x - can_col_eloid.center.x) * (cur_col_eloid.center.x - can_col_eloid.center.x) +
                (cur_col_eloid.center.y - can_col_eloid.center.y) * (cur_col_eloid.center.y - can_col_eloid.center.y) +
                (cur_col_eloid.center.z - can_col_eloid.center.z) * (cur_col_eloid.center.z - can_col_eloid.center.z));
            
            for(int iii = 0; iii < cur_col_eloid.axis.cols(); iii++){
                VectorXd cur_col_axis =  cur_col_eloid.axis.col(iii);
                VectorXd can_col_axis =  can_col_eloid.axis.col(iii);

                if(cur_col_axis.norm() == 0 || can_col_axis.norm() == 0)
                    continue; // don't count this sector pair.

                cos_col_ve_1 = (1 - ((cur_col_axis.dot(can_col_axis)) / (cur_col_axis.norm() * can_col_axis.norm())));
                can_col_axis *= (-1);
                cos_col_ve_2 = (1 - ((cur_col_axis.dot(can_col_axis)) / (cur_col_axis.norm() * can_col_axis.norm())));
                cos_col_ve += (cos_col_ve_1 < cos_col_ve_2 ? cos_col_ve_1 : cos_col_ve_2);
                
            }
            // cout << "voxel ellipsoid cur col: " << cur_col << "  can col: " << can_col << "  length distance: " << dist_length << endl;
            cout << " TEST mid distance: " << dist_mid << "  length distance: " << dist_length << " cos col: " << cos_col_ve << endl;

            // if(dist_mid < 10 && cur_col_eloid.mode == can_col_eloid.mode)
            if(dist_length < ND_VOXEL_ELIOD_DIST_THRES && dist_mid < 1.5)
            {
                avg_center[0] +=  cur_col_eloid.center.x - can_col_eloid.center.x;
                avg_center[1] +=  cur_col_eloid.center.y - can_col_eloid.center.y;
                avg_center[2] +=  cur_col_eloid.center.z - can_col_eloid.center.z;
                valid_center_num ++;

                cout <<  "  center:" << endl;
                cout << cur_col_eloid.center.x - can_col_eloid.center.x << " " 
                     << cur_col_eloid.center.y - can_col_eloid.center.y << " " 
                     << cur_col_eloid.center.z - can_col_eloid.center.z << " " 
                     << endl << endl;
            }



            if(dist_length < ND_VOXEL_ELIOD_DIST_THRES && dist_mid < 1.5)
                num_overlap_eloid++;
        }

        all_valid_eloid++;
    }

    //平均平移向量处理
    // if(valid_center_num < 6) 
    // {
    //     cout << "valid center num is so less !!!" << endl;
    //     avg_center.Zero();
    //     return avg_center;
    // }else{
        avg_center = avg_center / valid_center_num;
    // }

    cout << endl << "num: " << valid_center_num << "  avg center:" << endl;
    cout << avg_center << endl;


    cout << " TEST all valid voxel ellipsoid num: " << all_valid_eloid << "  overlap voxel ellipsoid num: " << num_overlap_eloid << endl;

    // cout << " TEST avg mid distance: " << avg_dist_mid << " avg length distance: " << avg_dist_length << " avg cos col: " << avg_cos_col << endl;


    return avg_center;    

}

//初始平移矩阵获取
Eigen::Vector3d NDManager::NDGetTranslationMatrix(std::vector<class Voxel_Ellipsoid> &v_eloid_cur,std::vector<class Voxel_Ellipsoid> &v_eloid_can, int num_shift)
{
    std::vector<Eigen::Vector3d> cur_z_arry;
    std::vector<Eigen::Vector3d> can_z_arry;
    int col_vaild_cnt = 0;
    //最大均值高度队列获取
    // cout << "endter get translation funsion" << endl;
    for(int col = 0; col < ND_PC_NUM_SECTOR; col++)
    {
        double cur_max_avg_z = -100;
        double can_max_avg_z = -100;
        int cur_max_row = -1;
        int can_max_row = -1;
        int cur_vaild_num = 0;
        int can_vaild_num = 0;

        Eigen::Vector3d cur_max_center = {0,0,0};
        Eigen::Vector3d can_max_center = {0,0,0};

        int cur_id = col;
        int can_id = (col + num_shift) % ND_PC_NUM_SECTOR;
        // cout << "get can id " << can_id << endl;
        for(int row = 0; row < ND_PC_NUM_RING; row ++)
        {
            //体素可靠计数
            if(v_eloid_cur[cur_id + row * ND_PC_NUM_SECTOR].valid == 1)
                cur_vaild_num++;
            if(v_eloid_can[can_id + row * ND_PC_NUM_SECTOR].valid == 1)
                can_vaild_num++;
            //最大均值高度判断
            if(v_eloid_cur[cur_id + row * ND_PC_NUM_SECTOR].center.z > cur_max_avg_z)
            {   
                cur_max_avg_z =  v_eloid_cur[cur_id + row * ND_PC_NUM_SECTOR].center.z;
                cur_max_row = row;
            }
                
            if(v_eloid_can[can_id + row * ND_PC_NUM_SECTOR].center.z > can_max_avg_z)
            {
                can_max_avg_z =  v_eloid_can[can_id + row * ND_PC_NUM_SECTOR].center.z;
                can_max_row = row;
            }
        }

        if((abs(cur_vaild_num - can_vaild_num) > 5) || (abs(can_max_avg_z - cur_max_avg_z) > 0.5))
        {
            cur_max_center[0] = 0;
            cur_max_center[1] = 0;
            cur_max_center[2] = 0;

            can_max_center[0] = 0;
            can_max_center[1] = 0;
            can_max_center[2] = 0;
            // cout << "col: " << col << " is unvaild" << endl;
        }else{
            cur_max_center[0] = v_eloid_cur[cur_id + cur_max_row * ND_PC_NUM_SECTOR].center.x;
            cur_max_center[1] = v_eloid_cur[cur_id + cur_max_row * ND_PC_NUM_SECTOR].center.y;
            cur_max_center[2] = v_eloid_cur[cur_id + cur_max_row * ND_PC_NUM_SECTOR].center.z;

            can_max_center[0] = v_eloid_can[can_id + can_max_row * ND_PC_NUM_SECTOR].center.x;
            can_max_center[1] = v_eloid_can[can_id + can_max_row * ND_PC_NUM_SECTOR].center.y;
            can_max_center[2] = v_eloid_can[can_id + can_max_row * ND_PC_NUM_SECTOR].center.z;

            col_vaild_cnt++;
        }

        cur_z_arry.push_back(cur_max_center);
        can_z_arry.push_back(can_max_center);
    }

    Eigen::Vector3d cur_avg_translation = {0,0,0};
    Eigen::Vector3d can_avg_translation = {0,0,0};
    Eigen::Vector3d avg_translation = {0,0,0};
    // cout << "finish getting feature voxel" << endl;
    for(int col = 0; col < ND_PC_NUM_SECTOR; col++)
    {
        Eigen::Vector3d col_translation = {0,0,0};

        cur_avg_translation += cur_z_arry[col];
        can_avg_translation += can_z_arry[col];
    }
    // cout << "col vaild cnt: " << col_vaild_cnt << endl;

    cur_avg_translation /= (double)col_vaild_cnt;
    can_avg_translation /= (double)col_vaild_cnt;
    avg_translation = can_avg_translation - cur_avg_translation;

    // std::cout << "avg_translation: " << endl << avg_translation << endl;

    return avg_translation;
}

//平移矩阵转换成环偏移 输入偏移 输出 环键偏移 -1为偏移无效
int NDManager::NDGetringshift(Eigen::Vector3d translation)
{
    int shift_row = -1;
    double dist = sqrt((translation[0]) * (translation[0]) + 
                        (translation[1]) * (translation[1]) +
                        (translation[2]) * (translation[2]));
    if(dist > 40)
    {
        return shift_row;
    }
    else
    {
        int integer = (int)(dist / ND_PC_UNIT_RINGGAP);
        double remainder = dist - integer;

        if(remainder > (ND_PC_UNIT_RINGGAP / 2))
            integer++;
        shift_row = integer;
    }

    return shift_row;
}

//转移矩阵组装
Eigen::Matrix4d NDManager::GetTransformMatrixCombine(int col_num_shift,Eigen::Vector3d translation)
{
    //实际上是是将绕z轴旋转和平移组装到一起
    double yaw = deg2rad((float)(360 - col_num_shift * ND_PC_UNIT_SECTORANGLE));     //这里的转移矩阵旋转角与列偏移角的和是360度

    // cout << "yaw: " << yaw << endl;

    Eigen::Matrix4d transform;
    transform = Eigen::Matrix4d::Zero();
    transform(0,0) = cos(yaw);
    transform(0,1) = -sin(yaw);
    transform(1,0) = sin(yaw);
    transform(1,1) = cos(yaw);
    transform(2,2) = 1;
    transform(0,3) = -translation[0];
    transform(1,3) = -translation[1];
    transform(2,3) = -translation[2];       //点云平移与坐标系平移方向相反
    transform(3,3) = 1;
    
    return transform;
}

//体素椭球版本 描述符偏移量计算 重合度计算
std::pair<double, int> NDManager::NDdistancevoxeleloid( MatrixXd &_sc1, MatrixXd &_sc2, std::vector<class Voxel_Ellipsoid> &v_eloid_cur,std::vector<class Voxel_Ellipsoid> &v_eloid_can)
{
    // 1. fast align using variant key (not in original IROS18)
    MatrixXd vkey_sc1 = NDmakeSectorkeyFromScancontext( _sc1 );   //计算列的均值并以行的形式返回整个矩阵的列向量均值
    MatrixXd vkey_sc2 = NDmakeSectorkeyFromScancontext( _sc2 );
    int argmin_vkey_shift = NDfastAlignUsingVkey( vkey_sc1, vkey_sc2 );   //将列均值向右移动并使用F-范数进行对比，返回偏移值
    // cout << "TEST real col shift: " << argmin_vkey_shift << endl;

    const int SEARCH_RADIUS = round( 0.5 * ND_SEARCH_RATIO * _sc1.cols() ); // a half of search range  //搜索范围
    std::vector<int> shift_idx_search_space { argmin_vkey_shift };
    for ( int ii = 1; ii < SEARCH_RADIUS + 1; ii++ )
    {
        shift_idx_search_space.push_back( (argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols() );
        shift_idx_search_space.push_back( (argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols() );
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());    //在vector中填入以获取到的偏移量为中心，前后范围为搜索范围的待匹配偏移量

    // 2. fast columnwise diff 
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for ( int num_shift_col: shift_idx_search_space )
    {
        // cout << "TEST current col shift: " << num_shift << endl << endl;
        // MatrixXd sc2_shifted = circshift(_sc2, num_shift_col);  //列位移函数
        // double cur_sc_dist_ca = NDdistDirectSC( _sc1, sc2_shifted ); //计算相似度，计算各个列向量的余弦距离的和的平均值（去除行列式为0的部分）

        // //椭球重叠率（偏移一一对齐） + 转移矩阵
        // Eigen::Vector3d translation_shift = NDGetTranslationMatrix(v_eloid_cur, v_eloid_can, num_shift_col);
        // int num_shift_row = NDGetringshift(translation_shift);
        // double cur_sc_dist = NDDistVoxeleloid( v_eloid_cur, v_eloid_can, num_shift_col,num_shift_row); //计算椭球重叠率

        //加入转移矩阵
        Eigen::Vector3d translation_shift = NDGetTranslationMatrix(v_eloid_cur, v_eloid_can, num_shift_col);
        double cur_sc_dist = NDDistVoxeleloidPlace(v_eloid_cur, v_eloid_can, num_shift_col, translation_shift);
        
        // double cur_sc_dist_single = NDDistVoxeleloid(v_eloid_cur, v_eloid_can, num_shift);
        // double cur_sc_dist = cur_sc_dist_col * 0.5 + cur_sc_dist_ca * 0.5;
        if( cur_sc_dist < min_sc_dist )
        {
            argmin_shift = num_shift_col;
            min_sc_dist = cur_sc_dist;
        }
        // cout << "TEST min distance: " << min_sc_dist << endl;
        // cout << "TEST min col shift: " << argmin_shift << endl;
    }

    return make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext

/*---------------------------------------------------------------------------SVD分解获取转移矩阵---------------------------------------------------------------------------*/
//svd分解 转移矩阵获取
Eigen::Matrix4d NDManager::NDGetTransformMatrixwithSVD(std::vector<Eigen::Matrix3Xd> inquiry_feature_p, std::vector<Eigen::Matrix3Xd> match_feature_p, int align_num)
{
    TicToc get_transform;
    get_transform.tic();
    //svd分解转移矩阵

    //获取偏移点集
    std::vector<Eigen::Matrix3Xd> aligned_inquiry_feature_p = NDAlignFeaturePoint(inquiry_feature_p, align_num);     //并未改变点的坐标，也无需后续进行还原旋转

    //获取转移矩阵
    // return GetTransformMatrix(match_feature_p,aligned_inquiry_feature_p);
    Eigen::Matrix4d result = GetTransformMatrixwithCERE(aligned_inquiry_feature_p, match_feature_p, (double)(align_num * ND_PC_UNIT_SECTORANGLE));

    step_timecost[4] += get_transform.toc();
    // printf("[Get transform] Time cost: %7.5fs\r\n", get_transform.toc());

    return result;

}

//获取特征点集 输入单帧点云的各个体素模型 输出点云的特征点集
std::vector<Eigen::Matrix3Xd> NDManager::NDGetFeaturePoint(std::vector<class Voxel_Ellipsoid> frame_eloid)
{
    std::vector<Eigen::Vector3d> feature_point_max(ND_PC_NUM_SECTOR, {0,0,-10});
    std::vector<Eigen::Vector3d> feature_point_min(ND_PC_NUM_SECTOR, {0,0,10});
    // std::vector<Eigen::Vector3d> feature_point_max_z(ND_PC_NUM_SECTOR, {0,0,-10});
    for(int i = 0; i < frame_eloid.size(); i++)
    {
        //过滤无用体素模型
        if(frame_eloid[i].num <= 50)
            continue;
        int sector_index = i % ND_PC_NUM_SECTOR;
        Eigen::Vector3d center = {frame_eloid[i].center.x, frame_eloid[i].center.y, frame_eloid[i].center.z};                       //提取最高中心点作为匹配点集
        // Eigen::Vector3d max_z_pt_ = {frame_eloid[i].max_h_center.x, frame_eloid[i].max_h_center.y, frame_eloid[i].max_h_center.z};  //提取最高点作为匹配点集

        if(center[2] > feature_point_max[sector_index][2])
        {
            feature_point_max[sector_index] = center;
        }

        if(center[2] < feature_point_min[sector_index][2])
        {
            feature_point_min[sector_index] = center;
        }
        // if(max_z_pt_[2] > feature_point_max_z[sector_index][2])
        // {
        //     feature_point_max_z[sector_index] = max_z_pt_;
        // }


    }

    //去除未被赋值的特征点集
    for(auto feature_p : feature_point_max)
    {
        if(feature_p[2] == -10)
        {
            feature_p = {0,0,0};
        }
    }
    for(auto feature_p : feature_point_min)
    {
        if(feature_p[2] == 10)
        {
            feature_p = {0,0,0};
        }
    }
    // for(auto feature_p : feature_point_max_z)
    // {
    //     if(feature_p[2] == 10)
    //     {
    //         feature_p = {0,0,0};
    //     }
    // }

    //打印特征点集
    // cout << "NDGetFeaturePoint   make feature point: " << endl;
    // for(auto feature_p : feature_point_max)
    // {
    //     cout << feature_p.transpose() << ","; 
    // }cout << endl;

    ////返回最高点与最低点的相对向量 
    std::vector<Eigen::Matrix3Xd> feature_point(ND_PC_NUM_SECTOR);
    for(int i = 0; i < ND_PC_NUM_SECTOR; i++)
    {
        // feature_point[i] = feature_point_max[i] - feature_point_min[i];
        feature_point[i].resize(3,2);
        feature_point[i].col(0) = feature_point_max[i];
        feature_point[i].col(1) = feature_point_min[i];
        // feature_point[i] = feature_point_max_z[i];
    }

    return feature_point;
}

//偏移对齐特征点集
std::vector<Eigen::Matrix3Xd> NDManager::NDAlignFeaturePoint(std::vector<Eigen::Matrix3Xd> feature_point, int align_scetor)
{
    // std::cout << "[ND] align feature point set" << std::endl;

    std::vector<Eigen::Matrix3Xd> aligned_feature_p(ND_PC_NUM_SECTOR);
    for(int i = 0; i < feature_point.size(); i++)
    {
        // aligned_feature_p[(i+align_scetor) % ND_PC_NUM_SECTOR] = feature_point[i];
        aligned_feature_p[(i+(ND_PC_NUM_SECTOR - align_scetor)) % ND_PC_NUM_SECTOR] = feature_point[i];
    }

    return aligned_feature_p;
}

/*---------------------------------------------------------------------------功能函数---------------------------------------------------------------------------*/



//体素椭球数据保存函数
void NDManager::NDSaveVoxelellipsoidData(std::vector<class Voxel_Ellipsoid> v_eloid_data, int id)
{
    static int init = 0;
    std::ostringstream out;
    out << std::internal << std::setfill('0') << std::setw(6) << id;
    string v_eloid_id = out.str();
    string voxel_ellipsoid_data_route = "/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMVoxelEllipsoid/CloudData/";    //体素椭球数据储存地址
    if(!init)
    {
        int unused = system((std::string("exec rm -r ") + voxel_ellipsoid_data_route).c_str());
        unused = system((std::string("mkdir ") + voxel_ellipsoid_data_route).c_str());
        unused = system((std::string("exec rm -r ") + voxel_ellipsoid_data_route).c_str());
        unused = system((std::string("mkdir -p ") + voxel_ellipsoid_data_route).c_str());
        init += 1;
    }

    string voxel_ellipsoid_data_file = voxel_ellipsoid_data_route + v_eloid_id + ".csv";

    ofstream outFile;
    outFile.open(voxel_ellipsoid_data_file, ios::out);
    outFile << "ID" << ',' << "valid" << ',' << "num" << ',' << "a" << ',' << "b" << ',' << "c" << ',' 
            << "a-mean" << ',' << "b-mean" << ',' << "c-mean" << ',' 
            << "a-max" << ',' << "b-max" << ',' << "c-max" << ',' 
            << "mode" << ','
            << "a-axis-x" << ',' << "a-axis-y" << ',' << "a-axis-z" << ',' 
            << "b-axis-x" << ',' << "b-axis-y" << ',' << "b-axis-z" << ',' 
            << "c-axis-x" << ',' << "c-axis-y" << ',' << "c-axis-z" << ',' 
            << endl; 
    int id_index = 0;
    for(auto &data_it : v_eloid_data)
    {
        outFile << id_index << ',' << data_it.valid << ',' << data_it.point_num << ',' 
                << data_it.axis_length[0] << ',' << data_it.axis_length[1] << ',' << data_it.axis_length[2] << ',' 
                << data_it.center.x << ',' << data_it.center.y << ',' << data_it.center.z << ','
                << data_it.max_h_center.x << ',' << data_it.max_h_center.y << ',' << data_it.max_h_center.z << ','
                << data_it.mode << ','
                << data_it.axis(0,0) << ',' << data_it.axis(1,0) << ',' << data_it.axis(2,0) << ','
                << data_it.axis(0,1) << ',' << data_it.axis(1,1) << ',' << data_it.axis(2,1) << ',' 
                << data_it.axis(0,2) << ',' << data_it.axis(1,2) << ',' << data_it.axis(2,2) << ',' 
                << endl; 
        id_index++;
    }   

    outFile.close();
} 

//体素点云分布情况可视化 概率分布函数
void ClouDistrubutionVisualization(std::vector<Eigen::Vector3d> leaf_cloud, Eigen::MatrixXd axis_all, Eigen::Vector3d center, int index)
{
    Eigen::Vector3d max_axis = axis_all.col(0);
    vector<double> dist;
    double probability[50] = {0};

    for(auto it : leaf_cloud)
    {
        Eigen::Vector3d leaf_point_shift = it - center;
        if(leaf_point_shift.norm() == 0 || center.norm() == 0)
            continue; // don't count this sector pair.
        double distance = ((leaf_point_shift.dot(center)) / (center.norm()));
        dist.push_back(distance);
    }

    sort(dist.begin(),dist.end());
    double uint = abs(dist.back() - dist.front()) / 50.0;
    cout << endl << "index: " << index << " " << dist.front() << "  " << dist.back() << "   " << uint << endl;

    int point_num = dist.size();

    //求个点落在对应区间的数量
    for(auto it : dist)
    {
        int index = ceil((it - dist.front()) / uint);
        if (index > 50)
            index = 50;
        probability[index]  += 1;
    }

    //计算概率分布函数
    cout << "num: " << endl;
    for(int i = 1; i < 50; i++)
    {
        cout << probability[i] << ",";
        probability[i] += probability[i - 1];
    }

    cout << endl << "probability: " << endl;
    for(int i = 0; i < 50; i++)
    {
        probability[i] /= point_num;

        cout << probability[i] << ",";
    }
    cout << endl;
}