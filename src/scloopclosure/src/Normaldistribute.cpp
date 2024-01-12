#include "Normaldistribute.h"
#include "Scancontext.h"

void NDManager::NDmakeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> & _scan_cloud)
{
    // cout << "[ND] make descriptor matrix and key" << endl;
    Eigen::MatrixXd sc = NDmakeScancontext(_scan_cloud); // v1     //制作NDSC描述矩阵
    Eigen::MatrixXd ringkey = NDmakeRingkeyFromScancontext( sc ); //制作ring键值 每行平均值
    Eigen::MatrixXd sectorkey = NDmakeSectorkeyFromScancontext( sc ); //制作sector键值 每列平均值
    std::vector<float> polarcontext_invkey_vec = eig2stdvec( ringkey ); //将ring键值传入vector容器(以数组的形式)

    polarcontexts_.push_back( sc );     //保存描述矩阵
    polarcontext_invkeys_.push_back( ringkey ); //保存ring键值
    polarcontext_vkeys_.push_back( sectorkey ); //保存sector键值
    polarcontext_invkeys_mat_.push_back( polarcontext_invkey_vec ); //保存vector格式的ring键值  
    
}

MatrixXd NDManager::NDmakeScancontext(pcl::PointCloud<SCPointType> & _scan_cloud)
{
    TicToc t_making_desc;

    // main
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(ND_PC_NUM_RING, ND_PC_NUM_SECTOR);  //创建空（无点云）的SC矩阵

    int num_pts_scan_down = _scan_cloud.points.size();
    std::vector<std::vector<std::vector<Eigen::Vector3d> > > nd_uint_point_queue
                                                            (ND_PC_NUM_RING,std::vector<std::vector<Eigen::Vector3d> >(ND_PC_NUM_SECTOR,std::vector<Eigen::Vector3d>(0)));  //单元内的点的数组


    SCPointType pt;
    float azim_angle, azim_range; // wihtin 2d plane
    int ring_idx, sctor_idx;
    Eigen::Vector3d miu_piont; 
    miu_piont.Zero();

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
        
        // cout << "[ND] computer bin index" << endl;
        ring_idx = std::max( std::min( ND_PC_NUM_RING, int(ceil( (azim_range / ND_PC_MAX_RADIUS) * ND_PC_NUM_RING )) ), 1 );    //从1开始
        sctor_idx = std::max( std::min( ND_PC_NUM_SECTOR, int(ceil( (azim_angle / 360.0) * ND_PC_NUM_SECTOR )) ), 1 );
        
        // cout << "printf ring index: " << ring_idx << " sector index: " << sctor_idx << endl;
        Eigen::Vector3d piont_data = {pt.x,pt.y,pt.z};
        nd_uint_point_queue[ring_idx - 1][sctor_idx - 1].push_back(piont_data);
    }

    // cout << "[ND] finish point classcify" << endl;


    ring_idx = -1;
    sctor_idx = -1;

    //单元均值 协方差提取
    for(auto &ring_it : nd_uint_point_queue)
    {
        sctor_idx = -1;
        ring_idx++;
        for(auto &bin_it : ring_it)
        {
            sctor_idx++;
            // cout << bin_it.size() << " ";
            if(bin_it.size() < 6)
            {
                bin_it.clear();
                continue;
            }
            // 求均值
            for(auto &point_it : bin_it)
            {
                miu_piont += point_it;
            }

            miu_piont /= static_cast<double>(bin_it.size());
            
            //填充描述矩阵
            // cout << "push miu_z into descriptor matrix" << ring_idx << " " << sctor_idx << endl;
            desc(ring_idx, sctor_idx) = miu_piont[2];   //将均值z编码  
            // cout << "miu_z is " << miu_piont[2] << endl; 
        }
        // cout << endl;

    }

    //将无点的网格描述值 = 0
    for ( int row_idx = 0; row_idx < desc.rows(); row_idx++ )
        for ( int col_idx = 0; col_idx < desc.cols(); col_idx++ )
            if( desc(row_idx, col_idx) == NO_POINT )
                desc(row_idx, col_idx) = 0;

    // cout << "[ND] finish make ND descriptor" << endl;

    t_making_desc.toc("PolarContext making");

    return desc;
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

std::pair<int, float> NDManager::NDdetectLoopClosureID ( void )
{
    int loop_id { -1 }; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")

    // cout << "enter descriptor detect" << endl;

    // auto curr_key = polarcontext_invkeys_mat_.back(); // current observation (query)    //取出vector格式的ring键值
    // auto curr_desc = polarcontexts_.back(); // current observation (query)              //取出描述矩阵,最近的

    /* 
     * step 1: candidates from ringkey tree_
     */
    if( (int)polarcontext_invkeys_mat_.size() < ND_NUM_EXCLUDE_RECENT + 1) //储存的描述符数量是否足够
    {
        std::pair<int, float> result {loop_id, 0.0};
        // cout << "[ND] descriptor number is not enough" << endl;
        return result; // Early return 
    }

    auto curr_key = polarcontext_invkeys_mat_.back(); // current observation (query)    //取出vector格式的ring键值
    auto curr_desc = polarcontexts_.back(); // current observation (query)              //取出描述矩阵,最近的

    // tree_ reconstruction (not mandatory to make everytime) //重建kd树 10秒重建一次树
    if( tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        TicToc t_tree_construction;

        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.assign( polarcontext_invkeys_mat_.begin(), polarcontext_invkeys_mat_.end() - ND_NUM_EXCLUDE_RECENT ) ; //去除最近的30个描述符

        //复位polarcontext_tree_指针，并开辟一段动态内存用于存放 同时构造出来的KDTreeVectorOfVectorsAdaptor类
        polarcontext_tree_.reset(); 
        polarcontext_tree_ = std::make_unique<InvKeyTree>(ND_PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_, 10 /* max leaf */ );  //开辟内存同时构造InvKeyTree类 并在构造函数内完成重建树
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
        t_tree_construction.toc("Tree construction");
    }
    tree_making_period_conter = tree_making_period_conter + 1;
        
    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    // knn search
    std::vector<size_t> candidate_indexes( ND_NUM_CANDIDATES_FROM_TREE ); 
    std::vector<float> out_dists_sqr( ND_NUM_CANDIDATES_FROM_TREE );

    TicToc t_tree_search;
    nanoflann::KNNResultSet<float> knnsearch_result( ND_NUM_CANDIDATES_FROM_TREE );
    knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );
    polarcontext_tree_->index->findNeighbors( knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10) );    //传入当前描述符 kd树搜索
    t_tree_search.toc("Tree search");

    /* 
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    TicToc t_calc_dist;   
    for ( int candidate_iter_idx = 0; candidate_iter_idx < ND_NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++ )
    {
        MatrixXd polarcontext_candidate = polarcontexts_[ candidate_indexes[candidate_iter_idx] ];
        std::pair<double, int> sc_dist_result = NDdistanceBtnScanContext( curr_desc, polarcontext_candidate );    //返回相似度值和列向量偏移量
        
        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if( candidate_dist < min_dist ) //获取到最相似的描述符的相似度值 偏移量 和 描述符id
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;

            nn_idx = candidate_indexes[candidate_iter_idx];
        }
    }
    t_calc_dist.toc("Distance calc");

    //储存回环帧的id和相似度距离
    std::pair<int,float> data{(polarcontexts_.size()-1),min_dist};
    loopclosure_id_and_dist.push_back(data);

    /* 
     * loop threshold check
     */
    if( min_dist < ND_SC_DIST_THRES )  //是否达到回环阈值判断
    {
        loop_id = nn_idx; 
        
        std::cout.precision(3); 
        cout << "[ND] [Loop found] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
        // cout << "[ND] [Loop found] yaw diff: " << nn_align * ND_PC_UNIT_SECTORANGLE << " deg." << endl;
    }
    else
    {
        std::cout.precision(3); 
        cout << "[ND] [Not loop] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
        // cout << "[ND] [Not loop] yaw diff: " << nn_align * ND_PC_UNIT_SECTORANGLE << " deg." << endl;
    }

    // To do: return also nn_align (i.e., yaw diff)
    float yaw_diff_rad = deg2rad(nn_align * ND_PC_UNIT_SECTORANGLE);
    std::pair<int, float> result {loop_id, yaw_diff_rad};   //返回达到回环阈值的描述符id和旋转角度

    return result;

} // NDManager::NDdetectLoopClosureID

double NDManager::NDdistDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 )
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


std::pair<double, int> NDManager::NDdistanceBtnScanContext( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    // 1. fast align using variant key (not in original IROS18)
    MatrixXd vkey_sc1 = NDmakeSectorkeyFromScancontext( _sc1 );   //计算列的均值并以1行的形式返回整个矩阵的列向量均值
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
    return polarcontexts_.back();
}
