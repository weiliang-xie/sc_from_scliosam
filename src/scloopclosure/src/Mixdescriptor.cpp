#include "Mixdescriptor.h"
#include "Scancontext.h"
#include <cmath>

#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

void MIXManager::MIXmakeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> & _scan_cloud)
{
    // cout << "[MIX] make descriptor matrix and key" << endl;
    Eigen::MatrixXd sc = MIXmakeScancontext(_scan_cloud); // v1     //制作MIXSC描述矩阵
    Eigen::MatrixXd ringkey_ca = MIXmakeCARingkeyFromScancontext( sc ); //制作ring键值 每行平均值
    Eigen::MatrixXd ringkey_num = MIXmakeNUMRingkeyFromScancontext( sc ); //制作ring键值 每行平均值
    // Eigen::MatrixXd sectorkey = MIXmakeSectorkeyFromScancontext( sc ); //制作sector键值 每列平均值
    std::vector<float> polarcontext_invkey_vec_ca = eig2stdvec( ringkey_ca ); //将ring键值传入vector容器(以数组的形式)  强制转换成float
    std::vector<float> polarcontext_invkey_vec_num = eig2stdvec( ringkey_num ); //将ring键值传入vector容器(以数组的形式)


    polarcontexts_.push_back( sc );     //保存描述矩阵
    // polarcontext_invkeys_.push_back( ringkey_ca ); //保存ring键值
    // polarcontext_vkeys_.push_back( sectorkey ); //保存sector键值
    polarcontext_invkeys_mat_ca.push_back( polarcontext_invkey_vec_ca ); //保存vector格式的ring键值  
    polarcontext_invkeys_mat_num.push_back( polarcontext_invkey_vec_num ); //保存vector格式的ring键值  

    
}

MatrixXd MIXManager::MIXmakeScancontext(pcl::PointCloud<SCPointType> & _scan_cloud)
{
    TicToc t_making_desc;

    // main
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(MIX_PC_NUM_RING, MIX_PC_NUM_SECTOR);  //创建空（无点云）的SC矩阵

    int num_pts_scan_down = _scan_cloud.points.size();
    std::vector<std::vector<std::vector<Eigen::Vector3d> > > nd_uint_point_queue
                                                            (MIX_PC_NUM_RING,std::vector<std::vector<Eigen::Vector3d> >(MIX_PC_NUM_SECTOR,std::vector<Eigen::Vector3d>(0)));  //单元内的点的数组
    std::vector<Eigen::MatrixXd> bin_cov;   //单个描述符的单元协方差序列
    std::vector<Eigen::MatrixXd> bin_singular;   //单个描述符的单元奇异值序列



    SCPointType pt;
    float azim_angle, azim_range; // wihtin 2d plane
    int ring_idx, sctor_idx;

    for(int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        pt.x = _scan_cloud.points[pt_idx].x;                 //循环获取帧内各点
        pt.y = _scan_cloud.points[pt_idx].y;
        pt.z = _scan_cloud.points[pt_idx].z + LIDAR_HEIGHT; // naive adding is ok (all points should be > 0).

        // xyz to ring, sector
        azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);   //求点到传感器中心的距离
        azim_angle = xy2theta(pt.x, pt.y);              //输出theta角        

        if( azim_range > MIX_PC_MAX_RADIUS )                //判断点距离传感器距离是否超出最大值
            continue;
        
        // cout << "[MIX] computer bin index" << endl;
        ring_idx = std::max( std::min( MIX_PC_NUM_RING, int(ceil( (azim_range / MIX_PC_MAX_RADIUS) * MIX_PC_NUM_RING )) ), 1 );    //从1开始
        sctor_idx = std::max( std::min( MIX_PC_NUM_SECTOR, int(ceil( (azim_angle / 360.0) * MIX_PC_NUM_SECTOR )) ), 1 );
        
        // cout << "printf ring index: " << ring_idx << " sector index: " << sctor_idx << endl;
        Eigen::Vector3d piont_data = {pt.x,pt.y,pt.z};
        nd_uint_point_queue[ring_idx - 1][sctor_idx - 1].push_back(piont_data);
    }

    // cout << "[MIX] finish point classcify" << endl;


    ring_idx = -1;
    sctor_idx = -1;
    //单元均值
    for(auto &ring_it : nd_uint_point_queue)
    {
        sctor_idx = -1;
        ring_idx++;
        for(auto &bin_it : ring_it)
        {
            sctor_idx++;
            // cout << bin_it.size() << " ";
            if(bin_it.size() < 50)      //6->10
            {   
                double max_high = -1000;
                for(auto &point_it : bin_it)
                {
                    if(point_it[2] > max_high)
                        max_high = point_it[2];
                }
                desc(ring_idx,sctor_idx) = max_high;
                // desc(ring_idx,sctor_idx) = bin_it.size();
                bin_it.clear();
                continue;
            }

            //计算协方差 奇异值
            Eigen::MatrixXd bin_cov_;
            Eigen::MatrixXd bin_singular_;
            bin_cov_ = MIXGetCovarMatrix(bin_it);
            bin_cov.push_back(bin_cov_);
            bin_singular_ = MIXGetSingularvalue(bin_cov_);
            bin_singular.push_back(bin_singular_);

            //计算椭球扁平程度
            double flat_ratio;
            if(bin_singular_.cols() != 1){
                cout << "The singular value is error !!!" << endl;
            }else{
                if(bin_singular_(2,0) != 0)
                    flat_ratio = bin_singular_(2,0) / bin_singular_(0,0);
                else flat_ratio = 0;
            }
            
            //填充描述矩阵
            desc(ring_idx,sctor_idx) = flat_ratio;

        }
        // cout << endl;

    }

    //将无点的网格描述值 = 0
    for ( int row_idx = 0; row_idx < desc.rows(); row_idx++ )
        for ( int col_idx = 0; col_idx < desc.cols(); col_idx++ )
            if( desc(row_idx, col_idx) == NO_POINT )
                desc(row_idx, col_idx) = 0;

    // cout << "[MIX] finish make MIX descriptor" << endl;

    t_making_desc.toc("PolarContext making");

    return desc;
}

//制作ring key
MatrixXd MIXManager::MIXmakeCARingkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: rowwise mean vector
    */
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for ( int row_idx = 0; row_idx < _desc.rows(); row_idx++ )
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        for(int i = 0; i < curr_row.cols(); ++i){
            if(curr_row(0,i) >= 1 || curr_row(0,i) < 0)
            // if(curr_row(0,i) >= 1)
                curr_row(0,i) = 0;
        }
        invariant_key(row_idx, 0) = curr_row.mean();    //求解每行的平均值
    }

    // cout << "AC ring key is: " << invariant_key << endl;
    return invariant_key;
} // MIXManager::makeRingkeyFromScancontext

//制作ring key
MatrixXd MIXManager::MIXmakeNUMRingkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: rowwise mean vector
    */
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for ( int row_idx = 0; row_idx < _desc.rows(); row_idx++ )
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        for(int i = 0; i < curr_row.cols(); ++i){
            if(curr_row(0,i) < 1 && curr_row(0,i) > 0)
            // if(curr_row(0,i) < 1)
                curr_row(0,i) = 0;
        }
        invariant_key(row_idx, 0) = curr_row.mean();    //求解每行的平均值
    }

    // cout << "NUM ring key is: " << invariant_key << endl;
    return invariant_key;
} // MIXManager::makeRingkeyFromScancontext

//制作sector key
MatrixXd MIXManager::MIXmakeSectorkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    /* 
     * summary: columnwise mean vector
    */
    Eigen::MatrixXd variant_key = Eigen::MatrixXd::Zero(2, _desc.cols());
    for ( int col_idx = 0; col_idx < _desc.cols(); col_idx++ )
    {
        Eigen::MatrixXd curr_col = _desc.col(col_idx);
        for(int i = 0; i < curr_col.rows(); ++i){
            if(curr_col(i,0) < 1 && curr_col(i,0) > 0){
                variant_key(0, col_idx) += curr_col(i,0);   //ac
            }else{
                variant_key(1, col_idx) += curr_col(i,0);   //num
            }
        }
    }
    variant_key /= _desc.rows();
    // cout << "variant is: " << endl << variant_key << endl;

    return variant_key;
} // MIXManager::makeSectorkeyFromScancontext

std::pair<int, float> MIXManager::MIXdetectLoopClosureID ( void )
{
    int loop_id { -1 }; // init with -1, -1 means no loop (== LeGO-LOAM's variable "closestHistoryFrameID")

    // cout << "enter descriptor detect" << endl;

    // auto curr_key = polarcontext_invkeys_mat_.back(); // current observation (query)    //取出vector格式的ring键值
    // auto curr_desc = polarcontexts_.back(); // current observation (query)              //取出描述矩阵,最近的


    if( (int)polarcontext_invkeys_mat_ca.size() < MIX_NUM_EXCLUDE_RECENT + 1 ) //储存的描述符数量是否足够
    {
        std::pair<int, float> result {loop_id, 0.0};
        // cout << "[MIX] descriptor number is not enough" << endl;
        return result; // Early return 
    }

    /* 
     * step 1: candidates from ringkey tree_
     */
    auto curr_key_ca = polarcontext_invkeys_mat_ca.back(); // current observation (query)    //取出vector格式的ac-ring键值
    auto curr_key_num = polarcontext_invkeys_mat_num.back(); // current observation (query)    //取出vector格式的num-ring键值
    auto curr_desc = polarcontexts_.back(); // current observation (query)              //取出描述矩阵,最近的

    //ca 部分
    // tree_ reconstruction (not mandatory to make everytime) //重建kd树 10秒重建一次树
    if( tree_making_period_conter_ca % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        TicToc t_tree_construction_ca;

        polarcontext_invkeys_to_search_ca.clear();
        polarcontext_invkeys_to_search_ca.assign( polarcontext_invkeys_mat_ca.begin(), polarcontext_invkeys_mat_ca.end() - MIX_NUM_EXCLUDE_RECENT ) ; //去除最近的30个描述符

        //复位polarcontext_tree_指针，并开辟一段动态内存用于存放 同时构造出来的KDTreeVectorOfVectorsAdaptor类
        polarcontext_tree_ca.reset(); 
        polarcontext_tree_ca = std::make_unique<InvKeyTree>(MIX_PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_ca, 10 /* max leaf */ );  //开辟内存同时构造InvKeyTree类 并在构造函数内完成重建树
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
        t_tree_construction_ca.toc("Tree construction");
    }
    tree_making_period_conter_ca = tree_making_period_conter_ca + 1;       

    // knn search
    std::vector<size_t> candidate_indexes_ca( MIX_NUM_CANDIDATES_FROM_TREE ); 
    std::vector<float> out_dists_sqr_ca( MIX_NUM_CANDIDATES_FROM_TREE );

    TicToc t_tree_search_ca;
    nanoflann::KNNResultSet<float> knnsearch_result_ca( MIX_NUM_CANDIDATES_FROM_TREE );
    knnsearch_result_ca.init( &candidate_indexes_ca[0], &out_dists_sqr_ca[0] );
    polarcontext_tree_ca->index->findNeighbors( knnsearch_result_ca, &curr_key_ca[0] /* query */, nanoflann::SearchParams(10) );    //传入当前描述符 kd树搜索
    t_tree_search_ca.toc("Tree search");

    //num 部分
    // tree_ reconstruction (not mandatory to make everytime) //重建kd树 10秒重建一次树
    if( tree_making_period_conter_num % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        TicToc t_tree_construction_num;

        polarcontext_invkeys_to_search_num.clear();
        polarcontext_invkeys_to_search_num.assign( polarcontext_invkeys_mat_num.begin(), polarcontext_invkeys_mat_num.end() - MIX_NUM_EXCLUDE_RECENT ) ; //去除最近的30个描述符

        //复位polarcontext_tree_指针，并开辟一段动态内存用于存放 同时构造出来的KDTreeVectorOfVectorsAdaptor类
        polarcontext_tree_num.reset(); 
        polarcontext_tree_num = std::make_unique<InvKeyTree>(MIX_PC_NUM_RING /* dim */, polarcontext_invkeys_to_search_num, 10 /* max leaf */ );  //开辟内存同时构造InvKeyTree类 并在构造函数内完成重建树
        // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
        t_tree_construction_num.toc("Tree construction");
    }
    tree_making_period_conter_num = tree_making_period_conter_num + 1;

    // knn search
    std::vector<size_t> candidate_indexes_num( MIX_NUM_CANDIDATES_FROM_TREE ); 
    std::vector<float> out_dists_sqr_num( MIX_NUM_CANDIDATES_FROM_TREE );

    TicToc t_tree_search_num;
    nanoflann::KNNResultSet<float> knnsearch_result_num( MIX_NUM_CANDIDATES_FROM_TREE );
    knnsearch_result_num.init( &candidate_indexes_num[0], &out_dists_sqr_num[0] );
    polarcontext_tree_num->index->findNeighbors( knnsearch_result_num, &curr_key_num[0] /* query */, nanoflann::SearchParams(10) );    //传入当前描述符 kd树搜索
    t_tree_search_num.toc("Tree search");



    double min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;

    std::vector<size_t> candidate_indexes; 
    candidate_indexes.insert(candidate_indexes.begin(),candidate_indexes_ca.begin(),candidate_indexes_ca.end());
    candidate_indexes.insert(candidate_indexes.end(),candidate_indexes_num.begin(),candidate_indexes_num.end());

    // cout << "candidate_indexs size is: " << candidate_indexes.size() << endl;

    /* 
     *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
     */
    TicToc t_calc_dist;   
    for ( int candidate_iter_idx = 0; candidate_iter_idx < MIX_NUM_CANDIDATES_FROM_TREE*2; candidate_iter_idx++ )
    {
        MatrixXd polarcontext_candidate = polarcontexts_[ candidate_indexes[candidate_iter_idx] ];
        std::pair<double, int> sc_dist_result = MIXdistanceBtnScanContext( curr_desc, polarcontext_candidate );    //返回相似度值和列向量偏移量
        
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
    if( min_dist < MIX_SC_DIST_THRES )  //是否达到回环阈值判断
    {
        loop_id = nn_idx; 
        
        std::cout.precision(3); 
        cout << "[MIX] [Loop found] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
        // cout << "[MIX] [Loop found] yaw diff: " << nn_align * MIX_PC_UNIT_SECTORANGLE << " deg." << endl;
    }
    else
    {
        std::cout.precision(3); 
        cout << "[MIX] [Not loop] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
        // cout << "[MIX] [Not loop] yaw diff: " << nn_align * MIX_PC_UNIT_SECTORANGLE << " deg." << endl;
    }

    // To do: return also nn_align (i.e., yaw diff)
    float yaw_diff_rad = deg2rad(nn_align * MIX_PC_UNIT_SECTORANGLE);
    std::pair<int, float> result {loop_id, yaw_diff_rad};   //返回达到回环阈值的描述符id和旋转角度

    return result;

} // MIXManager::MIXdetectLoopClosureID

double MIXManager::MIXdistDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    int num_eff_cols = 0; // i.e., to exclude all-nonzero sector
    double sum_sector_similarity = 0;
    for ( int col_idx = 0; col_idx < _sc1.cols(); col_idx++ )
    {
        // col_sc1_ac,col_sc1_num,col_sc2_ac,col_sc2_num;
        VectorXd col_sc1_ac = Eigen::VectorXd::Zero(MIX_PC_NUM_RING);
        VectorXd col_sc1_num = Eigen::VectorXd::Zero(MIX_PC_NUM_RING);
        VectorXd col_sc2_ac = Eigen::VectorXd::Zero(MIX_PC_NUM_RING);
        VectorXd col_sc2_num = Eigen::VectorXd::Zero(MIX_PC_NUM_RING);

        for(int row_idx = 0; row_idx < _sc1.rows(); row_idx++)
        {
            if(_sc1(row_idx,col_idx) < 1 && _sc1(row_idx,col_idx) > 0)
            // if(_sc1(row_idx,col_idx) < 1)
                col_sc1_ac(row_idx) = _sc1(row_idx,col_idx);
            else
                col_sc1_num(row_idx) = _sc1(row_idx,col_idx);              
            
            if(_sc2(row_idx,col_idx) < 1 && _sc2(row_idx,col_idx) > 0)
            // if(_sc2(row_idx,col_idx) < 1)
                col_sc2_ac(row_idx) = _sc2(row_idx,col_idx);
            else
                col_sc2_num(row_idx) = _sc2(row_idx,col_idx);               
                       
        }
        
        if( (col_sc1_ac.norm() == 0) || (col_sc2_ac.norm() == 0) || (col_sc1_num.norm() == 0) || (col_sc2_num.norm() == 0))
            continue; // don't count this sector pair. 

        double sector_similarity = (col_sc1_ac.dot(col_sc2_ac) / (col_sc1_ac.norm() * col_sc2_ac.norm())) * 0.5 + (col_sc1_num.dot(col_sc2_num) / (col_sc1_num.norm() * col_sc2_num.norm())) * 0.5;

        sum_sector_similarity = sum_sector_similarity + sector_similarity;
        num_eff_cols = num_eff_cols + 1;
    }
    
    double sc_sim = sum_sector_similarity / num_eff_cols;
    // cout << "distance: " << 1.0 - sc_sim << endl;
    return 1.0 - sc_sim;

} // distDirectSC


int MIXManager::MIXfastAlignUsingVkey( MatrixXd & _vkey1, MatrixXd & _vkey2)
{
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    for ( int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++ )
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);

        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;

        double cur_diff_norm_ca = vkey_diff.row(0).norm();
        double cur_diff_norm_num = vkey_diff.row(1).norm();

        double cur_diff_norm = cur_diff_norm_ca + cur_diff_norm_num;
        if( cur_diff_norm < min_veky_diff_norm )
        {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }

    return argmin_vkey_shift;

} // fastAlignUsingVkey


std::pair<double, int> MIXManager::MIXdistanceBtnScanContext( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    // 1. fast align using variant key (not in original IROS18)
    MatrixXd vkey_sc1 = MIXmakeSectorkeyFromScancontext( _sc1 );   //计算列的均值并以行的形式返回整个矩阵的列向量均值
    // cout << "sc1 cols avg is: " << vkey_sc1 << endl;
    MatrixXd vkey_sc2 = MIXmakeSectorkeyFromScancontext( _sc2 );
    int argmin_vkey_shift = MIXfastAlignUsingVkey( vkey_sc1, vkey_sc2 );   //将列均值向右移动并使用F-范数进行对比，返回偏移值
    // cout << "argmin vkey shift is: "  << argmin_vkey_shift << endl;

    const int SEARCH_RADIUS = round( 0.5 * MIX_SEARCH_RATIO * _sc1.cols() ); // a half of search range  //搜索范围
    std::vector<int> shift_idx_search_space { argmin_vkey_shift };
    for ( int ii = 1; ii < SEARCH_RADIUS + 1; ii++ )
    {
        shift_idx_search_space.push_back( (argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols() );
        shift_idx_search_space.push_back( (argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols() );
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());   // 排序 //在vector中填入以获取到的偏移量为中心，前后范围为搜索范围的待匹配偏移量

    // 2. fast columnwise diff 
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for ( int num_shift: shift_idx_search_space )
    {
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);  //列位移函数
        double cur_sc_dist = MIXdistDirectSC( _sc1, sc2_shifted ); //计算相似度，计算各个列向量的余弦距离的和的平均值（去除行列式为0的部分）
        if( cur_sc_dist < min_sc_dist )
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }

    return make_pair(min_sc_dist, argmin_shift);

} // distanceBtnScanContext

const Eigen::MatrixXd& MIXManager::MIXgetConstRefRecentSCD(void)
{
    return polarcontexts_.back();
}

 
Eigen::MatrixXd MIXManager::MIXGetCovarMatrix(std::vector<Eigen::Vector3d> bin_piont)
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
 
	return covar;
}

 
Eigen::MatrixXd MIXManager::MIXGetSingularvalue(Eigen::MatrixXd bin_cov)
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