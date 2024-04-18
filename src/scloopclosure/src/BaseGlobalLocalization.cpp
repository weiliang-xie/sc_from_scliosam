#include "BaseGlobalLocalization.h"
#include "Scancontext.h"
#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include <numeric>

using namespace std;

int TestClouDistrubutionVisualization(std::vector<Eigen::Vector3d> leaf_cloud, Eigen::Matrix3d axis_all, Eigen::Vector3d center);

//划分体素
//返回值：分割后的体素点云储存容器 二维vector[x方向索引][y方向索引] （从0开始，从最小的点所在的体素开始）
void BaseGlobalLocalization::DivideVoxel(pcl::PointCloud<SCPointType> & _scan_cloud)
{
    Frame_Voxeldata voxel_data_;
    int num_pts_scan_down = _scan_cloud.points.size();
    // cout << "[EL]  Divide Voxel   point num is: " << num_pts_scan_down << endl;

    for(int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        int horizontal_idx,vertical_idx;
        SCPointType pt;
        pt.x = _scan_cloud.points[pt_idx].x;                                //循环获取帧内各点
        pt.y = _scan_cloud.points[pt_idx].y;
        pt.z = _scan_cloud.points[pt_idx].z + LIDAR_HEIGHT;    

        if( abs(pt.x) > MAX_RANGE || abs(pt.y) > MAX_RANGE )                //判断点距离传感器距离是否超出最大值
            continue;
        
        //寻找z方向最小 最大值
        if(pt.z < voxel_data_.min_point_z)
            voxel_data_.min_point_z = pt.z;
        if(pt.z > voxel_data_.max_point_z)
            voxel_data_.max_point_z = pt.z;
        
        // cout << "[EL]  computer bin index" << endl;
        horizontal_idx = floor(double((pt.x + 80.0) / VOXEL_UNIT_HORIZANTAL));
        vertical_idx = floor(double((pt.y + 80.0) / VOXEL_UNIT_VERTICAL));
        
        // cout << "[EL]  printf ring index: " << ring_idx << " sector index: " << sector_idx << endl;
        Eigen::Vector3d point_data = {pt.x,pt.y,pt.z};
        voxel_data_.origin_voxel_data[horizontal_idx * VOXEL_NUM_VERTICAL + vertical_idx].point_cloud.push_back(point_data);
    }

    //填充体素坐标系和尺寸
    for(auto &voxel_it : voxel_data_.origin_voxel_data)
    {
        voxel_it.second.length << VOXEL_UNIT_HORIZANTAL, VOXEL_UNIT_VERTICAL, (voxel_data_.max_point_z - voxel_data_.min_point_z);

        int x_index = (int)(voxel_it.first / VOXEL_NUM_VERTICAL);
        int y_index = voxel_it.first % (int)(VOXEL_NUM_VERTICAL);
        voxel_it.second.coordinate[0] = MAX_RANGE * (-1) + x_index * VOXEL_UNIT_HORIZANTAL;
        voxel_it.second.coordinate[1] = MAX_RANGE * (-1) + y_index * VOXEL_UNIT_VERTICAL;
        voxel_it.second.coordinate[2] = voxel_data_.min_point_z;
        // cout << "[EL]  Divide Voxel x index: " << x_index << " y index: " << y_index << endl;
    }

    // cout << "[EL]  Divide Voxel  voxel num: " << voxel_data_.origin_voxel_data.size() << endl;
    // cout << "[EL]  Divide Voxel  max z: " << voxel_data_.max_point_z << " min z: " << voxel_data_.min_point_z << endl;
    frame_voxel.push_back(voxel_data_);


    //自适应划分体素
    AdaptiveSegmentation(voxel_data_.origin_voxel_data);
    // cout << "[EL]  Divide Voxel  cur frame origin voxel num: " << frame_voxel.back().origin_voxel_data.size() << endl;
    // cout << "[EL]  Divide Voxel  cur frame segment voxel num: " << frame_voxel.back().seg_voxel_data.size() << endl;
    // cout << endl;

    //自适应体素效果测试
    double seg_voxel_num = 0;
    double ori_voxel_num = 0;
    double seg_little_voxel_num = 0;
    double ori_little_voxel_num = 0;
    double seg_valid_voxel_num = 0;
    double ori_valid_voxel_num = 0;
    for(auto seg_frame_it : frame_voxel.back().seg_voxel_data)
    {
        if(seg_frame_it.point_cloud.size() > MIN_VAILD_VOXEL_POINT_NUM * 2)
        {
            Ellipsoid seg_eloid;
            std::pair<Eigen::MatrixXd, Eigen::MatrixXd> voxel_mean_cov = GetCovarMatrix(seg_frame_it.point_cloud);
            std::pair<Eigen::Vector3d,Eigen::Matrix3d> voxel_eigen_ = GetEigenvalues(voxel_mean_cov.second); 
            Eigen::Vector3d center_ = voxel_mean_cov.first.row(0);
            if(TestClouDistrubutionVisualization(seg_frame_it.point_cloud, voxel_eigen_.second, center_))
                seg_voxel_num++;
        }
        else if(seg_frame_it.point_cloud.size() < MIN_VAILD_VOXEL_POINT_NUM){
            seg_little_voxel_num++;
        }
        if(seg_frame_it.point_cloud.size() > MIN_VAILD_VOXEL_POINT_NUM)
            seg_valid_voxel_num++;
    }

    for(auto ori_frame_it : frame_voxel.back().origin_voxel_data)
    {
        if(ori_frame_it.second.point_cloud.size() > MIN_VAILD_VOXEL_POINT_NUM * 2)
        {
            Ellipsoid seg_eloid;
            std::pair<Eigen::MatrixXd, Eigen::MatrixXd> voxel_mean_cov = GetCovarMatrix(ori_frame_it.second.point_cloud);
            std::pair<Eigen::Vector3d,Eigen::Matrix3d> voxel_eigen_ = GetEigenvalues(voxel_mean_cov.second); 
            Eigen::Vector3d center_ = voxel_mean_cov.first.row(0);
            if(TestClouDistrubutionVisualization(ori_frame_it.second.point_cloud, voxel_eigen_.second, center_))
                ori_voxel_num++;
        }
        else if(ori_frame_it.second.point_cloud.size() < MIN_VAILD_VOXEL_POINT_NUM){
            ori_little_voxel_num++;
        }
        if(ori_frame_it.second.point_cloud.size() > MIN_VAILD_VOXEL_POINT_NUM)
            ori_valid_voxel_num++;
    }
    // cout << "[EL]  Divide Voxel  sgement voxel num: " << seg_voxel_num << " origin voxel num: " << ori_voxel_num << endl;
    std::pair<double, double> num_ = {seg_voxel_num,ori_voxel_num};
    std::pair<double, double> little_num_ = {seg_little_voxel_num,ori_little_voxel_num};
    std::pair<double, double> valid_num_ = {seg_valid_voxel_num,ori_valid_voxel_num};
    frame_seg_ori_littlevoxel_num.push_back(little_num_);
    frame_seg_ori_validvoxel_num.push_back(valid_num_);
    frame_seg_ori_num.push_back(num_);
}

//构建椭球模型 输入划分好体素的点云
Frame_Ellipsoid BaseGlobalLocalization::BulidingEllipsoidModel(void)
{
    Frame_Ellipsoid _frame_eloid;

#if SEGMENT_VOXEL_ENABLE
    for(auto &voxel_it : frame_voxel.back().seg_voxel_data)
#else  
    for(auto &voxel_it : frame_voxel.back().origin_voxel_data)
#endif
    {

#if SEGMENT_VOXEL_ENABLE
        std::vector<Eigen::Vector3d> point_ = voxel_it.point_cloud;
#else  
        std::vector<Eigen::Vector3d> point_ = voxel_it.second.point_cloud;
#endif

        Ellipsoid eloid;
        if(point_.size() < MIN_VAILD_VOXEL_POINT_NUM)                  //不处理少于20个点的体素
        {
            continue;
        }else{
            eloid.point_num = point_.size();
            std::pair<Eigen::MatrixXd, Eigen::MatrixXd> voxel_mean_cov = GetCovarMatrix(point_);
            std::pair<Eigen::Vector3d,Eigen::Matrix3d> voxel_eigen_ = GetEigenvalues(voxel_mean_cov.second);
            eloid.center = voxel_mean_cov.first.row(0);
            eloid.cov = voxel_mean_cov.second;
            eloid.eigen = voxel_eigen_.first;
            eloid.axis = voxel_eigen_.second;
            eloid.eloid_vaild = IsEllipsoidVaild(eloid);
            eloid.mode = ClassifyEllipsoid(eloid);

            //求解高度方向上的点云存在二值化情况 num_exit max_z_point
            for(auto point_it : point_)
            {
                int bit_shift = 0;
                bit_shift = floor(point_it[2] / 0.5) > 8 ? 8 : (floor(point_it[2] / 0.5) < 0 ? 0 : floor(point_it[2] / 0.5));
                // cout << point_it[2] << ",";
                if((eloid.num_exit & (1 << bit_shift)) == 0)
                {
                    // cout << bit_shift << ",";
                    eloid.num_exit |=  1 << bit_shift; 
                }
                
                //max_z_point
                if(point_it[2] > eloid.max_z_point[2])
                    eloid.max_z_point = point_it;
            }
            // cout << endl;
            //椭球是否有效
            if(eloid.eloid_vaild == 1)
            {
                //是否为地面椭球
                if(eloid.mode <= 2 && eloid.center[2] < GROUND_HEIGHT)
                    _frame_eloid.ground_voxel_eloid.push_back(eloid);
                else
                    _frame_eloid.nonground_voxel_eloid.push_back(eloid);
            }
        }
    }
    return _frame_eloid;
}



//椭球模型是否有效     返回当前椭球的有效性  -1：椭球不存在 0：椭球无效  1：椭球有效
bool BaseGlobalLocalization::IsEllipsoidVaild(Ellipsoid voxeleloid)
{
    if(voxeleloid.point_num >= MIN_VAILD_VOXEL_POINT_NUM)
    {
        return 1;
    }else{
        return 0;
    }
}

//椭球模型分类      返回当前椭球的类型： 1：线性 2：平面型 3：立体型
int BaseGlobalLocalization::ClassifyEllipsoid(Ellipsoid voxeleloid)
{
    double a = sqrt(voxeleloid.eigen[0]);
    double b = sqrt(voxeleloid.eigen[1]);
    double c = sqrt(voxeleloid.eigen[2]);
    return (((a-b) > (b-c)) && ((a-b) > c)) ? 1 : ((((b-c) > (a-b)) && ((b-c) > c)) ? 2 : 3);
}



//求解协方差 输入 点云集合  输出 均值-协方差对
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> BaseGlobalLocalization::GetCovarMatrix(std::vector<Eigen::Vector3d> bin_piont)
{
	// reference: https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
	const int rows = bin_piont.size();
    const int cols = bin_piont[0].size();

    // cout << "[EL]  rows: " << rows << " cols: " << cols << endl;
 
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
	// std::cout << "[EL]  print mean: " << std::endl << mean << std::endl;
 
	Eigen::MatrixXd tmp(rows, cols);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			tmp(y, x) = m(y, x) - mean(0, x);
		}
	}
	//std::cout << "[EL]  tmp: " << std::endl << tmp << std::endl;
 
	Eigen::MatrixXd covar = (tmp.adjoint() * tmp) / float(nsamples - 1);    //求协方差矩阵
	// std::cout << "[EL]  print covariance matrix: " << std::endl << covar << std::endl << std::endl;
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> result = {mean,covar};
 
	return result;
}

//特征值分解 返回值为降序 输入 协方差  输出 特征值-特征向量对
std::pair<Eigen::Vector3d,Eigen::Matrix3d> BaseGlobalLocalization::GetEigenvalues(Eigen::MatrixXd bin_cov)
{
    // cout << "[EL]  Here is a 3x3 matrix, bin_cov:" << endl << bin_cov << endl << endl;
    EigenSolver<Matrix3d> es(bin_cov);
	
	Matrix3d D = es.pseudoEigenvalueMatrix();
	Matrix3d V = es.pseudoEigenvectors();
	// cout << "[EL]  The pseudo-eigenvalue matrix D is:" << endl << D << endl;
	// cout << "[EL]  The pseudo-eigenvector matrix V is:" << endl << V << endl;
	// cout << "[EL]  Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;

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
	// cout << "[EL]  The pseudo-eigenvector matrix V is:" << endl << V << endl;
	// cout << "[EL]  Finally, V * D * V^(-1) = " << endl << V * D * V.inverse() << endl;

    //特征值赋值到vector
    Eigen::Vector3d eigen_value;
    eigen_value << D(0,0),D(1,1),D(2,2);
    // std::cout << "[EL]  eigen value: " << eigen_value[0] << " " << eigen_value[1] << " " << eigen_value[2] << endl;

    //将特征值和特征向量用pair形式返回
    std::pair<Eigen::Vector3d,Eigen::Matrix3d> result = {eigen_value,V};

    return result;
}

/*-------------------------SC改版描述符部分-------------------------*/

std::vector<float> BaseGlobalLocalization::MakeAndSaveDescriptorAndKey(std::map<int, VoxelData> origin_voxel_, int frame_id)
{
    Eigen::MatrixXd descriptor = GetMatrixDescriptor(origin_voxel_);
    Eigen::MatrixXd ver_key = MakeVerticalKeyFromDescriptor(descriptor);
    std::vector<float> vertical_invkey_vec = eig2stdvec(ver_key);

    // cout << "[EL]  Make Descriptor   descriptor: " << descriptor << endl;

    return vertical_invkey_vec;
}

Eigen::MatrixXd BaseGlobalLocalization::GetMatrixDescriptor(std::map<int, VoxelData> origin_voxel_)
{
    // cout << "[EL]  Make Descriptor   origin voxel num: " << origin_voxel_.size() << endl;
    const int NO_POINT = -1000;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(VOXEL_NUM_VERTICAL, VOXEL_NUM_HORIZONTAL);  //创建空（无点云）的SC矩阵

    for(auto voxel_it : origin_voxel_)
    {
        int horizontal, vertical;
        horizontal =  voxel_it.first / VOXEL_NUM_VERTICAL;
        vertical = voxel_it.first % VOXEL_NUM_VERTICAL;
        //均值高度 0.87
        double mean_z = 0;
        for(auto point_it : voxel_it.second.point_cloud)
        {
            mean_z += point_it[2];
        }        
        mean_z /= voxel_it.second.point_cloud.size();
        desc(horizontal,vertical) = mean_z;

        //最大高度 0.87
        // double max_z = -1000;
        // for(auto point_it : voxel_it.second.point_cloud)
        // {
        //     if(point_it[2] > max_z)
        //         max_z = point_it[2];
        // }        
        // desc(horizontal,vertical) = max_z;  //这里方向对换

        //点云存在二值 0.87
        // int num_exit = 0;
        // for(auto point_it : voxel_it.second.point_cloud)
        // {
        //     int bit_shift = 0;
        //     bit_shift = floor(point_it[2] / 0.5) > 8 ? 8 : (floor(point_it[2] / 0.5) < 0 ? 0 : floor(point_it[2] / 0.5));
        //     // cout << point_it[2] << ",";
        //     if((num_exit & (1 << bit_shift)) == 0)
        //     {
        //         // cout << bit_shift << ",";
        //         num_exit |=  1 << bit_shift; 
        //     }
        // }
        // desc(horizontal,vertical) = num_exit;  //这里方向对换

    }

    //将无点的网格描述值 = 0
    for ( int row_idx = 0; row_idx < desc.rows(); row_idx++ )
        for ( int col_idx = 0; col_idx < desc.cols(); col_idx++ )
            if( desc(row_idx, col_idx) == NO_POINT )
                desc(row_idx, col_idx) = 0;
    
    // cout << "[EL]  Make Descriptor   finish make descriptor" << endl;
    return desc;
}

//制作vertical key
Eigen::MatrixXd BaseGlobalLocalization::MakeVerticalKeyFromDescriptor( Eigen::MatrixXd &_desc )
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
} 

/*-------------------------自适应划分体素部分-------------------------*/

//返回：是否划分标志
void BaseGlobalLocalization::AdaptiveSegmentation(std::map<int, VoxelData> origin_voxel_)
{
    Frame_Voxeldata frame_voxel_;
    double tp = 0;
    double all_voxel_pre_test = 0; //预测已经分割总数
    double all_voxel_gt_seg_test = 0;  //实际需要分割总数
    for(auto &origin_voxel_it : origin_voxel_)
    {
        bool seg_state = true;
        std::vector<VoxelData> seg_voxel_;
        // cout << "[EL]  Adaptive Segment  enter judge segment voxel" << endl;
        if(origin_voxel_it.second.point_cloud.size() > MIN_VAILD_VOXEL_POINT_NUM * 2)
        {
            //获取特征向量 特征值 点云类型
            std::pair<Eigen::MatrixXd, Eigen::MatrixXd> voxel_mean_cov = GetCovarMatrix(origin_voxel_it.second.point_cloud);
            std::pair<Eigen::Vector3d,Eigen::Matrix3d> voxel_eigen_ = GetEigenvalues(voxel_mean_cov.second);   
            Ellipsoid voxel_eloid;
            voxel_eloid.eigen = voxel_eigen_.first;
            voxel_eloid.mode = ClassifyEllipsoid(voxel_eloid); 

            Eigen::Matrix<double, 1, 3> cur_vector_0; 
            Eigen::Matrix<double, 1, 3> cur_vector_1; 
            Eigen::Matrix<double, 1, 3> cur_vector_2; 

            cur_vector_0.row(0) = voxel_eigen_.second.col(0); 
            cur_vector_1.row(0) = voxel_eigen_.second.col(1); 
            cur_vector_2.row(0) = voxel_eigen_.second.col(2); 

            //处理线性模型
            if(voxel_eloid.mode == 1)
            {
                // cout << "[EL]  Adaptive Segment  voxel mode is: line" << endl;
                seg_voxel_ = project_pt(cur_vector_0, origin_voxel_it.second, voxel_eloid.center, 
                                        origin_voxel_it.second.coordinate, origin_voxel_it.second.length);
                // if(seg_voxel_.empty())
                // {
                //     seg_voxel_ = project_pt(cur_vector_1, origin_voxel_it.second, voxel_eloid.center, 
                //                             origin_voxel_it.second.coordinate, origin_voxel_it.second.length);                    
                // }
                if(seg_voxel_.empty())
                {
                    seg_state = false;
                }

            }else if(voxel_eloid.mode == 2)
            {
                // cout << "[EL]  Adaptive Segment  voxel mode is: place" << endl;
                seg_voxel_ = project_pt(cur_vector_0, origin_voxel_it.second, voxel_eloid.center, 
                                        origin_voxel_it.second.coordinate, origin_voxel_it.second.length);
                if(seg_voxel_.empty())
                {
                    seg_voxel_ = project_pt(cur_vector_1, origin_voxel_it.second, voxel_eloid.center, 
                                            origin_voxel_it.second.coordinate, origin_voxel_it.second.length);                    
                }
                // if(seg_voxel_.empty())
                // {
                //     seg_voxel_ = project_pt(cur_vector_2, origin_voxel_it.second, voxel_eloid.center, 
                //                             origin_voxel_it.second.coordinate, origin_voxel_it.second.length);     
                // }
                if(seg_voxel_.empty())
                {
                    seg_state = false;
                }

            }else if(voxel_eloid.mode == 3)
            {
                // cout << "[EL]  Adaptive Segment  voxel mode is: stereoscopic" << endl;
                seg_state = false;
            }
        }else{
            // cout << "[EL]  Adaptive Segment  voxel point num is not enough" << endl;
            seg_state = false;
        }


        if(seg_state == false)
        {
            // cout << "[EL]  Adaptive Segment  no segment voxel" << endl;
            frame_voxel.back().seg_voxel_data.push_back(origin_voxel_it.second);

        }else{
            // cout << "[EL]  Adaptive Segment  successfully segment voxel" << endl;
            for(auto seg_voxel_data_it : seg_voxel_)
            {
                frame_voxel.back().seg_voxel_data.push_back(seg_voxel_data_it);
            }
        }

    }
}

//自适应体素划分 以单个体素为单位 不在z方向上划分
//输入： 当前体素的特征方向 体素点云 当前体素点云均值 当前体素坐标系 当前体素尺寸
//返回：填入分割后的体素的存储容器 若无法分割 则返回空容器
std::vector<VoxelData>  
BaseGlobalLocalization::project_pt(Eigen::Matrix<double, 1, 3> cur_vector, VoxelData voxel_data, 
                                    Eigen::Vector3d cur_leaf_mean, Eigen::Vector3d cur_leaf_coordinate, Eigen::Vector3d cur_leaf_length)
{
    vector<Eigen::Vector3d> cur_leaf_point = voxel_data.point_cloud;
    // 降维，将点映射到特征值对应的方向
    std::vector<double> cur_project_points; //映射后的点云的存储容器
    cur_project_points.clear();
    std::map<double, int> projectPt_index;
    projectPt_index.clear();
    for(int i = 0; i < voxel_data.point_cloud.size(); ++i)
    {
      Eigen::Matrix<double, 3, 1> de_mean_point(cur_leaf_point[i][0] - cur_leaf_mean[0],
                                                cur_leaf_point[i][1] - cur_leaf_mean[1],
                                                cur_leaf_point[i][2] - cur_leaf_mean[2]);
      auto projectPt = cur_vector * de_mean_point;    //求出映射的坐标
      projectPt_index[projectPt] = i;
      cur_project_points.emplace_back(projectPt);
    }
    sort(cur_project_points.begin(), cur_project_points.end()); // 升序排列
    
    double last_project_points = 0;
    double D_project_points = 0; // 映射到特征值对应方向上的点间的距离
    Eigen::Vector3d seg_lidar_point;
    int overlap_x = 0, overlap_y = 0, overlap_z = 0;
    bool seg_x = false, seg_y = false, seg_z = false;
    //遍历映射后的点
    for (std::vector<double>::iterator ite1 = cur_project_points.begin(); ite1 != cur_project_points.end(); ite1++)
    {
        seg_lidar_point = Eigen::Vector3d::Zero();
        if(ite1 == cur_project_points.begin())
        {
          last_project_points = *ite1;
          continue;
        }
        D_project_points = fabs(*ite1 - last_project_points);
        //两点映射后的距离大于判定阈值
        if(D_project_points > DIVIDE_MIN_PROJECT_DISTANCE_FROM) // 阈值待定
        {
            int last_index = projectPt_index.find(last_project_points)->second;
            int cur_index = projectPt_index.find(*ite1)->second;
            //暂定两点中心为分割点
            seg_lidar_point.x() = (cur_leaf_point[last_index][0] + cur_leaf_point[cur_index][0]) / 2;
            seg_lidar_point.y() = (cur_leaf_point[last_index][1] + cur_leaf_point[cur_index][1]) / 2;
            seg_lidar_point.z() = (cur_leaf_point[last_index][2] + cur_leaf_point[cur_index][2]) / 2;
            // 上一个映射点相对于均值的坐标
            // double last_Project2Mean_x = sin(x_angle) * last_project_points;
            // double last_Project2Mean_y = sin(y_angle) * last_project_points;
            // double last_Project2Mean_z = sin(z_angle) * last_project_points;
            // // 上一个映射点相对于雷达系的坐标
            // double last_Project2Lidar_x = last_Project2Mean_x + cur_leaf_mean[0];
            // double last_Project2Lidar_y = last_Project2Mean_y + cur_leaf_mean[1];
            // double last_Project2Lidar_z = last_Project2Mean_z + cur_leaf_mean[2];

            // // 当前映射点相对于均值的坐标
            // double cur_Project2Mean_x = sin(x_angle) * (*ite1);
            // double cur_Project2Mean_y = sin(y_angle) * (*ite1);
            // double cur_Project2Mean_z = sin(z_angle) * (*ite1);
            // // 当前映射点相对于雷达系的坐标
            // double cur_Project2Lidar_x = cur_Project2Mean_x + cur_leaf_mean[0];
            // double cur_Project2Lidar_y = cur_Project2Mean_y + cur_leaf_mean[1];
            // double cur_Project2Lidar_z = cur_Project2Mean_z + cur_leaf_mean[2];

            // // 待分割点相对于雷达坐标系的坐标
            // seg_lidar_point.x() = (cur_Project2Lidar_x + last_Project2Lidar_x) / 2.0;
            // seg_lidar_point.y() = (cur_Project2Lidar_y + last_Project2Lidar_y) / 2.0;
            // seg_lidar_point.z() = (cur_Project2Lidar_z + last_Project2Lidar_z) / 2.0;
            // std::cout<<"cur_point:"<<cur_Project2Lidar_x<<","<<cur_Project2Lidar_y<<","<<cur_Project2Lidar_z<<std::endl;
            // std::cout<<"last_point:"<<last_Project2Lidar_z<<","<<last_Project2Lidar_y<<","<<last_Project2Lidar_x<<std::endl;

            //这里通过差值判断中心点是否在leaf的中心，或者在某一侧的中心
            Eigen::Vector3d top1_ = seg_lidar_point - cur_leaf_coordinate;
            Eigen::Vector3d top2(cur_leaf_coordinate.x()+cur_leaf_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z());
            Eigen::Vector3d top2_ = seg_lidar_point - top2;
            Eigen::Vector3d top3(cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z()+cur_leaf_length.z());
            Eigen::Vector3d top3_ = seg_lidar_point - top3;
            Eigen::Vector3d top4(cur_leaf_coordinate.x(), cur_leaf_coordinate.y()+cur_leaf_length.y(), cur_leaf_coordinate.z());
            Eigen::Vector3d top4_ = seg_lidar_point - top4;
            Eigen::Vector3d top5(cur_leaf_coordinate.x()+cur_leaf_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z()+cur_leaf_length.z());
            Eigen::Vector3d top5_ = seg_lidar_point - top5;
            Eigen::Vector3d top6(cur_leaf_coordinate.x(), cur_leaf_coordinate.y()+cur_leaf_length.y(), cur_leaf_coordinate.z()+cur_leaf_length.z());
            Eigen::Vector3d top6_ = seg_lidar_point - top6;
            Eigen::Vector3d top7(cur_leaf_coordinate.x()+cur_leaf_length.x(), cur_leaf_coordinate.y()+cur_leaf_length.y(), cur_leaf_coordinate.z());
            Eigen::Vector3d top7_ = seg_lidar_point - top7;
            Eigen::Vector3d top8(cur_leaf_coordinate.x()+cur_leaf_length.x(), cur_leaf_coordinate.y()+cur_leaf_length.y(), cur_leaf_coordinate.z()+cur_leaf_length.z());
            Eigen::Vector3d top8_ = seg_lidar_point - top8;
            
            //与边界距离判断，距离小于阈值不进行分割
            if(    (fabs(top1_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top1_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top1_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                || (fabs(top2_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top2_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top2_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                || (fabs(top3_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top3_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top3_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                || (fabs(top4_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top4_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top4_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                || (fabs(top5_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top5_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top5_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                || (fabs(top6_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top6_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top6_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                || (fabs(top7_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top7_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top7_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                || (fabs(top8_.x()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top8_.y()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY && fabs(top8_.z()) <= VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY))
            // if((top1_.x()<=0.01 &&top1_.y()<=0.01 &&top1_.z()<=0.01) || (top2_.x()<=0.01 &&top2_.y()<=0.01 &&top2_.z()<=0.01) || (top3_.x()<=0.01 &&top3_.y()<=0.01 &&top3_.z()<=0.01) || (top4_.x()<=0.01 &&top4_.y()<=0.01 &&top4_.z()<=0.01) || (top5_.x()<=0.01 &&top5.y()<=0.01 &top5_.z()<=0.01) || (top6_.x()<=0.01 &&top6_.y()<=0.01 &&top6_.z()<=0.01) || (top7_.x()<=0.01 &&top7_.y()<=0.01 &&top7_.z()<=0.01) || (top8_.x()<=0.01 &&top8_.y()<=0.01 &&top8_.z()<=0.01))
            {
                last_project_points = *ite1;
                continue;
            }

            //遍历计算 原始点在xyz方向上与分割点的垂直距离
            for (size_t i = 0; i < cur_leaf_point.size(); i++)
            {
                if(fabs(cur_leaf_point[i][0] - seg_lidar_point.x()) < MAX_DIVIDE_LINE_BOUNDARY_DISTANCE) overlap_x++;
                if(fabs(cur_leaf_point[i][1] - seg_lidar_point.y()) < MAX_DIVIDE_LINE_BOUNDARY_DISTANCE) overlap_y++;
                if(fabs(cur_leaf_point[i][2] - seg_lidar_point.z()) < MAX_DIVIDE_LINE_BOUNDARY_DISTANCE) overlap_z++;
            }

            if(overlap_x < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z < DIVIDE_LINE_BOUNDARY_MAX_NUM)
            {
                // std::cout<<"segement in three direction"<<std::endl;
                // return subdivision(seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);
                // std::cout<<"segement in 2 direction"<<std::endl;
                seg_x = true; seg_y = true; seg_z = false;
                return subdivision_2(seg_x, seg_y, seg_z, seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);
            }
            else if (overlap_x < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z >= DIVIDE_LINE_BOUNDARY_MAX_NUM)
            {
                // std::cout<<"segement in x and y direction"<<std::endl;
                seg_x = true; seg_y = true; seg_z = false;
                return subdivision_2(seg_x, seg_y, seg_z, seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);
            }
            // else if (overlap_x < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z < DIVIDE_LINE_BOUNDARY_MAX_NUM)
            // {
            //     std::cout<<"segement in x and z direction"<<std::endl;
            //     seg_x = true; seg_y = false; seg_z = true;
            //     return subdivision_2(seg_x, seg_y, seg_z, seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);;
            // }
            // else if (overlap_x >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z < DIVIDE_LINE_BOUNDARY_MAX_NUM)
            // {
            //     std::cout<<"segement in y and z direction"<<std::endl;
            //     seg_x = false; seg_y = true; seg_z = true;
            //     return subdivision_2(seg_x, seg_y, seg_z, seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);
            // }
            else if (overlap_x < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z >= DIVIDE_LINE_BOUNDARY_MAX_NUM)
            {   
                //这里再次判断一次 需要同时满足不在两个侧面边界
                if (seg_lidar_point.x() - cur_leaf_coordinate.x() < VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY || seg_lidar_point.x() - (cur_leaf_coordinate.x() + cur_leaf_length.x()) < VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY) 
                {
                    // cout << "[EL]  Project Point  only x   segment point boundary near" << endl;
                    continue;
                }
                // std::cout<<"segement in x direction"<<std::endl;
                seg_x = true; seg_y = false; seg_z = false;
                return subdivision_1(seg_x, seg_y, seg_z, seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);
            }
            else if (overlap_x >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y < DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z >= DIVIDE_LINE_BOUNDARY_MAX_NUM)
            {
                if (seg_lidar_point.y() - cur_leaf_coordinate.y() < VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY || seg_lidar_point.y() - (cur_leaf_coordinate.y() + cur_leaf_length.y()) < VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY)
                {
                    // cout << "[EL]  Project Point  only y   segment point boundary near" << endl;
                    continue;
                }
                // std::cout<<"segement in y direction"<<std::endl;
                seg_x = false; seg_y = true; seg_z = false;
                return subdivision_1(seg_x, seg_y, seg_z, seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);

            }
            // else if (overlap_x >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z < DIVIDE_LINE_BOUNDARY_MAX_NUM)
            // {
            //     if (seg_lidar_point.z() - cur_leaf_coordinate.z() < VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY || seg_lidar_point.z() - (cur_leaf_coordinate.z() + cur_leaf_length.z()) < VALID_DIVIDE_POINT_DISTANCE_FROM_BOUNDARY)
            //     {
            //         cout << "[EL]  Project Point  only z   segment point boundary near" << endl;
            //         continue;
            //     }
            //     std::cout<<"segement in z direction"<<std::endl;
            //     seg_x = false; seg_y = false; seg_z = true;
            //     return subdivision_1(seg_x, seg_y, seg_z, seg_lidar_point, cur_leaf_point, cur_leaf_coordinate, cur_leaf_length);
            // }
            else if (overlap_x >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_y >= DIVIDE_LINE_BOUNDARY_MAX_NUM && overlap_z >= DIVIDE_LINE_BOUNDARY_MAX_NUM)
            {
                // cout << "[EL]  Project Point  no segment point  segment line so much point nearly" << endl;
                return std::vector<VoxelData>();
            }
        }
        last_project_points = *ite1;
    }
    
    // cout << "[EL]  Project Point  no segment point  segment point boundary near" << endl;
    return std::vector<VoxelData>();
}


std::vector<VoxelData>  
BaseGlobalLocalization::subdivision_1(bool seg_x, bool seg_y, bool seg_z, Eigen::Vector3d seg_lidar_point, std::vector<Eigen::Vector3d> cur_leaf_point, Eigen::Vector3d cur_leaf_coordinate, Eigen::Vector3d cur_leaf_length)
{
    int index = 0;
    Eigen::Vector3d first_length(0, 0, 0);
    first_length = seg_lidar_point - cur_leaf_coordinate;
    std::vector<VoxelData> seg_voxel_;
    VoxelData voxel_data_0, voxel_data_1;
    for (size_t i = 0; i < cur_leaf_point.size(); i++)
    {   
        if (seg_x)
        {
            if (cur_leaf_point[i].x() <= seg_lidar_point.x())
            {
                voxel_data_0.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_0.is_sege_success = true;  
                voxel_data_0.coordinate = cur_leaf_coordinate;
                voxel_data_0.length << first_length.x(), cur_leaf_length.y(), cur_leaf_length.z();

            }else if (cur_leaf_point[i].x() > seg_lidar_point.x())
            {
                voxel_data_1.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_1.is_sege_success = true;  
                voxel_data_1.coordinate << cur_leaf_coordinate.x() + first_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
                voxel_data_1.length << cur_leaf_length.x() - first_length.x(), cur_leaf_length.y(), cur_leaf_length.z();

            }
        }else if (seg_y)
        {
            if (cur_leaf_point[i].y() <= seg_lidar_point.y())
            {
                voxel_data_0.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_0.is_sege_success = true;  
                voxel_data_0.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
                voxel_data_0.length << cur_leaf_length.x(), first_length.y(), cur_leaf_length.z();

            }else if (cur_leaf_point[i].y() > seg_lidar_point.y())
            {
                voxel_data_1.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_1.is_sege_success = true;  
                voxel_data_1.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y() + first_length.y(), cur_leaf_coordinate.z();
                voxel_data_1.length << cur_leaf_length.x(), cur_leaf_length.y() - first_length.y(), cur_leaf_length.z();

            }
        }else if (seg_z)
        {
            if (cur_leaf_point[i].z() <= seg_lidar_point.z())
            {
                voxel_data_0.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_0.is_sege_success = true;  
                voxel_data_0.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
                voxel_data_0.length << cur_leaf_length.x(), cur_leaf_length.y(), first_length.z();

            }else if (cur_leaf_point[i].z() > seg_lidar_point.z())
            {
                voxel_data_1.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_1.is_sege_success = true;  
                voxel_data_1.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z() + first_length.z();
                voxel_data_1.length << cur_leaf_length.x(), cur_leaf_length.y(), cur_leaf_length.z() - first_length.z();

            }
        }
    }

    seg_voxel_.push_back(voxel_data_0);
    seg_voxel_.push_back(voxel_data_1);

    // cout << "[EL]  Segment Voxel  finish divide 1 coordinate" << endl;

    return seg_voxel_;

}

std::vector<VoxelData> 
BaseGlobalLocalization::subdivision_2(bool seg_x, bool seg_y, bool seg_z, Eigen::Vector3d seg_lidar_point, std::vector<Eigen::Vector3d> cur_leaf_point,Eigen::Vector3d cur_leaf_coordinate,Eigen::Vector3d cur_leaf_length)
{
    int index = 0;
    Eigen::Vector3d first_length(0, 0, 0);
    first_length = seg_lidar_point - cur_leaf_coordinate;
    std::vector<VoxelData> seg_voxel_;
    VoxelData voxel_data_0, voxel_data_1, voxel_data_2, voxel_data_3;
    for (size_t i = 0; i < cur_leaf_point.size(); i++)
    {
        if (seg_x && seg_y)
        {
            if (cur_leaf_point[i].x() <= seg_lidar_point.x() && cur_leaf_point[i].y() <= seg_lidar_point.y())
            {
                voxel_data_0.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_0.is_sege_success = true;  
                voxel_data_0.coordinate = cur_leaf_coordinate;
                voxel_data_0.length << first_length.x(), first_length.y(), cur_leaf_length.z();
            }else if (cur_leaf_point[i].x() > seg_lidar_point.x() && cur_leaf_point[i].y() > seg_lidar_point.y())
            {
                voxel_data_1.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_1.is_sege_success = true;  
                voxel_data_1.coordinate << cur_leaf_coordinate.x() + first_length.x(), cur_leaf_coordinate.y() + first_length.y(), cur_leaf_coordinate.z();
                voxel_data_1.length << cur_leaf_length.x() - first_length.x(), cur_leaf_length.y() - first_length.y(), cur_leaf_length.z();

            }else if (cur_leaf_point[i].x() <= seg_lidar_point.x() && cur_leaf_point[i].y() > seg_lidar_point.y())
            {
                voxel_data_2.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_2.is_sege_success = true;  
                voxel_data_2.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y() + first_length.y(), cur_leaf_coordinate.z();
                voxel_data_2.length << first_length.x(), cur_leaf_length.y() - first_length.y(), cur_leaf_length.z();

            }else if (cur_leaf_point[i].x() > seg_lidar_point.x() && cur_leaf_point[i].y() <= seg_lidar_point.y())
            {
                voxel_data_3.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_3.is_sege_success = true;
                voxel_data_3.coordinate << cur_leaf_coordinate.x() + first_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
                voxel_data_3.length << cur_leaf_length.x() - first_length.x(), first_length.y(), cur_leaf_length.z();

            }
        }else if (seg_x && seg_z)
        {
            if (cur_leaf_point[i].x() <= seg_lidar_point.x() && cur_leaf_point[i].z() <= seg_lidar_point.z())
            {
                voxel_data_0.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_0.is_sege_success = true; 
                voxel_data_0.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
                voxel_data_0.length << first_length.x(), cur_leaf_length.y(), first_length.z();

            }else if (cur_leaf_point[i].x() <= seg_lidar_point.x() && cur_leaf_point[i].z() > seg_lidar_point.z())
            {
                voxel_data_1.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_1.is_sege_success = true;  
                voxel_data_1.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z() + first_length.z();
                voxel_data_1.length << first_length.x(), cur_leaf_length.y(), cur_leaf_length.z() - first_length.z();

            }else if (cur_leaf_point[i].x() > seg_lidar_point.x() && cur_leaf_point[i].z() <= seg_lidar_point.z())
            {
                voxel_data_2.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_2.is_sege_success = true;  
                voxel_data_2.coordinate << cur_leaf_coordinate.x() + first_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
                voxel_data_2.length << cur_leaf_length.x() - first_length.x(), cur_leaf_length.y(), first_length.z();

            }else if (cur_leaf_point[i].x() > seg_lidar_point.x() && cur_leaf_point[i].z() > seg_lidar_point.z())
            {
                voxel_data_3.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_3.is_sege_success = true;  
                voxel_data_3.coordinate << cur_leaf_coordinate.x() + first_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z() + first_length.z();
                voxel_data_3.length << cur_leaf_length.x() - first_length.x(), cur_leaf_length.y(), cur_leaf_length.z() - first_length.z();

            }
        }else if (seg_y && seg_z)
        {
            if (cur_leaf_point[i].y() <= seg_lidar_point.y() && cur_leaf_point[i].z() <= seg_lidar_point.z())
            {
                voxel_data_0.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_0.is_sege_success = true;  
                voxel_data_0.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
                voxel_data_0.length << cur_leaf_length.x(), first_length.y(), first_length.z();

            }else if (cur_leaf_point[i].y() <= seg_lidar_point.y() && cur_leaf_point[i].z() > seg_lidar_point.z())
            {
                voxel_data_1.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_1.is_sege_success = true;  
                voxel_data_1.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z() + first_length.z();
                voxel_data_1.length << cur_leaf_length.x(), first_length.y(), cur_leaf_length.z() - first_length.z();

            }else if (cur_leaf_point[i].y() > seg_lidar_point.y() && cur_leaf_point[i].z() <= seg_lidar_point.z())
            {
                voxel_data_2.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_2.is_sege_success = true;  
                voxel_data_2.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y() + first_length.y(), cur_leaf_coordinate.z();
                voxel_data_2.length << cur_leaf_length.x(), cur_leaf_length.y() - first_length.y(), first_length.z();

            }else if (cur_leaf_point[i].y() > seg_lidar_point.y() && cur_leaf_point[i].z() > seg_lidar_point.z())
            {
                voxel_data_3.point_cloud.push_back(cur_leaf_point[i]);
                voxel_data_3.is_sege_success = true;  
                voxel_data_3.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y() + first_length.y(), cur_leaf_coordinate.z() + first_length.z();
                voxel_data_3.length << cur_leaf_length.x(), cur_leaf_length.y() - first_length.y(), cur_leaf_length.z() - first_length.z();
            }
        }
    }

    seg_voxel_.push_back(voxel_data_0);
    seg_voxel_.push_back(voxel_data_1);
    seg_voxel_.push_back(voxel_data_2);
    seg_voxel_.push_back(voxel_data_3);

    // cout << "[EL]  Segment Voxel  finish divide 2 coordinate" << endl;

    return seg_voxel_;

}

std::vector<VoxelData> 
BaseGlobalLocalization::subdivision(Eigen::Vector3d seg_lidar_point, std::vector<Eigen::Vector3d> cur_leaf_point,Eigen::Vector3d cur_leaf_coordinate,Eigen::Vector3d cur_leaf_length)
{
    int index = 0;
    Eigen::Vector3d first_length(0, 0, 0);
    first_length = seg_lidar_point - cur_leaf_coordinate;
    std::vector<VoxelData> seg_voxel_;
    VoxelData voxel_data_0, voxel_data_1, voxel_data_2, voxel_data_3, voxel_data_4, voxel_data_5, voxel_data_6, voxel_data_7;
    for (size_t j = 0; j < cur_leaf_point.size(); j++)
    {
        if (cur_leaf_point[j].x() <= seg_lidar_point.x() && cur_leaf_point[j].y() <= seg_lidar_point.y() && cur_leaf_point[j].z() <= seg_lidar_point.z())
        {
            voxel_data_0.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_0.coordinate = cur_leaf_coordinate;
            voxel_data_0.length = first_length;
            voxel_data_0.is_sege_success = true;  
        }else if(cur_leaf_point[j].x() <= seg_lidar_point.x() && cur_leaf_point[j].y() <= seg_lidar_point.y() && cur_leaf_point[j].z() > seg_lidar_point.z())
        {
            voxel_data_1.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_1.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z()+first_length.z();
            voxel_data_1.length << first_length.x(), first_length.y(), cur_leaf_length.z() - first_length.z();
            voxel_data_1.is_sege_success = true; 
        }else if (cur_leaf_point[j].x() <= seg_lidar_point.x() && cur_leaf_point[j].y() > seg_lidar_point.y() && cur_leaf_point[j].z() <= seg_lidar_point.z())
        {
            voxel_data_2.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_2.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y()+first_length.y(), cur_leaf_coordinate.z();
            voxel_data_2.length << first_length.x(), cur_leaf_length.y() - first_length.y(), first_length.z();
            voxel_data_2.is_sege_success = true;
        }else if (cur_leaf_point[j].x() > seg_lidar_point.x() && cur_leaf_point[j].y() <= seg_lidar_point.y() && cur_leaf_point[j].z() <= seg_lidar_point.z())
        {
            voxel_data_3.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_3.coordinate << cur_leaf_coordinate.x()+first_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z();
            voxel_data_3.length << cur_leaf_length.x() - first_length.x(), first_length.y(), first_length.z();
            voxel_data_3.is_sege_success = true;
        }else if (cur_leaf_point[j].x() <= seg_lidar_point.x() && cur_leaf_point[j].y() > seg_lidar_point.y() && cur_leaf_point[j].z() > seg_lidar_point.z())
        {
            voxel_data_4.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_4.coordinate << cur_leaf_coordinate.x(), cur_leaf_coordinate.y()+first_length.y(), cur_leaf_coordinate.z()+first_length.z();
            voxel_data_4.length << first_length.x(), cur_leaf_length.y() - first_length.y(), cur_leaf_length.z() - first_length.z();
            voxel_data_4.is_sege_success = true;
        }else if (cur_leaf_point[j].x() > seg_lidar_point.x() && cur_leaf_point[j].y() <= seg_lidar_point.y() && cur_leaf_point[j].z() > seg_lidar_point.z())
        {
            voxel_data_5.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_5.coordinate << cur_leaf_coordinate.x()+first_length.x(), cur_leaf_coordinate.y(), cur_leaf_coordinate.z()+first_length.z();
            voxel_data_5.length << cur_leaf_length.x() - first_length.x(), first_length.y(), cur_leaf_length.z() - first_length.z();
            voxel_data_5.is_sege_success = true;
        }else if (cur_leaf_point[j].x() > seg_lidar_point.x() && cur_leaf_point[j].y() > seg_lidar_point.y() && cur_leaf_point[j].z() <= seg_lidar_point.z())
        {
            voxel_data_6.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_6.coordinate << cur_leaf_coordinate.x()+first_length.x(), cur_leaf_coordinate.y()+first_length.y(), cur_leaf_coordinate.z();
            voxel_data_6.length << cur_leaf_length.x() - first_length.x(), cur_leaf_length.y() - first_length.y(), first_length.z();
            voxel_data_6.is_sege_success = true;
        }else if (cur_leaf_point[j].x() > seg_lidar_point.x() && cur_leaf_point[j].y() > seg_lidar_point.y() && cur_leaf_point[j].z() > seg_lidar_point.z())
        {
            voxel_data_7.point_cloud.push_back(cur_leaf_point[j]);
            voxel_data_7.coordinate << seg_lidar_point.x(), seg_lidar_point.y(), seg_lidar_point.z();
            voxel_data_7.length << cur_leaf_length.x() - first_length.x(), cur_leaf_length.y() - first_length.y(), cur_leaf_length.z() - first_length.z();
            voxel_data_7.is_sege_success = true;
        }
    }

    seg_voxel_.push_back(voxel_data_0);
    seg_voxel_.push_back(voxel_data_1);
    seg_voxel_.push_back(voxel_data_2);
    seg_voxel_.push_back(voxel_data_3);
    seg_voxel_.push_back(voxel_data_4);
    seg_voxel_.push_back(voxel_data_5);
    seg_voxel_.push_back(voxel_data_6);
    seg_voxel_.push_back(voxel_data_7);

    // cout << "[EL]  Segment Voxel  finish divide 3 coordinate" << endl;

    return seg_voxel_;
}



//体素点云分布情况可视化 概率分布函数 需要分割返回 1 无需分割返回 0
int TestClouDistrubutionVisualization(std::vector<Eigen::Vector3d> leaf_cloud, Eigen::Matrix3d axis_all, Eigen::Vector3d center)
{
    Eigen::Vector3d max_axis = axis_all.col(0);
    vector<double> dist;
    double probability[50] = {0};

    for(auto it : leaf_cloud)
    {
        Eigen::Vector3d leaf_point_shift = it - center;
        // if(leaf_point_shift.norm() == 0 || center.norm() == 0)
            // continue; // don't count this sector pair.
        double distance = max_axis.transpose().dot(leaf_point_shift);
        dist.push_back(distance);
    }

    sort(dist.begin(),dist.end());
    double uint = abs(dist.back() - dist.front()) / 50.0;
    // cout << endl << " " << dist.front() << "  " << dist.back() << "   " << uint << endl;

    //求个点落在对应区间的数量
    for(auto it : dist)
    {
        int index = ceil((it - dist.front()) / uint);
        if (index > 50)
            index = 50;
        probability[index]  += 1;
    }

    //判断是否需要分割
    int max_zero_num = 0;
    for(int i = 0; i < 50; i++)
    {
        if(probability[i] == 0)
        {
            int continue_zero_num = 1;
            for(int j = i + 1; j < 50; j++)
            {
                if(probability[j] == 0)
                {
                    continue_zero_num++;                
                }
                else{
                    if(max_zero_num < continue_zero_num)
                        max_zero_num = continue_zero_num;
                    break;
                }
            }
        }
    }
    if(max_zero_num * uint > 1)
        return 1;
    else
        return 0;
}
