#include "BaseGlobalLocalization.h"

#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include <numeric>

using namespace std;

//划分体素
std::vector<std::vector<std::vector<Eigen::Vector3d> > > 
    BaseGlobalLocalization::DivideVoxel(pcl::PointCloud<SCPointType> & _scan_cloud)
{
    std::vector<std::vector<std::vector<Eigen::Vector3d> > > uint_point_queue
                                                            (VOXEL_NUM_HORIZONTAL,std::vector<std::vector<Eigen::Vector3d> >(VOXEL_NUM_VERTICAL,std::vector<Eigen::Vector3d>(0)));  //单元内的点的数组
    int num_pts_scan_down = _scan_cloud.points.size();
    for(int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++)
    {
        int horizontal_idx,vertical_idx;
        SCPointType pt;
        pt.x = _scan_cloud.points[pt_idx].x;                                //循环获取帧内各点
        pt.y = _scan_cloud.points[pt_idx].y;
        pt.z = _scan_cloud.points[pt_idx].z + LIDAR_HEIGHT;    

        if( abs(pt.x) > MAX_RANGE || abs(pt.y) > MAX_RANGE )                //判断点距离传感器距离是否超出最大值
            continue;
        
        // cout << "computer bin index" << endl;
        horizontal_idx = floor(double((pt.x + 80.0) / VOXEL_UNIT_HORIZANTAL));
        vertical_idx = floor(double((pt.y + 80.0) / VOXEL_UNIT_VERTICAL));
        
        // cout << "printf ring index: " << ring_idx << " sector index: " << sector_idx << endl;
        Eigen::Vector3d piont_data = {pt.x,pt.y,pt.z};
        uint_point_queue[horizontal_idx][vertical_idx].push_back(piont_data);
    }

    return uint_point_queue;
}

//构建椭球模型 输入划分好体素的点云
void BaseGlobalLocalization::BulidingEllipsoidModel(std::vector<std::vector<std::vector<Eigen::Vector3d> > > voxel_point)
{
    Frame_Ellipsoid _frame_eloid;
    for(auto horizontal_it : voxel_point)
    {
        for(auto voxel_it : horizontal_it)
        {
            Ellipsoid eloid;

            if(voxel_it.size() < 100)                  //不处理少于100个点的体素
            {
                continue;
            }else{
                eloid.point_num = voxel_it.size();
                std::pair<Eigen::MatrixXd, Eigen::MatrixXd> voxel_mean_cov = GetCovarMatrix(voxel_it);
                std::pair<Eigen::Vector3d,Eigen::Matrix3d> voxel_eigen_ = GetEigenvalues(voxel_mean_cov.second);

                eloid.center = voxel_mean_cov.first.row(0);
                eloid.cov = voxel_mean_cov.second;
                eloid.eigen = voxel_eigen_.first;
                eloid.axis = voxel_eigen_.second;
                eloid.eloid_vaild = IsEllipsoidVaild(eloid);
                eloid.mode = ClassifyEllipsoid(eloid);

                //求num_exit
                // cout << "z: ";
                for(auto point_it : voxel_it)
                {
                    int bit_shift = 0;
                    bit_shift = floor(point_it[2] / 0.5) > 8 ? 8 : (floor(point_it[2] / 0.5) < 0 ? 0 : floor(point_it[2] / 0.5));
                    // cout << point_it[2] << ",";
                    if((eloid.num_exit & (1 << bit_shift)) == 0)
                    {
                        // cout << bit_shift << ",";
                        eloid.num_exit |=  1 << bit_shift; 
                    }

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
    }
    frame_eloid.push_back(_frame_eloid);
}



//椭球模型是否有效     返回当前椭球的有效性  -1：椭球不存在 0：椭球无效  1：椭球有效
bool BaseGlobalLocalization::IsEllipsoidVaild(Ellipsoid voxeleloid)
{
    if(voxeleloid.point_num >= 100)
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

//特征值分解 返回值为降序 输入 协方差  输出 特征值-特征向量对
std::pair<Eigen::Vector3d,Eigen::Matrix3d> BaseGlobalLocalization::GetEigenvalues(Eigen::MatrixXd bin_cov)
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
    Eigen::Vector3d eigen_value;
    eigen_value << D(0,0),D(1,1),D(2,2);
    // std::cout << "eigen value: " << eigen_value[0] << " " << eigen_value[1] << " " << eigen_value[2] << endl;

    //将特征值和特征向量用pair形式返回
    std::pair<Eigen::Vector3d,Eigen::Matrix3d> result = {eigen_value,V};

    return result;
}