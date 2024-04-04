/**************************************************************************

Author: SHR

Date:2024.3.25

FileName: main.cpp

Function:main

**************************************************************************/

#include <iostream>          //基本输入输出库
#include <string>            //文件流库
#include <thread>            //线程
#include <pcl/point_types.h> //点云类型库
#include <pcl/io/pcd_io.h>   //pcl输入输出库
#include <pcl/io/ply_io.h>
// #include <pcl/filters/gaussian_filter.h>
// 可视化库
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h> //可视化

// 滤波库
#include <pcl/filters/passthrough.h>                 //直通滤波库
#include <pcl/filters/voxel_grid.h>                  //体素滤波库
#include <pcl/filters/statistical_outlier_removal.h> //离群值滤波库
#include <pcl/filters/radius_outlier_removal.h>      //离群值滤波器
#include <pcl/ModelCoefficients.h>                   //模型库
#include <pcl/filters/project_inliers.h>             //内点库

// 分割库
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>                       // for PointCloud
#include <pcl/common/io.h>                         // for copyPointCloud
#include <pcl/sample_consensus/ransac.h>           //随机一致性方法
#include <pcl/sample_consensus/sac_model_plane.h>  //平面模型
#include <pcl/sample_consensus/sac_model_sphere.h> //球体模型
#include <pcl/filters/extract_indices.h>           //内点
#include <pcl/features/normal_3d.h>                //法向量
#include <pcl/search/kdtree.h>                     //kd树
// 提炼模型库
#include <pcl/sample_consensus/method_types.h> //搜索方式
#include <pcl/sample_consensus/model_types.h>  //模型类别
#include <pcl/segmentation/sac_segmentation.h> //分割
// 欧几里得聚类分割
#include <pcl/segmentation/extract_clusters.h> //聚类分割
#include <iomanip>                             //io流对象 for setw, setfill
// 三维物体匹配
#include <pcl/correspondence.h>         //匹配库
#include <pcl/features/normal_3d_omp.h> //omp库
#include <pcl/features/shot_omp.h>      //shot库
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
// 凸包凹包库
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>

using namespace std;
using namespace pcl;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;

int view_4_cloud = 0;
int view_corr_lines = 0;

// 体素滤波 小立方体内所有点云以一个点表示
void voxelfilter(float size, const pcl::PointCloud<PointT>::Ptr &inputcloud,
                 const pcl::PointCloud<PointT>::Ptr &outputcloud);
// 该滤波器根据每个点与其邻居之间的平均距离来工作，并去除那些其平均距离超过一定阈值的点。
void SORfilter(int num, float threshoud, const pcl::PointCloud<PointT>::Ptr &inputcloud,
               const pcl::PointCloud<PointT>::Ptr &outputcloud);
// 该滤波器基于每个点在其一定范围内（即搜索半径内）的邻近点数量进行过滤。用户设定一个搜索半径和一个邻近点数量的阈值
// ，如果一个点在其搜索半径内的邻近点数量少于这个阈值，那么这个点就被认为是离群点并被删除。
void RORfilter(int num, float radius, const pcl::PointCloud<PointT>::Ptr &inputcloud,
               const pcl::PointCloud<PointT>::Ptr &outputcloud);
// 分割出平面和其他点云模型
void user_seg_plane(double instance, const pcl::PointCloud<PointT>::Ptr &inputcloud,
                    const pcl::PointCloud<PointT>::Ptr &inliers_cloud,
                    const pcl::PointCloud<PointT>::Ptr &outiers_cloud);
// 法向量提取 使用点数量，输入点云，法向量点云
void normal_vector_estimate(int num, const pcl::PointCloud<PointT>::Ptr &inputcloud,
                            const pcl::PointCloud<NormalType>::Ptr &normalcloud);
// 将模型点云进行旋转变换，到场景点云的对应位置
void findCorrespondencesAndSave(const pcl::PointCloud<PointT>::Ptr &model_rotated,
                                const pcl::PointCloud<PointT>::Ptr &scene_keypoints,
                                const pcl::PointCloud<PointT>::Ptr &correspondencesCloud,
                                const std::string &output_filename);
// 计算凸包并返回多边形、体积和面积
int computeConvexHull(const pcl::PointCloud<PointT>::Ptr &cloud,
                      const pcl::PointCloud<PointT>::Ptr &cloud_hull,
                      std::vector<pcl::Vertices> &polygons);
// 聚类分割

// 分割出模型点云

int main(int argc, char **argv)
{
    // 模型
    pcl::PointCloud<PointT>::Ptr model(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr model_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr model_convex(new pcl::PointCloud<PointT>);
    // 点云 过滤后点云
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered1(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloud_outiers(new pcl::PointCloud<PointT>);
    // 目标点云
    pcl::PointCloud<PointT>::Ptr target(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr target_convex(new pcl::PointCloud<PointT>);
    // 特征点
    pcl::PointCloud<PointT>::Ptr model_keypoints(new pcl::PointCloud<PointT>()); // 模型特征点
    pcl::PointCloud<PointT>::Ptr scene_keypoints(new pcl::PointCloud<PointT>()); // 场景特征点
    // 特征向量
    pcl::PointCloud<NormalType>::Ptr model_normals(new pcl::PointCloud<NormalType>()); // 模型特征向量
    pcl::PointCloud<NormalType>::Ptr scene_normals(new pcl::PointCloud<NormalType>()); // 场景特征向量
    // 描述符
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>());
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors(new pcl::PointCloud<DescriptorType>());
    // 索引
    std::vector<pcl::Vertices> polygons; // 保存的是凸包多边形的顶点坐标

    // pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGBA>);
    // 导入点云 milk_cartoon_all_small_clorox     table_scene_lms400
    pcl::io::loadPCDFile("../../data/milk_cartoon_all_small_clorox.pcd", *cloud);
    // pcl::io::loadPCDFile("../../data/milk_cartoon_all_small_clorox.pcd", *cloud_filtered2);
    // pcl::io::loadPCDFile("../../data/milk_filtered_cloud.pcd", *model_filtered);
    pcl::io::loadPCDFile("../../data/milk.pcd", *model);

    // 滤波 减少点云数量 减轻计算压力
    // voxelfilter(0.002f, cloud, cloud_filtered1);
    // RORfilter(10, 0.03f, cloud_filtered1, cloud_filtered2);
    // user_seg_plane(0.02, cloud_filtered2, cloud_plane, cloud_outiers);
    // 平面分割 使用分割后点云进行匹配
    voxelfilter(0.002f, cloud, cloud_filtered1);                       // 体素滤波 设定体积内以一点代表
    RORfilter(10, 0.03f, cloud_filtered1, cloud_outiers);              // ROR滤波去除噪声值
    user_seg_plane(0.02, cloud_outiers, cloud_plane, cloud_filtered2); // 分割出平面，使用非平面匹配

    // RORfilter(3, 0.01f, cloud, cloud_filtered2);
    // pcl::io::savePCDFileASCII("../../data/milk_filtered_cloud.pcd", *cloud_filtered2);
    // 法向量提取
    normal_vector_estimate(10, cloud, scene_normals); // 场景法向量提取，10个点或者0.03m
    normal_vector_estimate(10, model, model_normals); // 模型法向量提取，10个点或者0.03m

    // 2 对每个云进行降采样以找到少量关键点 简化计算
    pcl::UniformSampling<PointT> uniform_sampling; // 均匀降采样
    uniform_sampling.setInputCloud(model);         // 输入点云
    uniform_sampling.setRadiusSearch(0.01f);       // 搜索半径
    uniform_sampling.filter(*model_keypoints);     // 存储降采样后的点云
    std::cout << "Model total points: " << model->size() << "; Selected Keypoints: " << model_keypoints->size() << std::endl;

    uniform_sampling.setInputCloud(cloud_filtered2); // 输入点云
    uniform_sampling.setRadiusSearch(0.03f);         // 搜索半径
    uniform_sampling.filter(*scene_keypoints);       // 降采样后关键点
    std::cout << "Scene total points: " << cloud_filtered2->size() << "; Selected Keypoints: " << scene_keypoints->size() << std::endl;

    // 3 将关键点关联到SHOT特征描述符以执行模型aa和场景的关键点匹配并确定点对点对应关系。
    pcl::SHOTEstimationOMP<PointT, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch(0.02f); // 搜索半径

    descr_est.setInputCloud(model_keypoints); // 输入模型关键点点云
    descr_est.setInputNormals(model_normals); // 输入模型法向量
    descr_est.setSearchSurface(model);        // 输入模型初始点云
    descr_est.compute(*model_descriptors);    // 储存描述子

    descr_est.setInputCloud(scene_keypoints); // 输入场景点云
    descr_est.setInputNormals(scene_normals); // 场景法向量
    descr_est.setSearchSurface(cloud);        // 场景初始点云
    descr_est.compute(*scene_descriptors);    // 场景描述子

    std::cout << "model_descriptors total points: " << model_descriptors->size() << std::endl;
    std::cout << "scene_descriptors total points: " << scene_descriptors->size() << std::endl;
    // 基于欧氏距离找最相似的描述符，确定模型描述符和场景描述符之间的点对点对应关系
    pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

    // 使用KdTreeFLANN，将模型点云的描述子与场景点云的描述子进行匹配，并将匹配的对应关系存储在model_scene_corrs中
    // 匹配的准则是找到场景描述子的最近邻，并判断其平方距离小于0.25
    pcl::KdTreeFLANN<DescriptorType> match_search;
    match_search.setInputCloud(model_descriptors); // 输入模型描述子

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (std::size_t i = 0; i < scene_descriptors->size(); ++i) // 遍历场景描述子
    {
        std::vector<int> neigh_indices(1);                          // 初始化最近索引
        std::vector<float> neigh_sqr_dists(1);                      // 初始化最近平方距离
        if (!std::isfinite(scene_descriptors->at(i).descriptor[0])) // skipping NaNs跳过非有限描述子
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch(scene_descriptors->at(i), 1, neigh_indices, neigh_sqr_dists); // 搜索最近邻
        if (found_neighs == 1 && neigh_sqr_dists[0] < 0.25f)                                                         //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            pcl::Correspondence corr(neigh_indices[0], static_cast<int>(i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back(corr); // 将corr添加到model_scene_corrs中，表示模型与场景之间的对应关系
        }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;

    // 根据描述符的对应关系进行聚类。
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    pcl::GeometricConsistencyGrouping<PointT, PointT> gc_clusterer;
    gc_clusterer.setGCSize(0.01f);      // 设置几何一致性聚类尺寸
    gc_clusterer.setGCThreshold(15.0f); // 设置最小簇大小
    gc_clusterer.setInputCloud(model_keypoints);
    gc_clusterer.setSceneCloud(scene_keypoints);
    gc_clusterer.setModelSceneCorrespondences(model_scene_corrs); // 执行几何一致性聚类，将结果存储在clustered_corrs中
    gc_clusterer.recognize(rototranslations, clustered_corrs);    // 对聚类结果进行识别，将旋转-平移矩阵存在在rototranslations中

    std::cout << "Model instances found: " << rototranslations.size() << std::endl;
    std::cout << "cluster size: " << clustered_corrs.size() << std::endl;

    // 输出聚类结果 保存聚类分割出来的点云
    for (std::size_t i = 0; i < rototranslations.size(); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

        // Print the rotation matrix and translation vector
        Eigen::Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3, 1>(0, 3);

        printf("\n");
        printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
        printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
        printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
        printf("\n");
        printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));

        pcl::PointCloud<PointT>::Ptr model_rotated(new pcl::PointCloud<PointT>);
        // 得到旋转后的点云
        pcl::transformPointCloud(*model, *model_rotated, rototranslations[i]); // 0
        // for (std::size_t i = 0; i < rototranslations.size(); ++i)
        // {
        //     pcl::PointCloud<PointT>::Ptr rotated_model(new pcl::PointCloud<PointT>());
        //     pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);
        // }
        // 匹配 并将匹配结果保存
        findCorrespondencesAndSave(model_rotated, cloud_filtered2, target, "corrs.pcd");
        // pcl::io::savePCDFileASCII("corrs.pcd", *target);
        std::cout << "target size: " << target->size() << std::endl;
        // 目标点云计算体积
        computeConvexHull(target, target_convex, polygons);
        // 计算模型点云体积
        computeConvexHull(model, model_convex, polygons);
        // 比较
    }

    // 定义对象
    pcl::visualization::PCLVisualizer viewer;

    pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_filtered2);
    viewer.addPointCloud(cloud_filtered2, rgb, "scene_cloud");
    // 旋转点云
    pcl::visualization::PointCloudColorHandlerCustom<PointT> rgb2(target, 255, 0, 0);
    viewer.addPointCloud(target, rgb2, "off_scene_model");

    if (view_corr_lines)
    {
        pcl::PointCloud<PointT>::Ptr off_scene_model(new pcl::PointCloud<PointT>());
        pcl::PointCloud<PointT>::Ptr off_scene_model_keypoints(new pcl::PointCloud<PointT>());
        // 模型点云显示到可视化窗口中
        pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
        pcl::transformPointCloud(*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

        pcl::visualization::PointCloudColorHandlerCustom<PointT> off_scene_model_color_handler(off_scene_model, 255, 255, 128);
        viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");

        // 添加特征描述匹配线条
        for (std::size_t i = 0; i < rototranslations.size(); ++i)
        {
            pcl::PointCloud<PointT>::Ptr rotated_model(new pcl::PointCloud<PointT>());
            pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);

            std::stringstream ss_cloud;
            ss_cloud << "instance" << i;

            pcl::visualization::PointCloudColorHandlerCustom<PointT> rotated_model_color_handler(rotated_model, 255, 0, 0);
            viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());

            for (std::size_t j = 0; j < clustered_corrs[i].size(); ++j)
            {
                std::stringstream ss_line;
                ss_line << "correspondence_line" << i << "_" << j;
                PointT &model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
                PointT &scene_point = scene_keypoints->at(clustered_corrs[i][j].index_match);

                //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
                viewer.addLine<PointT, PointT>(model_point, scene_point, 0, 255, 0, ss_line.str());
            }
        }
    }

    if (view_4_cloud)
    {
        int v1(1);
        int v2(2);
        int v3(3);
        int v4(4);
        // createViewPort 参数是 double xmin, double ymin, double xmax, double ymax, int &viewport
        viewer.createViewPort(0, 0, 0.5, 0.5, v1);
        viewer.createViewPort(0.5, 0, 1, 0.5, v2);
        viewer.createViewPort(0, 0.5, 0.5, 1, v3);
        viewer.createViewPort(0.5, 0.5, 1, 1, v4);

        // --- 显示点云数据 ----
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb1(cloud_filtered1);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb2(cloud_filtered2);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb3(cloud_plane);
        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb4(cloud_outiers);

        // 将点云模型添加进可视化窗口
        viewer.addPointCloud(cloud_filtered1, rgb1, "voxel", v1);
        viewer.addPointCloud(cloud_filtered2, rgb2, "ror", v2);
        viewer.addPointCloud(cloud_plane, rgb3, "plane", v3);
        viewer.addPointCloud(cloud_outiers, rgb4, "outiers", v4);
    }

    // 开始显示2种方法,任选其一
    // 1. 阻塞式
    viewer.spin();
    return 0;
}

void voxelfilter(float size, const pcl::PointCloud<PointT>::Ptr &inputcloud, const pcl::PointCloud<PointT>::Ptr &outputcloud)
{
    pcl::VoxelGrid<PointT> sor; // 设置滤波器
    sor.setInputCloud(inputcloud);
    sor.setLeafSize(size, size, size); // 设置1m3的立方体
    sor.filter(*outputcloud);          // 保存滤波后点云

    // 计算点云数量进行比较
    std::cerr << "PointCloud before voxelfiltering: " << inputcloud->width * inputcloud->height
              << " data points (" << pcl::getFieldsList(*inputcloud) << ")." << std::endl;
    std::cerr << "PointCloud after voxelfiltering: " << outputcloud->width * outputcloud->height
              << " data points (" << pcl::getFieldsList(*outputcloud) << ")." << std::endl;
}

void SORfilter(int num, float threshoud, const pcl::PointCloud<PointT>::Ptr &inputcloud,
               const pcl::PointCloud<PointT>::Ptr &outputcloud)
{
    std::cerr << "Cloud before filtering: " << std::endl;
    std::cerr << *inputcloud << std::endl;

    // 创建一个pcl::StatisticalOutlierRemoval滤波器。
    // 所有距离查询点的平均距离的标准差大于1的点都将被标记为异常值并被删除。
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(inputcloud);
    sor.setMeanK(num);                 // 每个查询点要分析的邻居数量设置为 越大鲁棒性越好 算力要求越高
    sor.setStddevMulThresh(threshoud); // 标准差阈值设置为 越小滤越多
    sor.filter(*outputcloud);          // 计算输出并将其存储在cloud_filtered中。

    // 计算点云数量进行比较
    std::cerr << "PointCloud before SORiltering: " << inputcloud->width * inputcloud->height
              << " data points (" << pcl::getFieldsList(*inputcloud) << ")." << std::endl;
    std::cerr << "PointCloud after SORfiltering: " << outputcloud->width * outputcloud->height
              << " data points (" << pcl::getFieldsList(*outputcloud) << ")." << std::endl;

    // std::cerr << "Cloud after filtering: " << std::endl;
    // std::cerr << *outputcloud << std::endl;
    // // 保存滤波后的点云：
    // pcl::PCDWriter writer;
    // writer.write<PointT>("../../data/sor_inliers.pcd", *outputcloud, false);

    // // 保存异常点：
    // sor.setNegative(true);
    // sor.filter(*outputcloud);
    // writer.write<PointT>("../../data/sor_outliers.pcd", *outputcloud, false);
}

void RORfilter(int num, float radius, const pcl::PointCloud<PointT>::Ptr &inputcloud,
               const pcl::PointCloud<PointT>::Ptr &outputcloud)
{

    // 创建半径离群点去除滤波器对象
    pcl::RadiusOutlierRemoval<PointT> outrem;
    // 设置输入点云
    outrem.setInputCloud(inputcloud);
    // 设置搜索半径
    outrem.setRadiusSearch(radius);
    // 设置最小邻居点数阈值
    outrem.setMinNeighborsInRadius(num); // 执行滤波操作
    outrem.filter(*outputcloud);

    std::cerr << "PointCloud before RORfiltering: " << inputcloud->width * inputcloud->height
              << " data points (" << inputcloud->points.size() << ")." << std::endl;
    std::cerr << "PointCloud after RORfiltering: " << outputcloud->width * outputcloud->height
              << " data points (" << outputcloud->points.size() << ")." << std::endl;

    // 保存滤波后的点云到文件
    // pcl::io::savePCDFileASCII("../../data/RORfiltered_cloud.pcd", *outputcloud);
}

// 平面模型分离
void user_seg_plane(double instance, const pcl::PointCloud<PointT>::Ptr &inputcloud,
                    const pcl::PointCloud<PointT>::Ptr &inliers_cloud,
                    const pcl::PointCloud<PointT>::Ptr &outiers_cloud)
{
    // 从点云模型提炼对象

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients); // 模型参数
    pcl::PointIndices::Ptr inliers2(new pcl::PointIndices);               // 内点

    // 创建RANSAC对象并设置参数
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);     // 最小二乘法优化模型系数
    seg.setModelType(pcl::SACMODEL_PLANE); // 平面
    seg.setMethodType(pcl::SAC_RANSAC);    // RANSAC方法
    seg.setMaxIterations(1000);            // 最大迭代次数
    seg.setDistanceThreshold(instance);    // 阈值
    seg.setInputCloud(inputcloud);         // 输入点云

    // 估计平面模型参数
    seg.segment(*inliers2, *coefficients); // 分离
    // 检查是否成功找到了模型
    if (inliers2->indices.size() == 0)
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
    // 提取平面上的内点

    // 保存去除的平面Extract the planar inliers from the input cloud
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(inputcloud);
    extract.setIndices(inliers2);
    extract.setNegative(false); // 保留平面
    // Get the points associated with the planar surface
    extract.filter(*inliers_cloud);
    std::cout << "PointCloud representing the planar component: "
              << inliers_cloud->size() << " data points." << std::endl;

    // 除去平面模型的其他点云模型Remove the planar inliers, extract the rest
    extract.setNegative(true); // 保留除平面的模型
    extract.filter(*outiers_cloud);

    // 打印模型系数
    std::cout << "Model coefficients: " << coefficients->values[0] << " "
              << coefficients->values[1] << " "
              << coefficients->values[2] << " "
              << coefficients->values[3] << std::endl;
}

void normal_vector_estimate(int num, const pcl::PointCloud<PointT>::Ptr &inputcloud,
                            const pcl::PointCloud<NormalType>::Ptr &normalcloud)
{
    pcl::NormalEstimationOMP<PointT, NormalType> normal_estimation;         // 创建法向量估计对象
    normal_estimation.setInputCloud(inputcloud);                            // 点云输入
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>); // 创建一个空的kdtree
    normal_estimation.setSearchMethod(tree);                                // 将其传递给法线估计对象。

    normal_estimation.setRadiusSearch(0.03); // 利用半径为3cm的球体中的所有邻点进行估计
    // normal_estimation.setKSearch(num); // 使用当前点周围最近的10个点

    // 进行法线估计并保存
    normal_estimation.compute(*normalcloud);
    // pcl::io::savePCDFileASCII("../../data/normalvcs_cloud.pcd", *normalcloud);
}

void findCorrespondencesAndSave(const pcl::PointCloud<PointT>::Ptr &model_rotated,
                                const pcl::PointCloud<PointT>::Ptr &scene_keypoints,
                                const pcl::PointCloud<PointT>::Ptr &correspondencesCloud,
                                const std::string &output_filename)
{
    // 创建KD树对象，并与场景点云关联
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(scene_keypoints);

    // 存储找到的对应点
    std::vector<PointT> correspondences;

    // 遍历模型点云中的每个点
    for (const auto &modelPoint : *model_rotated)
    {
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);
        // 在场景点云中查找最近邻点
        if (tree->nearestKSearch(modelPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            // 获取最近邻点的索引
            int idx = pointIdxNKNSearch[0];
            // 从场景点云中获取对应的点并存储
            correspondences.push_back(scene_keypoints->points[idx]);
        }
    }

    std::cout << "correspondences size is " << correspondences.size() << std::endl;

    correspondencesCloud->width = static_cast<uint32_t>(correspondences.size());
    correspondencesCloud->height = 1;      // 表示点云是无组织的
    correspondencesCloud->is_dense = true; // 假设没有无效点
    correspondencesCloud->points.resize(correspondences.size());

    // 将向量中的数据复制到点云中
    for (size_t i = 0; i < correspondences.size(); ++i)
    {
        correspondencesCloud->points[i] = correspondences[i];
    }

    std::cout << "correspondencesCloud size is " << correspondencesCloud->size() << std::endl;

    // 保存PCD文件
    // pcl::io::savePCDFileASCII(output_filename, *correspondencesCloud);
}

// 计算凸包并返回多边形、体积和面积
int computeConvexHull(const pcl::PointCloud<PointT>::Ptr &cloud,
                      const pcl::PointCloud<PointT>::Ptr &cloud_hull,
                      std::vector<pcl::Vertices> &polygons)
{

    float volume;
    float area;
    // 创建凸包对象
    pcl::ConvexHull<PointT> convex_Hull;
    // 设置输入点云
    convex_Hull.setInputCloud(cloud);
    // 设置维度为3D
    convex_Hull.setDimension(3);
    // 设置计算面积和体积
    convex_Hull.setComputeAreaVolume(true);

    pcl::PointCloud<PointT>::Ptr cloud2(new pcl::PointCloud<PointT>);
    // 重建凸包
    convex_Hull.reconstruct(*cloud_hull, polygons);
    // *cloud_hull = *cloud2;

    // 获取体积和面积
    volume = convex_Hull.getTotalVolume();
    area = convex_Hull.getTotalArea();

    // 输出结果
    std::cout << "体积: " << volume << std::endl;
    std::cout << "面积: " << area << std::endl;

    return true;
}