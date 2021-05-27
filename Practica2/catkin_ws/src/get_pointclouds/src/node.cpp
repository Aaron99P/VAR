#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>


pcl::PointCloud<pcl::PointXYZRGB>::Ptr visu_pc (new pcl::PointCloud<pcl::PointXYZRGB>);

pcl::PointCloud<pcl::PointWithScale>::Ptr anterior_keypoints;
pcl::PointCloud<pcl::FPFHSignature33>::Ptr anterior_features;
bool primero = true;

void simpleVis ()
{
  	pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
	while(!viewer.wasStopped())
	{
	  viewer.showCloud (visu_pc);
	  boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
	}

}

pcl::PointCloud<pcl::PointNormal>::Ptr detectorCaracteristicas(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud){

	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> normalEst; //detector de características
 	normalEst.setInputCloud(cloud);

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZRGB>()); //método de búsqueda utilizado por el descriptor
  	normalEst.setSearchMethod(tree_n);
  	normalEst.setRadiusSearch(0.03);

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_detect(new pcl::PointCloud<pcl::PointNormal>); // será el resultado
  	normalEst.compute(*cloud_detect);

	return cloud_detect;
}


pcl::PointCloud<pcl::PointWithScale>::Ptr calculateKeyPoints(pcl::PointCloud<pcl::PointNormal>::Ptr cloud_detect){

	// Parameters for sift computation
  	const float min_scale = 0.1f;
 	const int n_octaves = 6;
 	const int n_scales_per_octave = 10;
  	const float min_contrast = 0.5f;
  
  
	// Estimate the sift interest points using Intensity values from RGB values
	pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
	pcl::PointCloud<pcl::PointWithScale>::Ptr result;
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
	sift.setSearchMethod(tree);
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);
	sift.setMinimumContrast(min_contrast);
	sift.setInputCloud(cloud_detect);
	sift.compute(*result);

	return result;

}


pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptorCarateristicas(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered, pcl::PointCloud<pcl::PointNormal>::Ptr cloud_detect,
										pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints){

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::copyPointCloud (*keypoints, *keypoints_cloud);

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr result;


	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
	pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
  	fpfh.setInputCloud (keypoints_cloud);
  	fpfh.setInputNormals (cloud_detect);
	fpfh.setSearchSurface (cloud_filtered);
	fpfh.setSearchMethod(tree);
  	fpfh.setRadiusSearch (0.07);
  	fpfh.compute (*result);

	return result;
}

void emparejar(){}

void ransac(){}

void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& msg)
{
	// Codigo inicial
	//---------------------------------------------------------------------------------------------------
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>(*msg));
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

	cout << "Puntos capturados: " << cloud->size() << endl;

	pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;
	vGrid.setInputCloud (cloud);
	vGrid.setLeafSize (0.03f, 0.03f, 0.03f);
	vGrid.filter (*cloud_filtered);

	cout << "Puntos tras VG: " << cloud_filtered->size() << endl;

	//visu_pc = cloud_filtered;
	//---------------------------------------------------------------------------------------------------

	//Detector de características
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_detect = detectorCaracteristicas(cloud_filtered);

	//Hallamos los key points
	pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints = calculateKeyPoints(cloud_detect);

	//Descriptor de características
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_features = descriptorCarateristicas(cloud_filtered, cloud_detect, keypoints);


	if(!primero){



	}

	*anterior_keypoints = *keypoints;
	*anterior_features = *cloud_features;

	primero = false;

}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "sub_pcl");
	ros::NodeHandle nh;
	ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZRGB> >("/camera/depth/points", 1, callback);

	boost::thread t(simpleVis);

	while(ros::ok()){
		ros::spinOnce();
	}

}
