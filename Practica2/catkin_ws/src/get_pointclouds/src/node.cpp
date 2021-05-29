#include <ros/ros.h>
#include <boost/foreach.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl_ros/point_cloud.h>
#include <geometry_msgs/Twist.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>




ros::Publisher cmd_vel_pub_;


//Función para la neavegación del robot (por teclado)
void driveKeyboard() {

	std::cout << "Type a command and then press enter.  "
	"Use 'w' to move forward, 'a' to turn left, "
	"'d' to turn right, '.' to exit.\n";

	//we will be sending commands of type "twist"
	geometry_msgs::Twist base_cmd;

	char cmd[50];
	std::cin.getline(cmd, 50);

	if(cmd[0]!='w' && cmd[0]!='a' && cmd[0]!='d' && cmd[0]!='s' && cmd[0]!='.') std::cout << "unknown command:" << cmd << "\n";

	base_cmd.linear.x = base_cmd.linear.y = base_cmd.angular.z = 0;   
	
	//move forward
	if(cmd[0]=='w'){
		base_cmd.linear.x = 0.3;//0.25;
	} 

	//turn left (yaw) and drive forward at the same time
	else if(cmd[0]=='a'){
		base_cmd.angular.z = 0.35;
		//base_cmd.linear.x = 0.25;
	
	} 

	//turn right (yaw) and drive forward at the same time
	else if(cmd[0]=='d'){
		base_cmd.angular.z = -0.35;
		//base_cmd.linear.x = 0.25;
	
	}

	//turn right (yaw) and drive forward at the same time
	else if(cmd[0]=='s'){
		base_cmd.linear.x = -0.3;
	
	} 

	//publish the assembled command
	cmd_vel_pub_.publish(base_cmd);   

}


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

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>); // será el resultado
  	normalEst.compute(*cloud_normals);

	return cloud_normals;
}


pcl::PointCloud<pcl::PointWithScale>::Ptr calculateKeyPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_filtered, pcl::PointCloud<pcl::PointNormal>::Ptr& cloud_normals){

	for(size_t i = 0; i<cloud_normals->points.size(); ++i)
  	{
    	cloud_normals->points[i].x = cloud_filtered->points[i].x;
    	cloud_normals->points[i].y = cloud_filtered->points[i].y;
    	cloud_normals->points[i].z = cloud_filtered->points[i].z;
  	}

	/*
	// Parameters for sift computation
  	const float min_scale = 0.1f;
 	const int n_octaves = 6;
 	const int n_scales_per_octave = 10;
  	const float min_contrast = 0.5f;
	*/

	// Parameters for sift computation
  	const float min_scale = 0.02f;
 	const int n_octaves = 3;
 	const int n_scales_per_octave = 4;
  	const float min_contrast = 0.001f;
  
  
	// Estimate the sift interest points using Intensity values from RGB values
	pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
	pcl::PointCloud<pcl::PointWithScale>::Ptr result(new pcl::PointCloud<pcl::PointWithScale> ());
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal> ());
	sift.setSearchMethod(tree);
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);
	sift.setMinimumContrast(min_contrast);
	sift.setInputCloud(cloud_normals);
	//sift.setRadiusSearch(0.05);
	sift.compute(*result);
	
	return result;

}


pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptorCarateristicas(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_filtered, pcl::PointCloud<pcl::PointNormal>::Ptr& cloud_normals,
										pcl::PointCloud<pcl::PointWithScale>::Ptr& keypoints){

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::copyPointCloud(*keypoints, *keypoints_cloud);

	cout << "cloud_filtered Size: " << cloud_filtered->size() << endl;
	cout << "keypoints Size: " << keypoints->size() << endl;
	cout << "keypoints_cloud Size: " << keypoints_cloud->size() << endl;


	pcl::PointCloud<pcl::FPFHSignature33>::Ptr result(new pcl::PointCloud<pcl::FPFHSignature33> ());


	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
	pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
  	fpfh.setInputCloud (keypoints_cloud);
	//fpfh.setInputCloud (cloud_filtered);
  	fpfh.setInputNormals (cloud_normals);
	fpfh.setSearchSurface (cloud_filtered);
	fpfh.setSearchMethod(tree);
  	fpfh.setRadiusSearch (0.07);
  	fpfh.compute (*result);

	return result;
}

void callback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& msg)
{
	// Codigo inicial
	//---------------------------------------------------------------------------------------------------
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>(*msg));
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

	cout << "Puntos capturados: " << cloud->size() << endl;

	pcl::VoxelGrid<pcl::PointXYZRGB > vGrid;
	vGrid.setInputCloud (cloud);
	vGrid.setLeafSize (0.02f, 0.02f, 0.02f);
	vGrid.filter (*cloud_filtered);

	cout << "Puntos tras VG: " << cloud_filtered->size() << endl;

	//visu_pc = cloud_filtered;
	//---------------------------------------------------------------------------------------------------
	
	//Detector de características
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals = detectorCaracteristicas(cloud_filtered);
	
	//Hallamos los key points
	pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints = calculateKeyPoints(cloud_filtered, cloud_normals);
	
	//Descriptor de características
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_features = descriptorCarateristicas(cloud_filtered, cloud_normals, keypoints);
	
	
	if(!primero){

		// Buscamos corespondencias ------------------------------------------------------------------
		pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
		est.setInputSource(anterior_features);
		est.setInputTarget(cloud_features);

		pcl::Correspondences all_correspondences;
		est.determineReciprocalCorrespondences(all_correspondences);
		//--------------------------------------------------------------------------------------------

		// Eliminamos las malas corespondencias ------------------------------------------------------
		pcl::CorrespondencesConstPtr correspondences_p(new pcl::Correspondences(all_correspondences));

		pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointWithScale> ransac;
		ransac.setInputSource(anterior_keypoints);
		ransac.setInputTarget(keypoints);
		ransac.setInlierThreshold(0.1);
		ransac.setMaximumIterations(100000);
		ransac.setRefineModel(true);
		ransac.setInputCorrespondences(correspondences_p); 

		pcl::Correspondences correspondences_out;
		ransac.getCorrespondences(correspondences_out);

		Eigen::Matrix4f transformation = ransac.getBestTransformation();		
		//--------------------------------------------------------------------------------------------


		pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::transformPointCloud(*visu_pc, *transformed_cloud, transformation);

		*visu_pc = *transformed_cloud + *cloud;


	}else{
		visu_pc = cloud_filtered;
	}

	anterior_keypoints = keypoints;
	anterior_features = cloud_features;

	primero = false;

	driveKeyboard();
	
}

/*
int main(int argc, char** argv)
{
  //init the ROS node
  ros::init(argc, argv, "robot_driver");
  ros::NodeHandle nh;

  RobotDriver driver(nh);
  driver.driveKeyboard();
}*/

int main(int argc, char** argv)
{
	ros::init(argc, argv, "sub_pcl");
	ros::NodeHandle nh;
	cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>("/mobile_base/commands/velocity", 1);
	ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZRGB> >("/camera/depth/points", 1, callback);
	visu_pc = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
	boost::thread t(simpleVis);
	
	while(ros::ok()){
		ros::spinOnce();
	}

}