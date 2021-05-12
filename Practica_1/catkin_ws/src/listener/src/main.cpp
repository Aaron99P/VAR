#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try { 
         cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
         cv::waitKey(30);
     }
     catch (cv_bridge::Exception& e) {
         ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
     }
}

void depthCallback(const sensor_msgs::ImageConstPtr& original_image){

    cv_bridge::CvImagePtr cv_ptr;
    //Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    try{

      //Always copy, returning a mutable CvImage
      //OpenCV expects color images to use BGR channel order
      cv_ptr = cv_bridge::toCvCopy(original_image);

    }catch(cv_bridge::Exception& e){

      //if there is an error during conversion, display it
      ROS_ERROR("%s", e.what());
      return;

    }

    //Copy the image.data to imageBuf
    cv::Mat depth_float_img = cv_ptr->image;
    cv::Mat depth_mono8_img;
    
    if(depth_mono8_img.rows != depth_float_img.rows || depth_mono8_img.cols != depth_float_img.cols){
        depth_mono8_img = cv::Mat(depth_float_img.size(), CV_8UC1);
    }
    cv::convertScaleAbs(depth_float_img, depth_mono8_img, 100, 0.0);

    cv::imshow("depth", depth_mono8_img);
    cv::waitKey(30);

}

int main(int argc, char **argv) {
     ros::init(argc, argv, "image_listener");
     ros::NodeHandle nh;
     cv::namedWindow("view");
     cv::namedWindow("depth");
     cv::startWindowThread();
     image_transport::ImageTransport it(nh);
     image_transport::Subscriber sub = it.subscribe("robot1/camera/rgb/image_raw", 1, imageCallback);
     image_transport::Subscriber subdepth = it.subscribe("robot1/camera/depth/image_raw", 1, depthCallback);
     ros::Rate rate(10.0);
     while(nh.ok()) {
       ros::spinOnce();
       rate.sleep();
     }
     ros::spin();
     ros::shutdown();
     cv::destroyWindow("view");
     cv::destroyWindow("depth");
}