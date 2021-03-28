# VAR


#Comandos:

-----------------------------------------------------------------------------

1er terminal:

roscore


-----------------------------------------------------------------------------

2º terminal:

cd catkin_ws

source devel/setup.bash

roslaunch ejemplogazebo create_multi_robot.launch

-----------------------------------------------------------------------------

3er 4º y 5º terminal:

cd catkin_ws

source devel/setup.bash

cd images

***************

cd adelante

rosrun image_view image_saver image:=robot1/camera/rgb/image_raw _save_all_image:=false __name:=image_adelante

***************

cd derecha

rosrun image_view image_saver image:=robot1/camera/rgb/image_raw _save_all_image:=false __name:=image_derecha

***************

cd izquierda

rosrun image_view image_saver image:=robot1/camera/rgb/image_raw _save_all_image:=false __name:=image_izquierda

***************


-----------------------------------------------------------------------------

6º terminal:

cd catkin_ws

source devel/setup.bash

rosrun listener listener

-----------------------------------------------------------------------------

7º terminal:

cd catkin_ws

source devel/setup.bash

rosrun send_velocity_commands send_velocity_commands

-----------------------------------------------------------------------------
