# VAR


#Comandos:

-------------------
1er terminal:

roscore
-------------------

-------------------
2º terminal:

cd catkin_ws

source devel/setup.bash

roslaunch ejemplogazebo create_multi_robot.launch
-------------------


-------------------
3er terminal:

cd catkin_ws

source devel/setup.bash

cd images

rosrun image_view image_saver image:=robot1/camera/rgb/image_raw _save_all_image:=false __name:=image_saver
-------------------

-------------------
4º terminal:

cd catkin_ws

source devel/setup.bash

rosrun listener listener
-------------------

-------------------
5º terminal:

cd catkin_ws

source devel/setup.bash

rosrun send_velocity_commands send_velocity_commands
-------------------
