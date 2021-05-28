# VAR


# AUTORES

Aarón Picó Pascual
Samuel Arévalo Cañestro


# ENTREGA

Se adjunta en la entrega el código realizado.

Para obtener el proyecto completo:

Repositorio en Github:
	https://github.com/Aaron99P/VAR

Descarga el repositorio en formato zip:
	https://drive.google.com/file/d/1jVMMdpUY1pSR1gtlNb3PCzRYqj6Mxl_m

Point Cloud tiene su propio leeme dentro de su directorio


# VERSIONES

Versiones utilizadas en ROS:
  - Python: 2.7
  - Tensorflow: 1.12.0
  - Keras: 2.2.4


# Comandos para poner en marcha el robot:

-----------------------------------------------------------------------------

1er terminal:

roscore


-----------------------------------------------------------------------------

2º terminal:

cd catkin_ws

source devel/setup.bash

roslaunch ejemplogazebo create_multi_robot.launch

-----------------------------------------------------------------------------

3er terminal:

cd catkin_ws

source devel/setup.bash

rosrun python_node red_neuronal.py

-----------------------------------------------------------------------------




# Comandos para obtener las imágenes para el dataset:

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



Para imágenes de ir adelante:

cd images/adelante

rosrun image_view image_saver image:=robot1/camera/rgb/image_raw _save_all_image:=false __name:=image_adelante



Para imágenes de ir a la derecha:

cd images/derecha

rosrun image_view image_saver image:=robot1/camera/rgb/image_raw _save_all_image:=false __name:=image_derecha



Para imágenes de ir a la izquierda:

cd images/izquierda

rosrun image_view image_saver image:=robot1/camera/rgb/image_raw _save_all_image:=false __name:=image_izquierda




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
