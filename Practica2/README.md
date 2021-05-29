
En catkin_ws:

	- En todos los terminales: source devel/setup.bash

	- Para compilar -> catkin_make
	
	- Para lanzar el simulador -> roslaunch load_model init.launch

	- Para lanzar el programa que registra las nuebes de puntos -> rosrun get_pointclouds get_pointclouds


Se captura una nube de puntos cada vez que se mueve el robot (por teclado).


Ahora mismo se está utilizando:
	
	- NormalEstimation para obtener las normales 
	- SIFTKeypoint para obtener los key points
	- FPFHEstimation para las características
	- CorrespondenceEstimation para encontrar correspondencias
	- CorrespondenceRejectorSampleConsensus (RANSAC) para descartar malas correspondencias

Enlaces a los videos demostrativos de los resultados:

Mapeado 3D: https://drive.google.com/file/d/19beIEhvxH-KHfd2X5Nz-ZICPyEm4N37r/view?usp=sharing

Seguimiento de línea: https://drive.google.com/file/d/1T59kCfgOwZZbcglDp5JbmnGBFOv4xCoh/view?usp=sharing
