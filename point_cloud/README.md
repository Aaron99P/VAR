
CALIBRACIÓN:

    Para calibrar las camaras: 

    python2 ./camera_calibration/calibration.py <NOMBRE_CAMARA> <carpeta de origen para imagenes de camara> <Carpeta de salida de calibración>


    EJEMPLO PARA trasera1::

    python2 ./camera_calibration/calibration.py trasera1 ./camera_calibration/calibration_images/ ./camera_calibration/calibration_result/


Creacion de "nubes de puntos":

    METODO 1: 
		python2 pc_metodo1.py <ruta a la carpeta que contiene las 2 fotos>
    METODO 2 (con rectificación de imagen): 
		python2 pc_metodo2.py <ruta a la carpeta que contiene las 2 fotos>

Para ejecutar todas las nubes de puntos de manera rapida se pueden usar los scripts sh:
    sh pcm1.sh
Y
    sh pcm2.sh
