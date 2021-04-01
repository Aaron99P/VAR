
from __future__ import print_function
import cv2
import numpy as np
import glob
import sys

#::Directorios de imagenes empleadas en la creacion de la nube de puntos
imagedir = str(sys.argv[1])+ '*.jpg'
camara1 = 'trasera1'
camara2 = 'trasera2'

#::Funcion para guardar nubes de puntos en ficheros ply
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')



#::Funcion principal del algoritmo
if __name__ == '__main__': 
    
    images = glob.glob(imagedir)

    # Obtenemos las imagenes desde las que realizarmos la nube de puntos
    imgL = cv2.imread(images[0])
    imgR = cv2.imread(images[1])

    # Preparamos el objeto StereoSGBM de opencv2 con atributos personalizados 
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(
        mode=3,
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 3,
        P1 = 4*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 3,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 16
    )

    print('Calculando disparidad ...')
    # Calculamos la disparidad entre las imagenes izquierda y derecha
    disp = stereo.compute(imgL,imgR).astype(np.float32) /16.0

    print('Generando nube de puntos 3d ...')

    h, w = imgL.shape[:2]
    # Creamos la nube de puntos a partir de la matriz disp y los valores de calibracion de la camara 
    # Estos valores deben haber sido obtenidos en la calibracion de la camara empleando (camera_calibration/calibration.py) 
    # o mediante otro metodo de obtencion.

    k = np.float32([[555.66493124, 0.0, 319.30727507],[0.0, 555.53295629, 239.24593941],[0.0, 0.0, 1.0]])
    f = k[0]
    Q = np.float32([
        [1,     0, 0,-320], 
        [0,    -1, 0, 240], 
        [0,     0, 0, -555.66493124],
        [0,     0, 1,   0]])

    # Empleamos las funciones de opencv2 para obtener los vectores de puntos y colores de la nube de puntos
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    # Guardamos la nube de puntos en un fichero de salida ply para su posterior visualizacion mediante MeshLab
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = './pointclouds_out/pointcloud_metodo1_out'+images[0].split('/')[2]+'.ply'
    create_output(out_points, out_colors,out_fn)
    print('Saved output file : \''+out_fn+'\'')

