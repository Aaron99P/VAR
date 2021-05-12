
from __future__ import print_function
import cv2
import numpy as np
import glob
import sys

#::Directorios de imagenes empleadas en la creacion de la nube de puntos
imagedir = str(sys.argv[1])+ '*.jpg'
camara1 = 'trasera1'
camara2 = 'trasera2'
calib1 = './camera_calibration/calibration_result/'+camara1+'/'
calib2 = './camera_calibration/calibration_result/'+camara2+'/'

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

#::Funcion dedicada a la rectificacion de una imagen recibida por parametros
def rectifyImage(originalImage, calibPath):

    mtx = np.load(calibPath+'array_mtx.npy')
    dist = np.load(calibPath+'array_dist.npy')
    h, w = 480, 640
    
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    dst = cv2.undistort(originalImage, mtx, dist, None, newcameramtx)
    
    # Se corta la imagen para retirar los bordes producidos al rectificar
    x,y,w,h = roi
    dst = dst[10:470, 10:630]
    
    # Se devuelve la imagen rectificada
    return dst
    
#::Funcion principal del algoritmo
if __name__ == '__main__': 
    print('Comenzando ...')
    images = glob.glob(imagedir)
    # Obtenemos las imagenes desde las que realizarmos la nube de puntos
    imgL = cv2.imread(images[0])
    imgR = cv2.imread(images[1])


    # Usamos la funcion rectifyIamge para rectificar las imagenes
    imgL = rectifyImage(imgL, './camera_calibration/calibration_result/'+camara1+'/')
    imgR = rectifyImage(imgR, './camera_calibration/calibration_result/'+camara2+'/')

    # Preparamos el objeto StereoSGBM de opencv2 con atributos personalizados 
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(
        mode= 3,
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 3,
        P1 = 4*3*window_size**2,
        P2 = 128*3*window_size**2,
        disp12MaxDiff = 4,
        uniquenessRatio = 6,
        speckleWindowSize = 128,
        speckleRange = 2

    )
    
    print('Calculando disparidad ...')
    # Calculamos la disparidad entre las imagenes izquierda y derecha
    disp = stereo.compute(imgL,imgR).astype(np.float32) /16.0

    print('Generando nube de puntos 3d ...')
    
    h, w = imgL.shape[:2]
    # Creamos la nube de puntos a partir de la matriz disp y los valores de calibracion de la camara 
    # Estos valores deben haber sido obtenidos en la calibracion de la camara empleando (camera_calibration/calibration.py) 
    # o mediante otro metodo de obtencion.

    k = np.float32([[554.38271, 0.0, 320.50000],[0.0, 554.38271, 240.50000],[0.0, 0.0, 1.0]])
    f = k[0]
    Q = np.float32([
        [1,     0, 0,-320.50000], 
        [0,    -1, 0, 240.50000], 
        [0,     0, 0, -554.38271],
        [0,     0, 1,   0]])

    # Empleamos las funciones de opencv2 para obtener los vectores de puntos y colores de la nube de puntos
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    # Guardamos la nube de puntos en un fichero de salida ply para su posterior visualizacion mediante MeshLab
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = './pointclouds_out/pointcloud_metodo2_out'+images[0].split('/')[2]+'.ply'
    create_output(out_points, out_colors,out_fn)
    print('Resultado guardado en fichero de salida: \''+out_fn+'\'')


