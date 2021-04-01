
from __future__ import print_function
import cv2
import numpy as np
import glob
import sys


imagedir = str(sys.argv[1])+ '*.jpg'

camara1 = 'trasera1'
camara2 = 'trasera2'
## Funcion para guardar ficheros ply

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



if __name__ == '__main__': 
    
    images = glob.glob(imagedir)

    imgL = cv2.imread(images[0])
    imgR = cv2.imread(images[1])

    # Calculamos disparity
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(

        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('Calculando disparidad ...')
    disp = stereo.compute(imgL,imgR).astype(np.float32) /16.0

    print('Generando nube de puntos 3d ...')

    h, w = imgL.shape[:2]

    k = np.float32([[555.66493124, 0.0, 319.30727507],[0.0, 555.53295629, 239.24593941],[0.0, 0.0, 1.0]])
    f = k[0]
    Q = np.float32([
        [1,     0, 0,-320], 
        [0,    -1, 0, 240], 
        [0,     0, 0, -555.66493124],
        [0,     0, 1,   0]])

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'pointcloud_out'+images[0].split('/')[2]+'.ply'
    create_output(out_points, out_colors,out_fn)
    print('Saved output file : \''+out_fn+'\'')

