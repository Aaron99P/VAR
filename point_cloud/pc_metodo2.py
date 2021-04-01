
from __future__ import print_function
import cv2
import numpy as np
import glob
import sys


imagedir = str(sys.argv[1])+ '*.jpg'

camara1 = 'trasera1'
camara2 = 'trasera2'
calib1 = './camera_calibration/calibration_result/'+camara1+'/'
calib2 = './camera_calibration/calibration_result/'+camara2+'/'
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




def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def rectifyImage(originalImage, calibPath):

    mtx = np.load(calibPath+'array_mtx.npy')
    dist = np.load(calibPath+'array_dist.npy')
    h, w = 480, 640
    print('h: '+ str(h) + ' w: ' + str(w)  )
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(originalImage, mtx, dist, None, newcameramtx)
    
    # crop the image
    x,y,w,h = roi
    dst = dst[10:470, 10:630]
    

    return dst
    
if __name__ == '__main__': 
    
    images = glob.glob(imagedir)

    imgL = cv2.imread(images[0])
    imgR = cv2.imread(images[1])



    imgL = rectifyImage(imgL, './camera_calibration/calibration_result/'+camara1+'/')
    imgR = rectifyImage(imgR, './camera_calibration/calibration_result/'+camara2+'/')

    # Calculamos disparity
    window_size = 4
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
        uniquenessRatio = 4,
        speckleWindowSize = 128,
        speckleRange = 2

    )

    print('Calculando disparidad ...')

    print('h1: '+ str(imgL.shape[0]) + ' w1: ' + str(imgL.shape[1])  )
    print('h2: '+ str(imgR.shape[0]) + ' w2: ' + str(imgR.shape[1])  )

    disp = stereo.compute(imgL,imgR).astype(np.float32) /16.0

    print('Generando nube de puntos 3d ...')

    h, w = imgL.shape[:2]

    k = np.float32([[554.38271, 0.0, 320.50000],[0.0, 554.38271, 240.50000],[0.0, 0.0, 1.0]])
    f = k[0]
    Q = np.float32([
        [1,     0, 0,-320.50000], 
        [0,    -1, 0, 240.50000], 
        [0,     0, 0, -554.38271],
        [0,     0, 1,   0]])

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'pointcloud_out'+images[0].split('/')[2]+'.ply'
    create_output(out_points, out_colors,out_fn)
    print('Saved output file : \''+out_fn+'\'')

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

