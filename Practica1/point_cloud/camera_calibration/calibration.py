#!/usr/bin/python
import numpy as np
import cv2
import glob
import sys

print 'Argumentos:', str(sys.argv)
#::Ubicaciones de las imagenes empleadas en la calibracion
camara=str(sys.argv[1])
image_resolve = str(sys.argv[2])+'/'+camara+'/'+camara+'_0001.jpg'
image_path = str(sys.argv[2])+'/'+camara+'/*.jpg'

#::Criterio para finalizacion del algoritmo de calibracion
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparacion de vectores de puntos empleados durante la calibración
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays empleandos apra almacenar puntos de todas las imagenes, de todas las iteraciones.
objpoints = [] # Puntos en el espacio 3d 
imgpoints = [] # Puntos en la imagen 2d 

# Obtencion de las rutas de las imagenes
images = glob.glob(image_path)
# Para todas las imagenes en la ruta realizamos el bucle de obtencion de puntos.
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Se buscan las esquinas de los cuadros de calibracion
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

    # Si se encuentran, se añaden a los vectores de puntos 2d y 3d
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Para cada imagen se dibuja la imagen y por encima las esquinas encontradas.
        img = cv2.drawChessboardCorners(img, (7,7), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
#::Calibramos la camara emplenado la funcion de opencv2 
#   Recibe como argumento los arrays de puntos y devvuelve los valores estimados de la camara
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#::Probaremos los calores de la camara con una imagen
img = cv2.imread(image_resolve)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#::Rectificamos la imagen usando la funcion undistort de opencv2 con las valores de la camara estimados anteriormente.
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Recortamos los bordes 
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(sys.argv[3]+ '/'+camara+'/calibresult.png',dst)
print("Dst:   " ,dst)
print("Ret:   " ,ret)
print("mtx:   " ,mtx)
print("dist:  " ,dist)
print("rvecs: " ,rvecs)
print("tvecs: ", tvecs)
#::Guardamos los vectores de valores de calibración de la camara en ficheros para un uso posterior
# (fix_imports=True para compatiblidad con Python2)
np.save(sys.argv[3]+ '/'+camara+'/array_dst.npy',dst,fix_imports=True)
np.save(sys.argv[3]+ '/'+camara+'/array_mtx.npy',mtx,fix_imports=True)
np.save(sys.argv[3]+ '/'+camara+'/array_ret.npy',ret,fix_imports=True)
np.save(sys.argv[3]+ '/'+camara+'/array_dist.npy',dist,fix_imports=True)
np.save(sys.argv[3]+ '/'+camara+'/array_rvecs.npy',rvecs,fix_imports=True)
np.save(sys.argv[3]+ '/'+camara+'/array_tvecs.npy',tvecs,fix_imports=True)


