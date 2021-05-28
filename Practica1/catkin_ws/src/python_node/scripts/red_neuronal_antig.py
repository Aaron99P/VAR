#!/usr/bin/env python

import rospy
from std_msgs.msg import String

import roslib
import sys
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model, model_from_json

import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import Image as im

import geometry_msgs.msg
from geometry_msgs.msg import Twist


def cnn_model(nb_classes):
  #
  # Neural Network Structure
  #
  
  activation = "relu"

  input_shape = (32, 32, 3)

  inputs = keras.Input(shape=input_shape)
  
  x = layers.Conv2D(52, (5, 5), activation=activation)(inputs)
  x = layers.MaxPooling2D(pool_size=(4, 4))(x)

  x = layers.Conv2D(124, (5, 5), activation=activation)(x)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)

  x = layers.Flatten()(x)

  x = layers.Dense(120, activation=activation)(x)
  x = layers.Dense(84, activation=activation)(x)
  x = layers.Dense(32, activation=activation)(x)
  

  outputs = layers.Dense(nb_classes, activation='softmax')(x)

  model = keras.Model(inputs=inputs, outputs=outputs)

  return model



class red_neuronal:

  global cmd_vel_pub #publisher para mandar las velocidades al robot

  longitud, altura = 32, 32

  #Modelo para la navegacion
  pesosNavegacion = './pesosNavegacion.h5'
  modelNavegacion = cnn_model(3)
  modelNavegacion.load_weights(pesosNavegacion)

  #Modelo para la deteccion de otro robot
  pesosDeteccion = './pesosDeteccion.h5'
  modelDeteccion = cnn_model(2)
  modelDeteccion.load_weights(pesosDeteccion)
  
  global graph
  graph = tensorflow.get_default_graph()

  def __init__(self):
    #El CVBridge lo utilizaremos para obtener las imagenes OpenCV
    self.bridge = CvBridge()
    #image_sub sera el subscriber mediante el que obtendremos los mensajes (imagenes) desde la camara
    self.image_sub = rospy.Subscriber("robot1/camera/rgb/image_raw", Image, self.callback)
    #cmd_vel_pub sera el publisher mediante el que enviaremos mensajes (velocidades) a la mobile_base del robot
    self.cmd_vel_pub = rospy.Publisher("/robot1/mobile_base/commands/velocity", Twist)
    

  def predictNavegacion(self, f):
    #Redimensionamos la imagen a la que necesita la red
    img = cv2.resize(f, (self.longitud, self.altura))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    #Usamos el metodo predict del modelo
    with graph.as_default():
      self.modelNavegacion._make_predict_function()
      array = self.modelNavegacion.predict(x)
      result = array[0]
    
    #Del array obtenido, la posicion con un
    #valor mas alto sera la categoria que la
    #red predice
    answer = np.argmax(result)
    #La categoria 0 es seguir recto
    if answer == 0:
      print("pred: Adelante")
    #La categoria 1 es girar a la derecha
    elif answer == 1:
      print("pred: Derecha")
    #La categoria 2 es girar a la izquierda 
    elif answer == 2:
      print("pred: Izquierda")

    return answer

  def predictDeteccion(self, f):
    #Redimensionamos la imagen a la que necesita la red
    img = cv2.resize(f, (self.longitud, self.altura))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    #Usamos el metodo predict del modelo
    with graph.as_default():
      self.modelDeteccion._make_predict_function()
      array = self.modelDeteccion.predict(x)
      result = array[0]
    
    #Del array obtenido, la posicion con un
    #valor mas alto sera la categoria que la
    #red predice
    answer = np.argmax(result)
    #La categoria 0 es que no ve ningun robot
    if answer == 0:
      print("pred: No veo un Robot")
    #La categoria 1 es que ve un robot
    elif answer == 1:
      print("pred: VEO UN ROBOT")

    return answer


  def velocidad(self, pred):
    #Este es el tipo de mensaje que debe recibir el robot
    cmd = geometry_msgs.msg.Twist()
    cmd.linear.x = cmd.angular.z = 0

    #Si el robot debe seguir recto hacia adelante
    if pred == 0:
      cmd.linear.x = 0.5
    #Si el robot tiene que girar a la derecha
    elif pred == 1:
      cmd.angular.z = -0.75
      cmd.linear.x = 0.25
    #Si el robot tiene que girar a la izquierda
    elif pred == 2:
      cmd.angular.z = 0.75
      cmd.linear.x = 0.25

    return cmd
      

  #Metodo asociado al subscriber image_sub de la camara
  def callback(self, image_message):
    try:
      #Obtenemos la imagen
      cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
      cv2.waitKey(3)

      #La red decide que movimiento hacer
      pred = self.predictNavegacion(cv_image)

      #Calculamos la velocidad que vamos a enviar al robot
      base_cmd = self.velocidad(pred)

      #Enviamos la velocidad
      self.cmd_vel_pub.publish(base_cmd)

      #La red de deteccion dice si hay o no un robot
      det = self.predictDeteccion(cv_image)

    except CvBridgeError as e:
      print(e)


def main(args):
  red = red_neuronal()
  rospy.init_node('red_neuronal', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  main(sys.argv)
