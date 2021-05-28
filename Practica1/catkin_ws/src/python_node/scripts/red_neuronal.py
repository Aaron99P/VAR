#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

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
  pesosEsquivar = './pesosEsquivar.h5'
  modelEsquivar = cnn_model(3)
  modelEsquivar.load_weights(pesosEsquivar)

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
    

  def predictNavegacion(self, img):
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

  def predictEsquivar(self, img):
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    #Usamos el metodo predict del modelo
    with graph.as_default():
      self.modelEsquivar._make_predict_function()
      array = self.modelEsquivar.predict(x)
      result = array[0]

    answer = np.argmax(result)
    #La categoria 0 es que no hay que esquivar
    #un robot
    if answer == 0:
      print("pred: NO HAY ESQUIVAR")
    #La categoria 1 es que hay un robot a esquivar
    #a la derecha
    elif answer == 1:
      print("pred: ROBOT A LA DERECHA. ESQUIVANDO")
    #La categoria 2 es que hay un robot a esquivar
    #a la izquierda
    elif answer == 2:
      print("pred: ROBOT A LA IZQUIERDA. ESQUIVANDO")

    return answer

  def predictDeteccion(self, img):
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


  def velocidadNavegacion(self, pred):
    #Este es el tipo de mensaje que debe recibir el robot
    cmd = geometry_msgs.msg.Twist()
    cmd.linear.x = cmd.angular.z = 0

    #Si el robot debe seguir recto hacia adelante
    if pred == 0:
      cmd.linear.x = 0.55
    #Si el robot tiene que girar a la derecha
    elif pred == 1:
      cmd.angular.z = -0.75
      cmd.linear.x = 0.25
    #Si el robot tiene que girar a la izquierda
    elif pred == 2:
      cmd.angular.z = 0.75
      cmd.linear.x = 0.25

    return cmd

  def velocidadEsquivar(self, pred):
    #Este es el tipo de mensaje que debe recibir el robot
    cmd = geometry_msgs.msg.Twist()
    cmd.linear.x = cmd.angular.z = 0

    #Si hay un robot a la derecha
    #Gira a la izquierda
    if pred == 1:
      cmd.angular.z = 0.75
      cmd.linear.x = 0.25

    #Si hay un robot a la izquierda
    #Gira a la derecha
    elif pred == 2:
      cmd.angular.z = -0.75
      cmd.linear.x = 0.25      

    return cmd
      

  #Metodo asociado al subscriber image_sub de la camara
  def callback(self, image_message):
    try:
      #Obtenemos la imagen
      cv_image = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
      #Redimensionamos la imagen a la que necesita la red
      img = cv2.resize(cv_image, (self.longitud, self.altura))

      base_cmd = geometry_msgs.msg.Twist()
      
      #Red para esquivar
      esquivar = self.predictEsquivar(img)

      #Si la red de esquivar no ha visto un robot
      #utilizamos la red de navegacion
      if esquivar == 0:
        #La red de navegacion decide que movimiento hacer
        navegacion = self.predictNavegacion(img)
        #Calculamos la velocidad que vamos a enviar al robot
        base_cmd = self.velocidadNavegacion(navegacion)
      else:
        #Calculamos la velocidad para esquivar
        base_cmd = self.velocidadEsquivar(esquivar)

      #Enviamos la velocidad
      self.cmd_vel_pub.publish(base_cmd)

      #La red de deteccion dice si hay o no un robot
      detectar = self.predictDeteccion(img)

      cv2.waitKey(3)

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
