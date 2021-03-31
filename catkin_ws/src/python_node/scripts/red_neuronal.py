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

import json

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
  #nb_classes = 3

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
    #self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("robot1/camera/rgb/image_raw", Image, self.callback)
    self.cmd_vel_pub = rospy.Publisher("/robot1/mobile_base/commands/velocity", Twist)
    
    #cnn = load_model(self.modelo)
    #with open(self.modelo, "r") as f:
    #  data = json.load(f)
    #cnn = self.cnn_model()
    #self.model.load_weights(self.pesos_modelo)

  def predictNavegacion(self, f):
    img = cv2.resize(f, (self.longitud, self.altura))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #keras.backend.clear_session()
    with graph.as_default():
      self.modelNavegacion._make_predict_function()
      array = self.modelNavegacion.predict(x)
      #print(array)
    
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
      print("pred: Adelante")
    elif answer == 1:
      print("pred: Derecha")
    elif answer == 2:
      print("pred: Izquierda")

    return answer

  def predictDeteccion(self, f):
    img = cv2.resize(f, (self.longitud, self.altura))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #keras.backend.clear_session()
    with graph.as_default():
      self.modelDeteccion._make_predict_function()
      array = self.modelDeteccion.predict(x)
      #print(array)
    
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
      print("pred: No veo un Robot")
    elif answer == 1:
      print("pred: VEO UN ROBOT")

    return answer

  def velocidad(self, pred):
    cmd = geometry_msgs.msg.Twist()
    cmd.linear.x = cmd.angular.z = 0

    if pred == 0:
      cmd.linear.x = 0.5
    elif pred == 1:
      cmd.angular.z = -0.75
      cmd.linear.x = 0.25
    elif pred == 2:
      cmd.angular.z = 0.75
      cmd.linear.x = 0.25

    return cmd
      



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

      #Si detecta un robot lo indicamos por teminal:
      #print("VEO UN ROBOT")

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
