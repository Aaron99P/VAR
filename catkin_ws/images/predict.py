import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura = 32, 32
modelo = './modelo.h5'
pesos_modelo = './pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Adelante")
  elif answer == 1:
    print("pred: Derecha")
  elif answer == 2:
    print("pred: Izquierda")

  return answer

def main():
  print("Adelante:")
  for i in range(30):
    if(i<10): 
      print(predict("./adelante/left000"+str(i)+".jpg"))
    else:
      print(predict("./adelante/left00"+str(i)+".jpg"))

  print()
  print("Izquierda:")
  for i in range(30):
    if(i<10):
      print(predict("./izquierda/left000"+str(i)+".jpg"))
    else:
      print(predict("./izquierda/left00"+str(i)+".jpg"))

  print()
  print("Derecha:")
  for i in range(30):
    if(i<10):
      print(predict("./derecha/left000"+str(i)+".jpg"))
    else:
      print(predict("./derecha/left00"+str(i)+".jpg"))

  return 0
