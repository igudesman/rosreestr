# USAGE
# python predict.py --image images/cat.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# или python predict.py --image images/1.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --flatten 1
# python predict.py --image images/panda.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64

# импортируем необходимые пакеты
import time
t1=time.time()
from keras.models import load_model
import argparse
import pickle
import cv2
import os
from PIL import Image
import io

t2=time.time()
print("Загрузка модулей: "+str(t2-t1)+" сек")

#os.chdir('images')
img = Image.open('images/cap.jpg')
area1=(26,0,48,37)
img1 = img.crop(area1)
area2=(47,0,67,37)
img2 = img.crop(area2)
area3=(67,0,87,37)
img3 = img.crop(area3)
area4=(85,0,103,37)
img4 = img.crop(area4)
area5=(102,0,123,37)
img5 = img.crop(area5)

img1.save("1"+".jpg")
img2.save("2"+".jpg")
img3.save("3"+".jpg")
img4.save("4"+".jpg")
img5.save("5"+".jpg")


def prescript(file):    
        # создаём парсер аргументов и передаём их
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image",type=str, default=file,help="path to input image we are going to classify")
        ap.add_argument("-m", "--model",type=str,default="simple_nn.model",help="path to trained Keras model")
        ap.add_argument("-l", "--label-bin",type=str,default="simple_nn_lb.pickle",help="path to label binarizer")
        ap.add_argument("-w", "--width", type=int, default=16, help="target spatial dimension width")
        ap.add_argument("-e", "--height", type=int, default=37, help="target spatial dimension height")
        ap.add_argument("-f", "--flatten", type=int, default=1, help="whether or not we should flatten the image")
        args = vars(ap.parse_args())

        # загружаем входное изображение и меняем его размер на необходимый
        image = cv2.imread(file)
        output = image.copy()
        image = cv2.resize(image, (args["width"], args["height"]))

        # масштабируем значения пикселей к диапазону [0, 1]
        image = image.astype("float") / 255.0

        # проверяем, необходимо ли сгладить изображение и добавить размер
        # пакета
        if args["flatten"] > 0:
                image = image.flatten()
                image = image.reshape((1, image.shape[0]))

        # в противном случае мы работаем с CNN -- не сглаживаем изображение
        # и просто добавляем размер пакета
        else:
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # загружаем модель и бинаризатор меток
        #print("[INFO] loading network and label binarizer...")
        model = load_model(args["model"])
        lb = pickle.loads(open(args["label_bin"], "rb").read())

        # делаем предсказание на изображении
        preds = model.predict(image)
        #print(preds)

        # находим индекс метки класса с наибольшей вероятностью
        # соответствия
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]

        # рисуем метку класса + вероятность на выходном изображении
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        print(text[0])
#cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# показываем выходное изображение
#cv2.imshow("Image", output)
#cv2.waitKey(0)

prescript("1.jpg")
prescript("2.jpg")
prescript("3.jpg")
prescript("4.jpg")
prescript("5.jpg")

#os.remove("1.jpg")
#os.remove("2.jpg")
#os.remove("3.jpg")
#os.remove("4.jpg")
#os.remove("5.jpg")

t3=time.time()
print("Решение #капчи: "+str(t3-t2)+" сек")