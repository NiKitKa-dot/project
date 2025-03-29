from time import sleep
from pioneer_sdk import Camera
from pioneer_sdk import Pioneer
from geobot_sdk import Geobot
import time
import cv2
import threading
import math
import numpy as np



gussian_koef_neh = 51  # коррект значения для большего размытия
speed = 0   #принимает значение от 0 до 1
angel = 0   #от -180 до 180 угол и сторона поворота
res = [[[0] for _ in range(400)] for _ in range(600)]   #тут будет маска с камеры 480 массивов с 640 пикселями
x= [0,0]


#полёт вперёд в зависимости от speed меняется скорость и направление
# 0-0.5 вперёд
# 0.5 никуда
# 0.5 - 1 назад
def forward(speed):
    for _ in range(1000):
        pioneer.send_rc_channels(1500, 1500, 3000*speed, 1500, 2000)
    time.sleep(0.01)

def turn(angel):
    if angel > 0:
        pioneer.send_rc_channels(1500, 0, 1501, 1500, 1500)
    else:
        pioneer.send_rc_channels(1500, 3000, 1600, 1500, 2000)
    time.sleep(0.2)

def data_color_dots():
    global res
    top_dot =  res[20][320]
    for i in range(-1, 1):
        for j in range(-1, 1):
            top_dot+= res[20+i][320+j]
    top_dot = sum(top_dot) #если в top_dot лежит 0 то чёрный


    left_dot = res[160][101]

    for i in range(-1, 1):
        for j in range(-1, 1):
            left_dot += res[160 + i][101 + j]
    left_dot = sum(left_dot)  # если в top_dot лежит 0 то чёрный

    right_dot = res[160][505]

    for i in range(-1, 1):
        for j in range(-1, 1):
            right_dot += res[160 + i][505 + j]
    right_dot = sum(right_dot)  # если в top_dot лежит 0 то чёрный
    print(*[top_dot, left_dot, right_dot])
    return [top_dot, left_dot, right_dot]


def camera_stream(camera: Camera):
    """Получение изображения с камеры Пионера"""
    global res
    while True:
        frame = camera.get_cv_frame()
        if frame is not None:
            cv2.imshow("Socket Camera",  frame)
            # define range of blue color in SV
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([255, 255, 255])

            Gaussian = cv2.GaussianBlur(frame, (gussian_koef_neh, gussian_koef_neh), 0)
            #cv2.imshow('Gaussian Blurring', Gaussian)
            mask = cv2.inRange(Gaussian, lower_blue, upper_blue)
            res = cv2.bitwise_and(Gaussian, Gaussian, mask=mask)
            res = cv2.circle(res, (320, 20), 8, (0.0,255,255),1)
            res = cv2.circle(res, (505, 160), 8, (0.0, 255, 255), 1)
            res = cv2.circle(res, (101, 160), 8, (0.0, 255, 255), 1)
            cv2.imshow('res', res)
            if cv2.waitKey(1) == 27:
                break
                cv2.destroyAllWindows()

def geobot_control(geobot: Geobot):
    """Пример управления Геоботом"""
    sleep(3)
    while True:
        if x != None:
            geobot.go_to_local_point(x[0], x[1])
            while not geobot.point_reached():
                pass
        sleep(2)

#сам код работы
def pioneer_control(pioneer: Pioneer):
    """Пример управления Пионером"""
    pioneer.arm()
    time.sleep(1)
    pioneer.takeoff()
    #ВЗЛЁТ ВЫШЕ
    x = pioneer.get_local_position_lps()
    pioneer.go_to_local_point(x[0],x[1],1.5, -math.pi/2)
    time.sleep(3)
    top = 1
    lf = data_color_dots()[1]
    #вылетаем со стартовой площадки
    while top != 0:
        top = data_color_dots()[0]
        forward(0)
    print("край")
    rd = data_color_dots()[2]
    while rd!=0:
        turn(1)
        rd = data_color_dots()[2]
    while rd == 0:
        rd = data_color_dots()[2]
        forward(0)
        turn(0)
    print("трорпинка справа")
    top = data_color_dots()[0]
    lf = data_color_dots()[1]
    rd = data_color_dots()[2]
    while rd!= lf!= 0:
        turn(1)
        time.sleep(0.02)
        lf = data_color_dots()[1]
        rd = data_color_dots()[2]
    turn(1)
    time.sleep(2)
    print("мы на тропинке")
    top = data_color_dots()[0]
    lf = data_color_dots()[1]
    rd = data_color_dots()[2]
    #global x
    x = pioneer.get_local_position_lps()
    #далее полёт по тропинке
    while top != lf!= rd:
        pass
if __name__ == "__main__":
    args = [8000, 18000, 8001]
    if len(args) < 3:
        print(
            "Не переданы необходимые аргументы: порт Пионера, порт камеры Пионера, порт Геобота. Пример: python main.py 8000 18000 8001"
        )
        exit(1)

    pioneer = Pioneer(ip="127.0.0.1", mavlink_port=int(args[0]), log_connection=False)
    camera = Camera(ip="127.0.0.1", port=int(args[1]))
    geobot = Geobot(ip="127.0.0.1", mavlink_port=int(args[2]))

    camera_thread = threading.Thread(target=camera_stream, args=(camera,))
    pioneer_thread = threading.Thread(target=pioneer_control, args=(pioneer,))
    #color_dots_thread = threading.Thread(target=data_color_dots())
    geobot_thread = threading.Thread(target=geobot_control, args=(geobot,))
    x = pioneer.get_local_position_lps()

    try:
        camera_thread.start()
        pioneer_thread.start()
        #color_dots_thread.start()
        geobot_thread.start()

        camera_thread.join()
        pioneer_thread.join()
        #color_dots_thread.join()
        geobot_thread.join()
    except KeyboardInterrupt:
        pioneer.land()
        pioneer.disarm()






