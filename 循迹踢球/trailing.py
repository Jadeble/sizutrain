from controller import Controller
# from onnxmodel import ONNXModel
import struct
import time
import cv2
import numpy as np
import threading
import time
import subprocess

client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)

real_capture = cv2.VideoCapture('/dev/video4')
real_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
real_img = None

basic_forward_vel = 8000
basic_forward_vel1 = 9000
basic_turn_vel = 8000
basic_left_right_vel = 20000


WIDTH = 640
HEIGHT = 480
bright_thres = 200
bright_area_thres = 0.09


# 在图像内划五个矩形框，识别每个矩形框内白色像素的比率，返回对应的值
def Check_Line(img):
    squire_length = WIDTH // 5
    squire_begin_y = HEIGHT // 2 - squire_length // 2
    squire_end_y = HEIGHT // 2 + squire_length
    result = 0

    for i in range(1, 6):
        squire_begin_x = (i - 1) * squire_length
        squire_end_x = i * squire_length
        cnt = 0

        squire_region = img[squire_begin_y:squire_end_y, squire_begin_x:squire_end_x]
        bright_pixels = np.sum((squire_region[:, :, 0] > bright_thres) & (squire_region[:, :, 1] > bright_thres) & (
                squire_region[:, :, 2] > bright_thres))
        cnt = bright_pixels

        if cnt > bright_area_thres * squire_length * squire_length:
            result |= (1 << (i - 1))

    return result

# 在近端判断看到球，通过在视野内划若干个范围框，返回球在视野内的大致左右位置
def Check_Ball2(img):
    squire_length = WIDTH // 5
    squire_begin_y = HEIGHT // 2 - squire_length // 2
    squire_end_y = HEIGHT // 2 + squire_length

    result = 0
    for i in range(1, 6):
        squire_begin_x = (i - 1) * squire_length
        squire_end_x = i * squire_length
        cnt = 0
        squire_region = img[squire_end_y:480, squire_begin_x:squire_end_x]
        red_pixels = np.sum(
            (squire_region[:, :, 0] > 0) & (squire_region[:, :, 1] > 0) & (squire_region[:, :, 2] > 200))
        # print(f"{i}:{red_pixels / squire_length / squire_length}",f"{red_pixels}")
        if red_pixels > 0.5 * squire_length * squire_length:
            result |= (1 << (i - 1))
    return result


def heart_exchange(con):
    pack = struct.pack('<3i', 0x21040001, 0, 0)
    while True:
        con.send(pack)
        time.sleep(0.25)  # 4Hz


controller = Controller(server_address)
heart_exchange_thread = threading.Thread(target=heart_exchange, args=(controller,))
heart_exchange_thread.start()  # 心跳

if __name__ == '__main__':
    controller.Stop()
    controller.Continue(2)
    pack = struct.pack('<3i', 0x21010202, 0, 0)
    print(1)
    controller.send(pack)
    time.sleep(5)
    print(2)
    controller.send(pack)
    time.sleep(5)
    print(3)
    controller.send(pack)
    time.sleep(5)
    controller.Stop()

    squire_length = WIDTH // 5
    squire_length_n = WIDTH // 10
    squire1_end_y = squire_length // 2
    squire1_end_y_n = squire_length_n // 2
    squire_begin_y = HEIGHT // 2 - squire_length // 2
    squire_end_y = HEIGHT // 2 + squire_length
    for i in range(10):
        _, real_img = real_capture.read()

    cv2.rectangle(real_img, (int(0), int(176)), (int(640), int(304)), (0, 0, 255))
    cv2.rectangle(real_img, (int(0), int(squire_length // 5)), (int(640), int(squire1_end_y + 10)), (255, 0, 0))
    cv2.rectangle(real_img, (int(0), int(squire1_end_y + 10)), (int(640), int(squire1_end_y * 3)), (0, 255, 0))
    zhuan1 = 0
    # 用and连接判断置标志位。
    result_ckl = 0
    result_Ball2 = 0

    controller.Continue(-1)
    time.sleep(2)
    controller.Velocity(-8000, 0, 0)
    time.sleep(2)
    controller.Stop()
    controller.Continue(-1)
    time.sleep(1)
    
    
#循迹部分
    for i in range(10):
        _, real_img = real_capture.read()
        result_ckl = Check_Line(real_img)
    while True:
        _, real_img = real_capture.read()
        result_ckl = Check_Line(real_img)
        for i in range(1, 6):
            squire_begin_x = (i - 1) * squire_length
            squire_end_x = i * squire_length
            cv2.rectangle(real_img, (squire_begin_x, squire_begin_y), (squire_end_x, squire_end_y), (255, 255, 255))
        cv2.imshow('real', real_img)
        cv2.waitKey(1)
        if result_ckl == 4:
            print("jiuzheng")
            break
        elif result_ckl == 3 or result_ckl == 1 or result_ckl == 2 or result_ckl == 6:
            controller.Velocity(6950, 25000, 0)
        elif result_ckl == 24 or result_ckl == 16 or result_ckl == 8 or result_ckl == 12:
            controller.Velocity(6950, -25000, 0)

    while True:
        _, real_img = real_capture.read()
        for i in range(1, 6):
            squire_begin_x = (i - 1) * squire_length
            squire_end_x = i * squire_length
            cv2.rectangle(real_img, (squire_begin_x, squire_begin_y), (squire_end_x, squire_end_y), (0, 0, 255))
            cv2.rectangle(real_img, (squire_begin_x, squire_begin_y), (squire_end_x, 480), (0, 0, 255))
        for i in range(3, 10):
            squire_begin_x = (i - 1) * squire_length_n
            squire_end_x = i * squire_length_n
            cv2.rectangle(real_img, (squire_begin_x, int(squire1_end_y_n // 2)),
                          (squire_end_x, int(squire1_end_y_n * 4)), (0, 255, 0))
        cv2.imshow('real', real_img)
        cv2.waitKey(1)
        result_ckl = Check_Line(real_img)

#检测交叉区域，判定踢球点
        if result_ckl == 4:
            controller.Velocity(9000, 0, 0)
        elif result_ckl == 3 or result_ckl == 1 or result_ckl == 2 or result_ckl == 6:
            controller.Velocity(9000, 22000, 0)
        elif result_ckl == 24 or result_ckl == 16 or result_ckl == 8 or result_ckl == 12:
            controller.Velocity(9000, -22000, 0)
        elif (result_ckl > 7 and zhuan1 == 0):  # 第一个踢球点
            controller.Velocity(basic_forward_vel, 0, 0)
            time.sleep(4)
            controller.Stop()
            controller.Continue(-1)
            time.sleep(1)
            controller.Turn(1)
            time.sleep(0.5)
            controller.Stop()
            controller.Continue(-1)
            time.sleep(0.5)
            controller.Velocity(-basic_forward_vel, 0, 0)
            time.sleep(1)
            controller.Stop()
            controller.Continue(-1)
            time.sleep(0.3)
            while True:
                _, real_img = real_capture.read()
                result_ckl = Check_Line(real_img)
                for i in range(1, 6):
                    squire_begin_x = (i - 1) * squire_length
                    squire_end_x = i * squire_length
                    cv2.rectangle(real_img, (squire_begin_x, squire_begin_y), (squire_end_x, squire_end_y),
                                  (255, 255, 255))
                if result_ckl == 4:
                    print("jiuzheng1")
                    break
                elif result_ckl == 3 or result_ckl == 1 or result_ckl == 2 or result_ckl == 6:
                    controller.Velocity(6950, 25000, 0)
                elif result_ckl == 24 or result_ckl == 16 or result_ckl == 8 or result_ckl == 12:
                    controller.Velocity(6950, -25000, 0)
            while True:
                _, real_img = real_capture.read()
                for i in range(3, 10):
                    squire_begin_x = (i - 1) * squire_length_n
                    squire_end_x = i * squire_length_n
                    cv2.rectangle(real_img, (squire_begin_x, int(squire1_end_y_n // 2)),
                                  (squire_end_x, int(squire1_end_y_n * 4)), (0, 255, 0))
                result_Ball2 = Check_Ball2(real_img)
                print(f"Ball 2 {result_Ball2}")
                if result_Ball2 == 0:
                    controller.Velocity(8000, 0, 0)
                elif result_Ball2 == 3:  # 3
                    controller.Velocity(9000, 15000, 0)
                    time.sleep(2)
                    break
                elif result_Ball2 == 1:  # 1
                    controller.Velocity(0, basic_left_right_vel, 0)
                elif result_Ball2 == 2 or result_Ball2 > 3:
                    controller.Velocity(0, -basic_left_right_vel, 0)
                for i in range(1, 6):
                    squire_begin_x = (i - 1) * squire_length
                    squire_end_x = i * squire_length
                    cv2.rectangle(real_img, (squire_begin_x, squire_begin_y), (squire_end_x, 480), (0, 0, 255))
                cv2.imshow('ball2', real_img)
                cv2.waitKey(1)
            controller.Continue(-1)
            time.sleep(0.3)
            controller.Velocity(-basic_forward_vel1, 0, 0)
            time.sleep(2)
            controller.Continue(-1)
            time.sleep(1)
            controller.Turn(0)
            time.sleep(1.5)
            controller.Stop()
            controller.Continue(-1)
            time.sleep(1)
            controller.Velocity(basic_forward_vel, 0, 0)
            time.sleep(4)
            controller.Left_Right(1)
            time.sleep(1)
            for i in range(10):
                _, real_img = real_capture.read()
                result_ckl = Check_Line(real_img)
            while True:
                _, real_img = real_capture.read()
                result_ckl = Check_Line(real_img)
                if result_ckl == 4:
                    print("youzheng1")
                    break
                elif result_ckl == 3 or result_ckl == 1 or result_ckl == 2 or result_ckl == 6:
                    controller.Velocity(6950, 25000, 0)
                elif result_ckl == 24 or result_ckl == 16 or result_ckl == 8 or result_ckl == 12:
                    controller.Velocity(6950, -25000, 0)
            zhuan1 = 1
        elif (result_ckl == 31 and zhuan1 == 1):  # 第一个十字
            controller.Velocity(8800, 0, 0)
            time.sleep(3)
            for i in range(10):
                _, real_img = real_capture.read()
                result_ckl = Check_Line(real_img)
            while True:
                _, real_img = real_capture.read()
                result_ckl = Check_Line(real_img)
                cv2.imshow('real', real_img)
                cv2.waitKey(1)
                if result_ckl == 4:
                    print("jiuzheng2")
                    break
                elif result_ckl == 3 or result_ckl == 1 or result_ckl == 2 or result_ckl == 6:
                    controller.Velocity(6950, 25000, 0)
                elif result_ckl == 24 or result_ckl == 16 or result_ckl == 8 or result_ckl == 12:
                    controller.Velocity(6950, -25000, 0)
            zhuan1 = 2
