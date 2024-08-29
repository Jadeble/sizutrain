import struct
import time
import cv2 as cv
import numpy as np
import threading
import os
import psutil


from utils.ColorDetection import ColorDetection
from utils.Controller import Controller
from utils.ArmController import ArmController

# global config
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)

# creat a controller
controller = Controller(server_address)   
arm_controller = ArmController("/dev/ttyUSB0")
cam = cv.VideoCapture("/dev/video5", cv.CAP_V4L2) # 机械臂摄像头



global stop_heartbeat
stop_heartbeat = False


# start to exchange heartbeat pack
def heart_exchange(con):
    pack = struct.pack('<3i', 0x21040001, 0, 0)
    while True:
        con.send(pack)
        time.sleep(0.25)  # 4Hz
heart_exchange_thread = threading.Thread(target = heart_exchange, args = (controller, ))
heart_exchange_thread.start() # 心跳

def touch_box():
    arm_controller.set_pose(0)
    time.sleep(3)
    arm_controller.set_pose(1)
    time.sleep(3)
    arm_controller.set_pose(2)
    time.sleep(2)
    arm_controller.set_pose(3)
    time.sleep(2)
    arm_controller.set_pose(0)


if __name__ == '__main__':
    controller.Stop()
    controller.Continue(2) 
    arm_controller.set_pose(0)
    time.sleep(2)
    arm_controller.set_pose(5)
    time.sleep(2)
    arm_controller.set_pose(6)
    time.sleep(2)
    arm_controller.set_pose(7)
    time.sleep(2)
    arm_controller.set_pose(8)
    time.sleep(2)
    arm_controller.set_pose(9)
    time.sleep(2)
 

