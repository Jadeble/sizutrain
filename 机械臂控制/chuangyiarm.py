from controller import Controller
#from onnxmodel import ONNXModel

import struct
import time
import cv2
import numpy as np
import threading
import time
import math
#import psutil
import os

from utils.ColorDetection import ColorDetection
from utils.ArmController import ArmController


camera =cv2.VideoCapture('/dev/video5',cv2.CAP_V4L2) 
time.sleep(1)
arm_controller = ArmController("/dev/ttyUSB0")
time.sleep(3)
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)
dog_cam_mtx = np.array([[547.3786,  0.    ,326.2253],
 			[  0.    ,419.916 ,243.6405],
 			[  0.    ,  0.    ,  1.    ]])  # 相机内参矩阵
dog_cam_dist = np.array([ 0.0295,-0.3043, 0.0001, 0.0031, 2.7265])  # 畸变系数


arcode_mode = cv2.aruco.DICT_4X4_50  # 用于定位的ar码的规格
marker_size = 10  # aruco码的实际尺寸是10厘米
left_right_thr = 10  # 左右调整位置时，可容忍的偏差 [-thr, thr] 像素
forward_thr = 30  # 直走时，目标偏差超过thr像素时，开始调整左右位置
target_id = 1  # 目标ar码的值为多少
ar_distance = 12 # 距离ar码多少cm时停下
font = cv2.FONT_HERSHEY_SIMPLEX
arrive = False



basic_forward_vel = 8000
basic_turn_vel = 8000
basic_left_right_vel = 20000



stage_two_begin_forward_time = 7.5
stage_two_find_line_left_time = 40

def heart_exchange(con):
    pack = struct.pack('<3i', 0x21040001, 0, 0)
    while True:
        con.send(pack)
        time.sleep(0.25)  # 4Hz

controller = Controller(server_address)
heart_exchange_thread = threading.Thread(target = heart_exchange, args = (controller, ))
heart_exchange_thread.start() # 心跳
#model = ONNXModel('model.onnx')

WIDTH = 640
HEIGHT = 480
bright_thres = 200
bright_area_thres = 0.1
turn_time = 2

def Checktong(img):

    result = 0
    squire_region1 = img[50:240, 50:300]
    squire_region2 = img[50:240, 350:600]
    bright_pixels1 = np.sum((squire_region1[:, :, 0] < 150) & (squire_region1[:, :, 1] < 150) & (
                squire_region1[:, :, 2] > 160))
    bright_pixels2 = np.sum((squire_region2[:, :, 0] < 150) & (squire_region2[:, :, 1] < 150) & (
                squire_region2[:, :, 2] > 160))
    print(f"{bright_pixels1}")
    print(f"{bright_pixels2}")
    if bright_pixels1 > 500:#红在zuo
        result = 1
    elif bright_pixels2 > 500:#红在you
        result = 2
    else:
        result = 0
    return result
    
if __name__ == '__main__':
    # warm up  
    controller.Stop()
    controller.Continue(2) 
    time.sleep(1)
    #controller.Zero()
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
    time.sleep(5)
    dictionary = cv2.aruco.getPredefinedDictionary(arcode_mode)
    dog_cam = cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2)
    arm_controller.set_pose(0)
    time.sleep(2)
    # 找二维码的一些定义，看不懂
    center_x = dog_cam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
    center_y = dog_cam.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
    drive_left_right = True
    drive_forward = False
    while drive_left_right or drive_forward:
        #print('12345')
        ret, frame = dog_cam.read()
        # print('23456')
        # print('34567')
        
        if not ret:
            continue
        # 检测aruco码
        arcode_detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, rejected = arcode_detector.detectMarkers(frame)
        # 估计aruco码的姿态
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, -marker_size / 2, 0],
                                  [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        distance_list = []
        for c in corners:
            _, rvec, tvec = cv2.solvePnP(marker_points, c, dog_cam_mtx, dog_cam_dist, False, cv2.SOLVEPNP_IPPE_SQUARE)
            R, _ = cv2.Rodrigues(rvec)
            cam_tvec = -R.T @ tvec
            distance_list.append(np.linalg.norm(cam_tvec))
        # 遍历每个aruco码
        if ids is None:
            continue

        distance = None
        for i in range(len(ids)):
            if ids[i] == target_id:

                '''
                distance = distance_list[i]
                aruco_center_x = corners[i].mean(axis=1)[0][0]

                break
                '''
                aruco_center_x = corners[i].mean(axis=1)[0][0]
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, dog_cam_mtx, dog_cam_dist)

                (rvec-tvec).any() # get rid of that nasty numpy value array error

                for i in range(rvec.shape[0]):
                    # cv2.aruco.drawAxis(frame, dog_cam_mtx, dog_cam_dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                    cv2.aruco.drawDetectedMarkers(frame, corners,ids)
                    pass

                cv2.putText(frame, "Id: " + str(ids), (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

                distance = tvec[0][0][2] * 100 * 0.0194 * 100 - 0.0322 * 100  # 单位是cm

                cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('cm'), (0, 110), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        
        if distance is not None:
            diff_x = aruco_center_x - center_x

            # 控制机器狗向前走
            if drive_forward:
                # 左右偏差超出阈值
                if diff_x > forward_thr or diff_x < -forward_thr:
                    drive_left_right = True
                    drive_forward = False
                    controller.Stop()
                # 左右偏差在阈值内，只考虑前后
                else:
                    # 距离箱子的距离达到阈值时停下
                    print("ArUco code distance: {:.2f} cm".format(distance))
                    if distance <= ar_distance:
                        print("Arrived!!")
                        controller.Stop()
                        dog_cam.release()
                        cv2.destroyWindow("camera")
                        drive_forward = False
                        arrive = True
                        controller.Stop()
                    else:
                        if distance <= 50:
                            controller.Velocity(7500, 0, 0)
                        else:
                            controller.Velocity(8000, 0, 0)
            # 控制机器狗左右移动
            if drive_left_right:
                if diff_x > left_right_thr: # 右
                    controller.Velocity(0, -15000, 0)
                elif diff_x < - left_right_thr: # 左
                    controller.Velocity(0, 15000, 0)
                else:
                    controller.Stop() # 停，继续控制前后
                    drive_left_right = False
                    drive_forward = True
        if arrive:
            # controller.velocity(7500, 0, 0)
            # time.sleep(0.5)
            controller.Stop()
            print("stop")
            time.sleep(1)
            break
    
    
    arm_controller.set_pose(4)
    time.sleep(4)
    a = 0
    for i in range (20):
    #while True:
      _, frame = camera.read()
      _, frame = camera.read()
      _, frame = camera.read()
      wb = cv2.xphoto.createSimpleWB()
      frame = wb.balanceWhite(frame)
      cv2.rectangle(frame, (50, int(50)),
                  (300, int(240)), (255, 0, 0))
      cv2.rectangle(frame, (350, int(50)),
                  (600, int(240)), (255, 0, 0))      
      cv2.imshow("h", frame)
      cv2.waitKey(1) 
      a = Checktong(frame)
      print(a)

    # arm_controller.set_pose(5)
    time.sleep(3)
    if a==1:
        controller.Mini_Left_Right(1)
        time.sleep(2)
        controller.Stop()
        time.sleep(1)
        arm_controller.set_pose(5)
        time.sleep(2)
        arm_controller.set_pose(6)
        time.sleep(2)
        arm_controller.set_pose(7)
        time.sleep(2)
        controller.Left_Right(1)
        time.sleep(3)
        time.sleep(4)
        controller.Stop()
        arm_controller.set_pose(8)
        time.sleep(2)
        arm_controller.set_pose(9)
        time.sleep(2)
        arm_controller.set_pose(0)
    elif a==2:
        controller.Mini_Left_Right(0)
        time.sleep(2)
        controller.Stop()
        time.sleep(1)
        arm_controller.set_pose(5)
        time.sleep(2)
        arm_controller.set_pose(6)
        time.sleep(2)
        arm_controller.set_pose(7)
        time.sleep(2)
        controller.Left_Right(1)
        time.sleep(3)
        time.sleep(5)
        controller.Stop()
        arm_controller.set_pose(8)
        time.sleep(2)
        arm_controller.set_pose(9)
        time.sleep(2)
        arm_controller.set_pose(0)

