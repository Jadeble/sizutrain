import socket
import struct
import threading
import time


class Controller:
    def __init__(self, dst):
        self.lock = False
        self.last_ges = "stop"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = dst

    # used to send a pack to robot dog
    def send(self, pack):
        self.socket.sendto(pack, self.dst)

    def drive_dog(self, ges, val = 10000):
        print(ges)
        if self.lock or ges == self.last_ges:
            return

        if self.last_ges == "squat" and ges != "squat":
            # stand up
            self.send(struct.pack('<3i', 0x21010202, 0, 0))
        else:
            # stop all actions
            self.send(struct.pack('<3i', 0x21010130, 0, 0))
            self.send(struct.pack('<3i', 0x21010131, 0, 0))
            self.send(struct.pack('<3i', 0x21010102, 0, 0))
            self.send(struct.pack('<3i', 0x21010135, 0, 0))

        if self.last_ges != "squat" and ges == "squat":
            self.send(struct.pack('<3i', 0x21010202, 0, 0))
        elif ges == "turning":
            def turn_360():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010135, 13000, 0))
                time.sleep(2.84)  # need time to turn 360 degrees
                self.send(struct.pack('<3i', 0x21010135, 0, 0))
                self.lock = False

            turn_thread = threading.Thread(target=turn_360)
            turn_thread.start()
        elif ges == "twisting":
            def dance():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010204, 0, 0))
                time.sleep(22)  # dance 22 seconds then it can process next gesture
                self.lock = False

            twist_thread = threading.Thread(target=dance)
            twist_thread.start()
        elif ges == "forward":
            self.send(struct.pack('<3i', 0x21010130, val, 0))
        elif ges == "back":
            self.send(struct.pack('<3i', 0x21010130, -val, 0))
        elif ges == "right":
            self.send(struct.pack('<3i', 0x21010131, val, 0))
        elif ges == "left":
            self.send(struct.pack('<3i', 0x21010131, -val, 0))

        self.last_ges = ges
         # 发消息咯
    def send(self, pack):
        self.socket.sendto(pack, self.dst)

    # 速度 left_right  zheng zuo
    #controller.Velocity(6600,-30000,0)
    #controller.Velocity(7000,30000,0)
    
    def Velocity(self, forward_back, left_right, rotate):
        msg1 = struct.pack('<3i', 0x21010130, forward_back, 0)
        self.send(msg1)
        msg2 = struct.pack('<3i', 0x21010131, -left_right, 0)
        self.send(msg2)
        msg3 = struct.pack('<3i', 0x21010135, rotate, 0)
        self.send(msg3)
        print("Change Velocity to LeftRight: %d, ForwardBack: %d, Rotate: %d" % (left_right, forward_back, rotate))
        
    # 停止
    def Stop(self):
        self.Velocity(0, 0, 0)
    
    # 转弯
    def Turn(self, direction):
        basic_turn_velocity = 16384
        if (direction == 1):
            self.Velocity(0, 0, basic_turn_velocity)
        elif (direction == 0):
            self.Velocity(0, 0, -basic_turn_velocity)
        # 0 zuozhuan 1 youzhuan
    
    MAX_VEL = int(32767)
    basic_vel = int(MAX_VEL // 2)
    turn_vel = [int(basic_vel / 1.5), int(basic_vel / 2), int(0), int(-basic_vel / 2), int(-basic_vel / 1.5)]
    turn_check_area = [2, 1, 3, 0, 4]
    #yidong moshi
    def Move_Mode(self):
        msg = struct.pack('<3i', 0x21010D06, 0, 0)
        self.send(msg)
        print("Change to Move Mode")
    
    #zhanli moshi
    def Stand_Mode(self):
        msg = struct.pack('<3i', 0x21010D05, 0, 0)
        self.send(msg)
        print("Change to Stand Mode")
       
    def Creep_Mode(self):
        msg = struct.pack('<3i', 0x21010406, 0, 0)
        self.send(msg)
        print("Change to Creep Mode")
        
    def Stairs_Mode(self):
        msg = struct.pack('<3i', 0x21010401, 0, 0)
        self.send(msg)
        print("Change to Stairs Mode")
        
    def Shoot_ball(self,foot):
        msg = struct.pack('<3i', 0x2101020C, foot, 0)
        self.send(msg)
        print("Succeed to Shoot")
        #foot = 1 zuojiao 2 youjiao
        
    def Continue(self,a):
        msg = struct.pack('<3i', 0x21010C06,a, 0)
        self.send(msg)
        #a = -1 qidong 2 quxiao 
               
    def Walk_Mode(self):
        msg = struct.pack('<3i', 0x21010300, 0, 0)
        self.send(msg)
        print("Change to Walk Mode")
        
import socket
import struct
import threading
import time


class Controller:
    def __init__(self, dst):
        self.lock = False
        self.last_ges = "stop"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = dst

    # used to send a pack to robot dog
    def send(self, pack):
        self.socket.sendto(pack, self.dst)

    def drive_dog(self, ges, val = 10000):
        print(ges)
        if self.lock or ges == self.last_ges:
            return

        if self.last_ges == "squat" and ges != "squat":
            # stand up
            self.send(struct.pack('<3i', 0x21010202, 0, 0))
        else:
            # stop all actions
            self.send(struct.pack('<3i', 0x21010130, 0, 0))
            self.send(struct.pack('<3i', 0x21010131, 0, 0))
            self.send(struct.pack('<3i', 0x21010102, 0, 0))
            self.send(struct.pack('<3i', 0x21010135, 0, 0))

        if self.last_ges != "squat" and ges == "squat":
            self.send(struct.pack('<3i', 0x21010202, 0, 0))
        elif ges == "turning":
            def turn_360():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010135, 13000, 0))
                time.sleep(2.84)  # need time to turn 360 degrees
                self.send(struct.pack('<3i', 0x21010135, 0, 0))
                self.lock = False

            turn_thread = threading.Thread(target=turn_360)
            turn_thread.start()
        elif ges == "twisting":
            def dance():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010204, 0, 0))
                time.sleep(22)  # dance 22 seconds then it can process next gesture
                self.lock = False

            twist_thread = threading.Thread(target=dance)
            twist_thread.start()
        elif ges == "forward":
            self.send(struct.pack('<3i', 0x21010130, val, 0))
        elif ges == "back":
            self.send(struct.pack('<3i', 0x21010130, -val, 0))
        elif ges == "right":
            self.send(struct.pack('<3i', 0x21010131, val, 0))
        elif ges == "left":
            self.send(struct.pack('<3i', 0x21010131, -val, 0))

        self.last_ges = ges
         # åæ¶æ¯å¯
    def send(self, pack):
        self.socket.sendto(pack, self.dst)

    # éåº¦ left_right  zheng zuo
    #controller.Velocity(6600,-30000,0)
    #controller.Velocity(7000,30000,0)
    
    def Velocity(self, forward_back, left_right, rotate):
        msg1 = struct.pack('<3i', 0x21010130, forward_back, 0)
        self.send(msg1)
        msg2 = struct.pack('<3i', 0x21010131, -left_right, 0)
        self.send(msg2)
        msg3 = struct.pack('<3i', 0x21010135, rotate, 0)
        self.send(msg3)
        print("Change Velocity to LeftRight: %d, ForwardBack: %d, Rotate: %d" % (left_right, forward_back, rotate))
        
    # åæ­¢
    def Stop(self):
        self.Velocity(0, 0, 0)
    
    # è½¬å¼¯
    def Turn(self, direction):
        basic_turn_velocity = 16384
        if (direction == 1):
            self.Velocity(0, 0, basic_turn_velocity)
        elif (direction == 0):
            self.Velocity(0, 0, -basic_turn_velocity)
        # 0 zuozhuan 1 youzhuan
    
    MAX_VEL = int(32767)
    basic_vel = int(MAX_VEL // 2)
    turn_vel = [int(basic_vel / 1.5), int(basic_vel / 2), int(0), int(-basic_vel / 2), int(-basic_vel / 1.5)]
    turn_check_area = [2, 1, 3, 0, 4]
    #yidong moshi
    def Move_Mode(self):
        msg = struct.pack('<3i', 0x21010D06, 0, 0)
        self.send(msg)
        print("Change to Move Mode")
    
    #zhanli moshi
    def Stand_Mode(self):
        msg = struct.pack('<3i', 0x21010D05, 0, 0)
        self.send(msg)
        print("Change to Stand Mode")
       
    def Creep_Mode(self):
        msg = struct.pack('<3i', 0x21010406, 0, 0)
        self.send(msg)
        print("Change to Creep Mode")
        
    def Stairs_Mode(self):
        msg = struct.pack('<3i', 0x21010401, 0, 0)
        self.send(msg)
        print("Change to Stairs Mode")
        
    def Shoot_ball(self,foot):
        msg = struct.pack('<3i', 0x2101020C, foot, 0)
        self.send(msg)
        print("Succeed to Shoot")
        #foot = 1 zuojiao 2 youjiao
        
    def Continue(self,a):
        msg = struct.pack('<3i', 0x21010C06,a, 0)
        self.send(msg)
        #a = -1 qidong 2 quxiao 
               
    def Walk_Mode(self):
        msg = struct.pack('<3i', 0x21010300, 0, 0)
        self.send(msg)
        print("Change to Walk Mode")
        

