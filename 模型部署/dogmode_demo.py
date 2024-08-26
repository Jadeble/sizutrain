from conical import model_run_conical as conical
from modelrun import YOLOv8Runner
import cv2
import time
from controller import Controller
import struct
import threading
import time



client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)

def heart_exchange(con):
    pack = struct.pack('<3i', 0x21040001, 0, 0)
    while True:
        con.send(pack)
        time.sleep(0.25)  # 4Hz

controller = Controller(server_address)
heart_exchange_thread = threading.Thread(target = heart_exchange, args = (controller, ))
heart_exchange_thread.start() # 心跳

if __name__ == "__main__":
    #realimg = cv2.VideoCapture('/dev/video4')
    video_capture = cv2.VideoCapture('/dev/video0')
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    start_time = time.time()
    model_runner = YOLOv8Runner("cy960ir8.onnx", 0.3, 0.1)
    model_runner.load_camera(video_capture)
    model_runner.start_warmup()  # 第一次调用，启动预热
    controller.Stop()
    controller.Continue(2)    
    pack = struct.pack('<3i', 0x21010202, 0, 0)
    print(1)
    model_runner.continue_warmup()  # 第二次调用，继续预热
    controller.send(pack)
    time.sleep(4)
    print(2)
    model_runner.continue_warmup()  # 第三次调用，继续预热
    controller.send(pack)
    time.sleep(4)
    print(3)
    model_runner.continue_warmup()  # 第四次调用，完成预热
    controller.send(pack)
    time.sleep(3)
    controller.Stop()
    if model_runner.is_warmup_complete():
        print("Warmup complete, ready to process frames.")
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    
    time.sleep(3)
    num, dash = model_runner.process_next_frame()
    print(f"num  is {num}")
    print(f"dash is {dash}")
    
    controller.Continue(-1)
    controller.Velocity(8000, 0, 0)
    time.sleep(1)
    for i in range(5):
        num, dash = model_runner.process_next_frame()
        print(f"num  is {num}")
        print(f"dash is {dash}")
    controller.Stop()
    for i in range(5):
        num, dash = model_runner.process_next_frame()
        print(f"num  is {num}")
        print(f"dash is {dash}")
    time.sleep(1)
    controller.Continue(2)
    controller.send(pack)
    time.sleep(3)
    
    
    
