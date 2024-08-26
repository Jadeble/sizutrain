import threading
import cv2
import time
import numpy as np
from collections import Counter

class YOLOv8:
    """YOLOv8目标检测模型类，用于处理推理和可视化操作。"""
    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        初始化YOLOv8类的实例。
        参数:
            onnx_model: ONNX模型的路径。
            confidence_thres: 过滤检测的置信度阈值。
            iou_thres: 非极大抑制的IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.classes = ['1', '2', '3', '4', '5', '6', '7', '8','high', 'normal', 'low']
        # 类别数量
        self.nc = len(self.classes)
        # 为类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(self.nc, 3))

        # 初始化ONNX会话
        self.initialize_session(self.onnx_model)

    def draw_detections(self, img, box, score, class_id):
        """
        根据检测到的对象在输入图像上绘制边界框和标签。
        参数:
            img: 要绘制检测的输入图像。
            box: 检测到的边界框。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。
        返回:
            None
        """

        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别ID对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类名和得分的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, frame):
        """
        在进行推理之前，对输入图像进行预处理。
        参数:
            frame: 从视频流捕获的输入帧。
        返回:
            image_data: 预处理后的图像数据，准备好进行推理。
        """
        self.img = frame

        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]
        # 将图像颜色空间从BGR转换为RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # 将图像调整为匹配输入形状(640,640,3)
        img = cv2.resize(img, (self.input_width, self.input_height))
        # 将图像数据除以255.0进行归一化
        image_data = np.array(img) / 255.0
        # 转置图像，使通道维度成为第一个维度(3,640,640)
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
        # 扩展图像数据的维度以匹配期望的输入形状(1,3,640,640)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        # 返回预处理后的图像数据
        return image_data

    def postprocess(self, input_image, output):
        """
        对模型的输出进行后处理，以提取边界框、分数和类别ID。
        参数:
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回:
            list: 未排序的识别结果，格式为[(class_id, score, (left, top, width, height)), ...]
        """
        # 转置并压缩输出以匹配期望的形状：(8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))
        # 获取输出数组的行数
        rows = outputs.shape[0]
        # 存储检测到的边界框、分数和类别ID的列表
        boxes = []
        scores = []
        class_ids = []
        # 计算边界框坐标的比例因子
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # 遍历输出数组的每一行
        for i in range(rows):
            # 从当前行提取类别的得分
            classes_scores = outputs[i][4:]
            # 找到类别得分中的最大值
            max_score = np.amax(classes_scores)

            # 如果最大得分大于或等于置信度阈值
            if max_score >= self.confidence_thres:
                # 获取得分最高的类别ID
                class_id = np.argmax(classes_scores)

                # 从当前行提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算边界框的缩放坐标
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # 将类别ID、得分和边界框坐标添加到相应的列表中
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # 应用非极大抑制以过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # 存储检测结果的列表
        results = []

        # 遍历非极大抑制后选择的索引
        for i in indices:
            # 获取与索引对应的边界框、得分和类别ID
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            # 将结果添加到列表中
            results.append((self.classes[class_id], score, box))  # 将class_id转换为标签
            # 在输入图像上绘制检测结果
            self.draw_detections(input_image, box, score, class_id)

        # 返回未排序的结果
        return results

    def initialize_session(self, onnx_model):
        """
        初始化ONNX模型会话。
        :return:
        """
        # 检查是否存在 CUDA GPU 环境
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()

            if "CUDAExecutionProvider" in providers:
                print("Using CUDA")
                execution_providers = ["CUDAExecutionProvider"]
            else:
                print("Using CPU")
                execution_providers = ["CPUExecutionProvider"]
        
            # 初始化 InferenceSession，指定执行提供程序
            self.session = ort.InferenceSession(onnx_model, providers=execution_providers)
        except Exception as e:
            print(f"Error initializing session: {e}")
            raise

        return self.session
        
    def run(self, frame, save_path=None):
        """
        使用ONNX模型进行推理，并返回带有检测结果的输出图像。
        参数:
            frame: 从视频流捕获的输入帧。
            save_path: 保存检测结果图像的路径（如果需要）。
        返回:
            tuple: 按照x轴从左到右排序的识别结果，格式为[(class_id, score, (left, top, width, height)), ...]
        """
        # 获取模型的输入
        model_inputs = self.session.get_inputs()
        # 保存输入的形状，稍后使用
        # input_shape：(1,3,640,640)
        # self.input_width:640,self.input_height:640
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # 对图像数据进行预处理
        img_data = self.preprocess(frame)
        # 使用预处理后的图像数据运行推理,outputs:(1,84,8400)  8400 = 80*80 + 40*40 + 20*20
        outputs = self.session.run(None, {model_inputs[0].name: img_data})
        # 对输出进行后处理以获取输出图像
        results = self.postprocess(self.img, outputs)  # 输出图像
        
        # 如果提供了保存路径，则保存检测后的图像
        if save_path:
            cv2.imwrite(save_path, self.img)
        
        return results

class YOLOv8Runner:
    def __init__(self, model_path, conf_thres, iou_thres):
        """
        初始化YOLOv8Runner类的实例，并进行模型预热。
        参数:
            model_path: ONNX模型的路径。
            conf_thres: 置信度阈值。
            iou_thres: IoU（交并比）阈值。
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.detection = YOLOv8(self.model_path, self.conf_thres, self.iou_thres)
        self.video_capture = None

        # 进行模型预热
        self.warmup()

    def warmup(self):
        """
        进行模型预热，以便优化运行效率。
        """
        print("Warming up the model...")
        # 创建一个空的图像帧（与实际处理的图像大小相同）
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # 运行几次模型推理
        for _ in range(3):
            _ = self.detection.run(dummy_frame)
        print("Model warmup complete.")

    def load_camera(self, video_capture):
        """
        使用给定的cv2.VideoCapture对象设置摄像头并准备进行逐帧处理。
        参数:
            video_capture: cv2.VideoCapture 对象。
        """
        self.video_capture = video_capture
        if not self.video_capture.isOpened():
            raise ValueError("Error opening camera")

    def process_next_frame(self):
        """
        处理视频的下一帧，并返回检测结果。
        如果视频结束，返回 None。
        返回:
            处理后的帧和检测到的标签列表，或者在视频结束时返回 None。
        """
        if self.video_capture is None:
            raise ValueError("Camera not opened. Call `load_camera` first.")

        if not self.is_warmup_complete():
            print("Model is still warming up. Please wait.")
            return None

        ret, frame = self.video_capture.read()
        if not ret:
            print("End of video")
            return None

        # 运行模型并返回检测结果
        results = self.detection.run(frame)

        # 根据类别组筛选并获取面积最大的检测结果
        group1_result, group2_result = self.filter_and_select(results)

        # 显示处理后的帧
        #cv2.imshow('YOLOv8 Detection', frame)
        #cv2.waitKey(1)  # 仅用于刷新图像，不阻塞

        return group1_result, group2_result

    def filter_and_select(self, results):
        """
        筛选检测结果并返回两个最大面积的检测，一个属于前八种类别，一个属于后三种类别。
        参数:
            results: 模型检测结果列表 [(class_id, score, (left, top, width, height)), ...]
        返回:
            group1_result: 前八种类别中的最大检测结果或None。
            group2_result: 后三种类别中的最大检测结果或None。
        """
        group1_result = None
        group2_result = None

        for result in results:
            class_name = result[0]
            box = result[2]
            area = box[2] * box[3]  # 计算面积

            if class_name in self.group1:
                if group1_result is None or area > group1_result[2][2] * group1_result[2][3]:
                    group1_result = result
            elif class_name in self.group2:
                if group2_result is None or area > group2_result[2][2] * group2_result[2][3]:
                    group2_result = result

        return group1_result, group2_result

    def release(self):
        """
        释放视频捕获资源。
        """
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        
    def detect_and_count(self, iterations=5):
        """
        进行多次检测，并统计各类别的出现次数。
        参数:
            iterations: 迭代次数，默认为5次。
        返回:
            最常见的group1类别和group2类别。
        """
        group1_counter = Counter()
        group2_counter = Counter()

        for _ in range(iterations):
            group1_result, group2_result = self.process_next_frame()

            if group1_result:
                group1_counter[group1_result[0]] += 1
            if group2_result:
                group2_counter[group2_result[0]] += 1

        most_common_group1 = group1_counter.most_common(1)
        most_common_group2 = group2_counter.most_common(1)

        return (most_common_group1[0] if most_common_group1 else None,
                most_common_group2[0] if most_common_group2 else None)

if __name__ == "__main__":
    # 初始化模型运行器
    model_runner = YOLOv8Runner("cy960ir8.onnx", 0.3, 0.1)
    video_capture = cv2.VideoCapture('/dev/video0')
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 打开摄像头
    model_runner.load_camera(video_capture)  # 默认摄像头
    start_time = time.time()
    most_common_group1, most_common_group2 = model_runner.detect_and_count(iterations=5)
    print(f"Most common in group 1: {most_common_group1}")
    print(f"Most common in group 2: {most_common_group2}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    time.sleep(2)
    start_time = time.time()
    most_common_group1, most_common_group2 = model_runner.detect_and_count(iterations=5)
    print(f"Most common in group 1: {most_common_group1}")
    print(f"Most common in group 2: {most_common_group2}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    time.sleep(3)
    start_time = time.time()
    most_common_group1, most_common_group2 = model_runner.detect_and_count(iterations=5)
    print(f"Most common in group 1: {most_common_group1}")
    print(f"Most common in group 2: {most_common_group2}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    # 释放资源
    model_runner.release()
    print("Program finished.")
