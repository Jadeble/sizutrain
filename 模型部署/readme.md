## 在Jetson Xavier NX中使用CUDA加速模型推理
### 实训背景
做移动平台的本地视觉识别，需要有高算力的支持，使用CUDA对模型的推理加速是必不可少，有什么什么优势，balabala..........................
### 实训内容
将深度学习模型成功部署在平台中，并保持较高的推理速度和准确度
### 实训目的及要求
掌握在边缘平台中配置模型运行环境并部署模型的方法
实现2024中国高校智能机器人创意大赛四足专项中锥形桶识别
实现使用CUDA对模型推理进行加速
### 实验软硬件环境
>硬件：Jetson Xavier NX开发板，摄像头（可选）
>软件：
>-  JetPack 4.4
>- Python 3.7
>- onnxruntime 1.11.0
>- opencv-python 4.9.0
>

### 实训操作指南
1. 先训练好自己用来进行识别的模型，由于当前用于视觉识别的模型大部分使用yolo，所以使用YOLOv8s模型为例
- 准备好拍摄的数据集，由于使用机械狗进行图像的识别，建议使用机械狗进行数据集拍摄，在拍摄过程中可以采用将识别对象进行手动旋转、更换场景增加复杂度等方式，采集合适的数据集，一般采集3k~5k张图像。
- 对采集到的数据进行增强处理，由于YOLO自带了旋转、裁剪、模糊等较为通用的数据增强手段，我们需要进行其他的数据增强方式，例如将图像进行空间扭曲，模拟不同角度观察物体；还可以将感兴趣区域裁剪后贴在其他背景图中例如coco2017数据集，人为创造新的环境。使用增强后的数据集训练一个YOLOv8s模型，并使用YOLO官方工具，将模型转为ONNX平台
- 训练好一个模型后，就可以在机械狗上进行模型部署。首先是环境的选择，YOLOv8s本身是使用Pytorch进行推理，但是pytorch原生推理速度较慢，且在当前平台上部署pytorch对环境要求较为严格；TensorRT是Nvidia自家的加速推理框架，推理速度快，但是部署难度较高；ONNX部署难度低，跨平台兼容性好，推理速度稍逊于TensorRT，故选择ONNX作为运行模型的平台。配置ONNX时，需要在Jetson zoo下载onnxruntime-gpu的离线安装包，以适配当前的平台。
- 运行模型需要进行如下几步：加载模型、预处理、后处理，加载模型即初始化onnx会话，将模型详细传递至算力设备，预处理是将获取到的图像处理为模型可计算的图像格式，后处理是将模型推理的结果进行解算，提取出需要的有效信息。参考代码如下
```python
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
        self.classes = ['blue', 'yellow', 'red', 'orange']
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
        image_data = np.expand_dims(image_data, axis=0).astype(np.float16)

        # 返回预处理后的图像数据
        return image_data

    def postprocess(self, input_image, output):
        """
        对模型的输出进行后处理，以提取边界框、分数和类别ID。
        参数:
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回:
            tuple: 按照x轴从左到右排序的识别结果，格式为[(class_id, score, (left, top, width, height)), ...]
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

        # 按照x轴从左到右对结果进行排序
        results.sort(key=lambda x: x[2][0])  # x[2][0] 是left的值

        # 返回排序后的结果
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
        
    def run(self, frame):
        """
        使用ONNX模型进行推理，并返回带有检测结果的输出图像。
        参数:
            frame: 从视频流捕获的输入帧。
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
        return self.postprocess(self.img, outputs)  # 输出图像
```
- 构建完运行YOLO模型的类之后，可以编写运行模型的函数
```python
def model_run_conical(video_capture, model_path="conical_f_ir8.onnx", conf_thres=0.1, iou_thres=0.1):
    """
    封装的函数，用于执行YOLOv8模型推理。
    参数:
        video_capture (cv2.VideoCapture): 视频流捕获对象。
        model_path (str): ONNX模型的路径。
        conf_thres (float): 置信度阈值。
        iou_thres (float): IoU（交并比）阈值。
    返回:
        tuple: 按照x轴从左到右排序的识别结果，格式为[(class_id, score, (left, top, width, height)), ...]
    """
    # 创建YOLOv8实例
    detection = YOLOv8(model_path, conf_thres, iou_thres)
    
    while True:
        # 从视频流捕获一帧
        ret, frame = video_capture.read()
        if not ret:
            break

        wb = cv2.xphoto.createSimpleWB()
        frame = wb.balanceWhite(frame)
        
        brightness_value = 5
        frame = frame.astype(np.float32)
        # 将图像转换为 HSV 色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v + brightness_value, 1, 254)
        # 确保通道的尺寸相同
        h = cv2.resize(h, (s.shape[1], s.shape[0]))
        v = cv2.resize(v, (s.shape[1], s.shape[0]))

        # 确保通道的类型相同
        v = v.astype(h.dtype)
        # 合并通道并转换回 BGR
        hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # 模型推理
        results = detection.run(frame)
        break

    # 释放视频捕获对象和关闭所有窗口
    video_capture.release()
    cv2.destroyAllWindows()

    # 用于存储 red 和 orange 标签的相关信息列表
    red_info_list = []
    orange_info_list = []

    # 遍历识别结果，查找 red 和 orange 标签
    for i, result in enumerate(results):
        label, score, box = result
        if label == "red":
            # 计算 red 标签区域的中心点
            center_x = box[0] + box[2] // 2
            center_y = box[1] + box[3] // 2
    
            # 提取中心点的颜色值
            center_color = detection.img[center_y, center_x]
            r, g, b = center_color

            # 存储 red 标签的信息
            red_info_list.append({"index": i, "g_value": g, "score": score, "box": box})
            print(f"red 标签中心点: ({center_x}, {center_y}), 像素值: (R: {r}, G: {g}, B: {b})")

        elif label == "orange":
            # 计算 orange 标签区域的中心点
            center_x = box[0] + box[2] // 2
            center_y = box[1] + box[3] // 2
    
            # 提取中心点的颜色值
            center_color = detection.img[center_y, center_x]
            r, g, b = center_color
    
            # 存储 orange 标签的信息
            orange_info_list.append({"index": i, "g_value": g, "score": score, "box": box})
            print(f"orange 标签中心点: ({center_x}, {center_y}), 像素值: (R: {r}, G: {g}, B: {b})")

    # 处理检测结果：两种情况——多个 red, 多个 orange 或 red 与 orange 同时存在
    if len(red_info_list) > 0 and len(orange_info_list) > 0:
        # 如果同时检测到 red 和 orange 标签
        for red_info in red_info_list:
            for orange_info in orange_info_list:
                if red_info["g_value"] > orange_info["g_value"]:
                    # 将 red 修改为 orange，orange 修改为 red
                    results[red_info["index"]] = ("orange", red_info["score"], red_info["box"])
                    results[orange_info["index"]] = ("red", orange_info["score"], orange_info["box"])
                else:
                    # 保持原样，但保证标签的一致性
                    results[red_info["index"]] = ("red", red_info["score"], red_info["box"])
                    results[orange_info["index"]] = ("orange", orange_info["score"], orange_info["box"])

    elif len(red_info_list) > 1:
        # 如果存在多个 red 标签
        red_info_list.sort(key=lambda x: x["g_value"], reverse=True)
        # 将 G 值较大的 red 标签修改为 orange
        results[red_info_list[0]["index"]] = ("orange", red_info_list[0]["score"], red_info_list[0]["box"])

    elif len(orange_info_list) > 1:
        # 如果存在多个 orange 标签
        orange_info_list.sort(key=lambda x: x["g_value"], reverse=True)
        for i in range(1, len(orange_info_list)):
            # 将 G 值较小的 orange 标签修改为 red
            results[orange_info_list[i]["index"]] = ("red", orange_info_list[i]["score"], orange_info_list[i]["box"])

    class_ids = [result[0] for result in results]
    # 返回排序后的结果
    return class_ids, detection.img
```
- 在这段代码中，不仅有模型运行部分，还有对输入图像的图像处理和对模型输出的进一步处理。在输入图像时进行了白平衡处理和亮度修改，后处理时针对橙色和红色在不同光照环境下难以区分的问题进行了进一步的修正

### 参考
