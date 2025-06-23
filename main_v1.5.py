import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

class HeadMovementDetector:
    """
    头部动作检测器类 v1.5
    使用滑动窗口机制进行更稳定的头部动作检测
    用于检测视频中的头部动作，包括点头、摇头、眨眼和张嘴等动作
    使用MediaPipe面部网格检测器实现面部关键点的检测和跟踪
    """

    def __init__(self):
        """初始化头部动作检测器"""
        #初始化mediapipe的服务
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # 嘴部和眼睛的关键点索引
        # MediaPipe 468点面部网格的关键点索引
        self.mouth_indices = [61, 291, 0, 17, 291, 405, 17, 314, 405, 314, 17, 291, 405, 314]  # 嘴部轮廓
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]  # 左眼轮廓
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384,
                                  398]  # 右眼轮廓

        # 头部姿态计算的关键点
        self.nose_tip = 1  # 鼻尖
        self.left_eye_center = 33  # 左眼中心
        self.right_eye_center = 263  # 右眼中心
        self.chin = 152  # 下巴

        # 嘴部开合度计算的关键点
        self.upper_lip = 13  # 上唇中点
        self.lower_lip = 14  # 下唇中点

        # 初始正脸状态记录
        self.initial_pitch = None  # 初始pitch值
        self.initial_yaw = None  # 初始yaw值
        self.is_initialized = False  # 是否已初始化
        self.initialization_frames = 10  # 初始化所需的帧数
        self.init_pitch_values = []  # 初始化期间收集的pitch值
        self.init_yaw_values = []  # 初始化期间收集的yaw值

        # 动作阈值设置
        # self.pitch_up_threshold = 0.5  # 抬头动作阈值（基于离散值检测）
        # self.pitch_down_threshold = 0.5  # 点头动作阈值（基于离散值检测）
        self.pitch_threshold = 15
        self.yaw_threshold = 0.8  # 摇头动作阈值（度）
        self.eye_closed_threshold = 0.5  # 眼睛闭合阈值 (EAR标准阈值)
        self.mouth_open_threshold = 0.05  # 张嘴阈值

        #滑动窗口帧数设置
        self.last_action_list = [] #上一帧的动作列表
        self.yaw_num_threshold = 3 #摇头滑动窗口
        self.pitch_num_threshold = 5 #点头滑动窗口
        self.mouth_num_threshold = 3 #嘴巴滑动窗口
        self.eye_num_threshold = 0 #眼睛滑动窗口

        self.num_yaw_left = 0
        self.num_yaw_right = 0
        self.num_pitch_up = 0
        self.num_pitch_down = 0
        self.num_eye_close = 0
        self.num_mouth_open= 0

        #保存路径
        self.output_dir = None

        #人脸开始出现的帧
        self.start_index = 0

    def calculate_head_pose(self, landmarks):
        """
        计算头部姿态（pitch和yaw）
        使用更准确的3D头部姿态计算方法

        参数:
            landmarks: MediaPipe检测到的面部关键点列表

        返回:
            tuple: (pitch, yaw) 俯仰角和偏航角（单位：度）
        """
        # 获取关键点坐标
        nose = landmarks[self.nose_tip]  # 鼻尖
        left_eye = landmarks[self.left_eye_center]  # 左眼中心
        right_eye = landmarks[self.right_eye_center]  # 右眼中心
        chin = landmarks[self.chin]  # 下巴

        # 计算眼睛中心点
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        eye_center_z = (left_eye.z + right_eye.z) / 2

        # 计算头部姿态
        # 使用眼睛中心到鼻子的向量计算pitch
        pitch_dx = nose.x - eye_center_x
        pitch_dy = nose.y - eye_center_y
        pitch_dz = nose.z - eye_center_z
        
        # 使用左右眼的向量计算yaw
        yaw_dx = right_eye.x - left_eye.x
        yaw_dy = right_eye.y - left_eye.y
        yaw_dz = right_eye.z - left_eye.z

        # 计算角度
        pitch = np.arctan2(pitch_dy, np.sqrt(pitch_dx**2 + pitch_dz**2))
        yaw = np.arctan2(yaw_dx, np.sqrt(yaw_dy**2 + yaw_dz**2))

        # 将弧度转换为角度
        return np.degrees(pitch), np.degrees(yaw)
    def calculate_mouth_openness(self, landmarks):
        """
        计算嘴部开合程度

        参数:
            landmarks: MediaPipe检测到的面部关键点列表

        返回:
            float: 嘴部开合度，值越大表示嘴张得越大
        """
        # 获取嘴部上下关键点坐标
        upper_lip = landmarks[self.upper_lip]  # 上唇中点
        lower_lip = landmarks[self.lower_lip]  # 下唇中点

        # 计算上下唇之间的垂直距离作为开合度
        mouth_distance = abs(upper_lip.y - lower_lip.y)
        return mouth_distance

    def calculate_eye_openness(self, landmarks):
        """
        计算眼睛开合程度 (EAR - Eye Aspect Ratio)
        使用标准的EAR公式：EAR = (A + B) / (2.0 * C)
        其中A和B是垂直距离，C是水平距离

        参数:
            landmarks: MediaPipe检测到的面部关键点列表

        返回:
            float: 眼睛开合度，值越小表示眼睛闭得越紧
        """
        # MediaPipe 468点面部网格中眼睛的标准关键点索引
        # 左眼关键点 (按EAR公式顺序)
        left_eye_points = [
            landmarks[33],   # 左眼外角
            landmarks[160],  # 左眼上眼睑中点
            landmarks[158],  # 左眼内角
            landmarks[133],  # 左眼下眼睑中点
            landmarks[153],  # 左眼外角下方
            landmarks[144]   # 左眼内角下方
        ]
        
        # 右眼关键点 (按EAR公式顺序)
        right_eye_points = [
            landmarks[362],  # 右眼外角
            landmarks[387],  # 右眼上眼睑中点
            landmarks[263],  # 右眼内角
            landmarks[373],  # 右眼下眼睑中点
            landmarks[380],  # 右眼外角下方
            landmarks[374]   # 右眼内角下方
        ]
        
        # 计算左眼EAR
        left_ear = self._calculate_ear(left_eye_points)
        
        # 计算右眼EAR
        right_ear = self._calculate_ear(right_eye_points)
        
        # 返回两只眼睛的平均EAR值
        return (left_ear + right_ear) / 2.0
    
    def _calculate_ear(self, eye_points):
        """
        计算单只眼睛的EAR值
        
        参数:
            eye_points: 眼睛的6个关键点列表
            
        返回:
            float: EAR值
        """
        # 计算垂直距离A (上眼睑中点到下眼睑中点)
        A = np.sqrt((eye_points[1].x - eye_points[3].x)**2 + 
                   (eye_points[1].y - eye_points[3].y)**2)
        
        # 计算垂直距离B (外角下方到内角下方)
        B = np.sqrt((eye_points[4].x - eye_points[5].x)**2 + 
                   (eye_points[4].y - eye_points[5].y)**2)
        
        # 计算水平距离C (外角到内角)
        C = np.sqrt((eye_points[0].x - eye_points[2].x)**2 + 
                   (eye_points[0].y - eye_points[2].y)**2)
        
        # 计算EAR值
        if C > 0:
            ear = (A + B) / (2.0 * C)
        else:
            ear = 0
            
        return ear

    def save_action_frame(self,action_type,frame,frame_idx):
        # 创建动作类型的子目录（如果不存在）
        action_dir = str(os.path.join(self.output_dir, action_type))
        os.makedirs(action_dir, exist_ok=True)

        # 生成文件名，使用时间戳作为前缀
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{timestamp}_{action_type}_{frame_idx:04d}.jpg"

        # 保存图片
        output_path = os.path.join(action_dir, filename)
        cv2.imwrite(output_path, frame)

    def detect_movement(self,image, landmarks,frame_idx,):
        """
           检测视频帧中的头部动作

           参数:
               image: 输入的视频帧图像(rgb格式的）
               frame_idx: 当前帧的索引号

           返回:
               tuple: (movement, image)
               movement: 检测到的动作信息，格式为(action_type, magnitude)或None
               image: 绘制了关键点和状态信息的图像
       """
        #当前动作类型
        current_action_list = []

        # 计算当前帧的头部姿态
        pitch, yaw = self.calculate_head_pose(landmarks)

        # 计算嘴部和眼睛的开合程度
        mouth_openness = self.calculate_mouth_openness(landmarks)
        eye_openness = self.calculate_eye_openness(landmarks)

        print(f"当前第{frame_idx}帧图像的状态==》Pitch: {pitch:.2f}-Yaw:{yaw:.2f}-Mouth:{mouth_openness:.3f}-Eyes:{eye_openness:.3f}")

        # 在图像上显示各种状态值
        cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Mouth: {mouth_openness:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Eyes: {eye_openness:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #计算pitch和yaw的变化度
        #计算点头幅度（使用左眉毛与左右脸轮廓距离）
        eyebrow_to_chin, eyebrow_to_left_contour, eyebrow_to_right_contour = self.calculate_eyebrow_to_face_contour_distance(landmarks)
        
        # 更新历史数据用于点头检测
        if not hasattr(self, 'eyebrow_contour_history'):
            self.eyebrow_contour_history = []
            self.initial_eyebrow_to_chin = None
            self.initial_eyebrow_to_sides_ratio = None
            
        # 计算眉毛到两侧的平均距离
        eyebrow_to_sides_avg = (eyebrow_to_left_contour + eyebrow_to_right_contour) / 2
        
        self.eyebrow_contour_history.append({
            'eyebrow_to_chin': eyebrow_to_chin,
            'eyebrow_to_left_contour': eyebrow_to_left_contour,
            'eyebrow_to_right_contour': eyebrow_to_right_contour,
            'eyebrow_to_sides_avg': eyebrow_to_sides_avg
        })
        if len(self.eyebrow_contour_history) > 10:
            self.eyebrow_contour_history.pop(0)
            
        # 设置初始值
        if self.initial_eyebrow_to_chin is None and len(self.eyebrow_contour_history) >= 5:
            recent_chin_distances = [h['eyebrow_to_chin'] for h in self.eyebrow_contour_history[-5:]]
            recent_sides_ratios = [h['eyebrow_to_chin'] / h['eyebrow_to_sides_avg'] for h in self.eyebrow_contour_history[-5:]]
            
            self.initial_eyebrow_to_chin = sum(recent_chin_distances) / len(recent_chin_distances)
            self.initial_eyebrow_to_sides_ratio = sum(recent_sides_ratios) / len(recent_sides_ratios)
            
        # 计算点头变化
        pitch_change = 0
        if self.initial_eyebrow_to_chin is not None:
            # 计算当前眉毛到下巴距离的变化
            current_chin_distance = sum([h['eyebrow_to_chin'] for h in self.eyebrow_contour_history[-3:]]) / 3
            
            # 计算变化量
            chin_change = current_chin_distance - self.initial_eyebrow_to_chin
            
            # 检查人脸是否在画面中间（通过yaw角度判断）
            face_center_threshold = 15  # 人脸偏离中心的阈值（度）
            is_face_centered = abs(yaw) < face_center_threshold
            
            # 根据人脸位置调整阈值
            if is_face_centered:
                threshold = 0.02  # 人脸在中间时使用较小阈值
            else:
                threshold = 0.04  # 人脸偏左或偏右时使用较大阈值，减少误检
            
            # 简化的点头检测逻辑：基于眉毛到下巴距离的变化
            print("chin_change:",chin_change)
            if abs(chin_change) > threshold:
                if chin_change < 0:  # 眉毛到下巴距离减小 = 低头
                    pitch_change = -1
                elif chin_change > 0:  # 眉毛到下巴距离增大 = 抬头
                    pitch_change = 1

            
        #计算摇头幅度（使用鼻子到面部轮廓距离）
        left_nose_distance, right_nose_distance = self.calculate_nose_to_face_contour_distance(landmarks)
        nose_distance_ratio = left_nose_distance / right_nose_distance if right_nose_distance > 0.001 else 1.0
        
        # 更新历史数据用于摇头检测
        if not hasattr(self, 'nose_contour_history'):
            self.nose_contour_history = []
            self.initial_nose_ratio = None
            
        self.nose_contour_history.append(nose_distance_ratio)
        if len(self.nose_contour_history) > 10:
            self.nose_contour_history.pop(0)
            
        # 设置初始鼻子距离比值
        if self.initial_nose_ratio is None and len(self.nose_contour_history) >= 5:
            self.initial_nose_ratio = sum(self.nose_contour_history[-5:]) / 5
            
        # 计算鼻子距离比值变化
        yaw_change = 0
        if self.initial_nose_ratio is not None:
            yaw_change = nose_distance_ratio - self.initial_nose_ratio

        # 检测摇头动作
        print("yaw_change:",yaw_change)
        if abs(yaw_change) > self.yaw_threshold:  # 使用专门的摇头阈值
            if yaw_change < 0:  # 鼻子距离比值减小，表示左摇头
                current_action_list.append("yaw_left")
                if "yaw_left" in self.last_action_list:
                    self.num_yaw_left = self.num_yaw_left + 1
                    if self.num_yaw_left > self.yaw_num_threshold:
                        self.save_action_frame("yaw_left",image,frame_idx)
                        print(f"第{frame_idx}检测到左摇头,已存入文件夹")
                        self.num_yaw_left = 0

            else:#鼻子距离比值增大，表示右摇头
                current_action_list.append("yaw_right")
                if "yaw_right" in self.last_action_list:
                    self.num_yaw_right = self.num_yaw_right + 1
                    if self.num_yaw_right > self.yaw_num_threshold:
                        self.save_action_frame("yaw_right",image,frame_idx)
                        print(f"第{frame_idx}检测到右摇头,已存入文件夹")
                        self.num_yaw_right = 0

        # 检测点头动作
        pitch_change = pitch - self.initial_pitch
        print("pitch_change:", pitch_change)
        if abs(pitch_change) > self.pitch_threshold:
        # if pitch_change != 0:  # 有点头或抬头动作
            if pitch_change < 0:  # 俯仰角增大，表示抬头
                current_action_list.append("pitch_up")
                if "pitch_up" in self.last_action_list:
                    self.num_pitch_up = self.num_pitch_up + 1
                    if self.num_pitch_up > self.pitch_num_threshold:
                        self.save_action_frame("pitch_up", image, frame_idx)
                        print(f"第{frame_idx}检测到抬头,已存入文件夹")
                        self.num_pitch_up = 0
            else:# 俯仰角减小，表示低头
                current_action_list.append("pitch_down")
                if "pitch_down" in self.last_action_list:
                    self.num_pitch_down = self.num_pitch_down + 1
                    if self.num_pitch_down > self.pitch_num_threshold:
                        self.save_action_frame("pitch_down", image, frame_idx)
                        print(f"第{frame_idx}检测到低头,已存入文件夹")
                        self.num_pitch_down = 0

        # 检测张嘴动作
        if mouth_openness > self.mouth_open_threshold:
            current_action_list.append("mouth_open")
            if "mouth_open" in self.last_action_list:
                self.num_mouth_open = self.num_mouth_open + 1
                if self.num_mouth_open > self.mouth_num_threshold:
                    self.save_action_frame("mouth_open", image, frame_idx)
                    print(f"第{frame_idx}检测到张嘴,已存入文件夹")
                    self.num_mouth_open = 0


        # 检测眨眼动作
        if eye_openness < self.eye_closed_threshold:
            current_action_list.append("eye_close")
            self.save_action_frame("eye_close", image, frame_idx)
            print(f"第{frame_idx}检测到眨眼,已存入文件夹")

            
        self.last_action_list = current_action_list

        return current_action_list,image

    def calculate_nose_to_face_contour_distance(self, landmarks):
        """
        计算鼻子到面部轮廓的距离，用于摇头检测
        
        参数:
            landmarks: MediaPipe检测到的面部关键点列表
            
        返回:
            tuple: (left_distance, right_distance) 鼻子到左右面部轮廓的距离
        """
        # 鼻子关键点
        nose_tip = landmarks[1]  # 鼻尖
        
        # 面部轮廓关键点（左右对称）
        # 左脸颊轮廓点
        left_cheek_contour = [
            landmarks[50],   # 左脸颊
            landmarks[123],  # 左脸颊轮廓
            landmarks[116],  # 左脸颊轮廓
            landmarks[117],  # 左脸颊轮廓
            landmarks[118],  # 左脸颊轮廓
            landmarks[119],  # 左脸颊轮廓
            landmarks[120],  # 左脸颊轮廓
            landmarks[121],  # 左脸颊轮廓
            landmarks[122],  # 左脸颊轮廓
        ]
        
        # 右脸颊轮廓点
        right_cheek_contour = [
            landmarks[280],  # 右脸颊
            landmarks[352],  # 右脸颊轮廓
            landmarks[345],  # 右脸颊轮廓
            landmarks[346],  # 右脸颊轮廓
            landmarks[347],  # 右脸颊轮廓
            landmarks[348],  # 右脸颊轮廓
            landmarks[349],  # 右脸颊轮廓
            landmarks[350],  # 右脸颊轮廓
            landmarks[351],  # 右脸颊轮廓
        ]
        
        # 计算鼻子到左脸颊轮廓的最小距离
        left_distances = []
        for point in left_cheek_contour:
            distance = np.sqrt((nose_tip.x - point.x)**2 + (nose_tip.y - point.y)**2)
            left_distances.append(distance)
        left_min_distance = min(left_distances)
        
        # 计算鼻子到右脸颊轮廓的最小距离
        right_distances = []
        for point in right_cheek_contour:
            distance = np.sqrt((nose_tip.x - point.x)**2 + (nose_tip.y - point.y)**2)
            right_distances.append(distance)
        right_min_distance = min(right_distances)
        
        return left_min_distance, right_min_distance
    
    def detect_yaw_by_nose_contour_distance(self, landmarks):
        """
        基于鼻子到面部轮廓距离的摇头检测
        
        参数:
            landmarks: MediaPipe检测到的面部关键点列表
            
        返回:
            str: 检测到的摇头动作类型
        """
        # 计算当前帧的鼻子到面部轮廓距离
        left_distance, right_distance = self.calculate_nose_to_face_contour_distance(landmarks)
        
        # 计算左右距离的比值（用于判断头部朝向）
        distance_ratio = left_distance / right_distance if right_distance > 0.001 else 1.0
        
        # 更新历史数据
        if not hasattr(self, 'nose_contour_history'):
            self.nose_contour_history = []
            self.initial_distance_ratio = None
            
        self.nose_contour_history.append({
            'left_distance': left_distance,
            'right_distance': right_distance,
            'distance_ratio': distance_ratio
        })
        
        # 保持历史窗口大小
        if len(self.nose_contour_history) > 10:
            self.nose_contour_history.pop(0)
            
        # 初始化阶段
        if not self.is_initialized:
            return None
            
        # 设置初始距离比值（如果还没有设置）
        if self.initial_distance_ratio is None:
            recent_ratios = [h['distance_ratio'] for h in self.nose_contour_history[-5:]]
            self.initial_distance_ratio = sum(recent_ratios) / len(recent_ratios)
            return None
            
        # 计算距离比值的变化
        ratio_change = distance_ratio - self.initial_distance_ratio
        
        # 检测摇头动作
        if abs(ratio_change) > 0.1:  # 阈值可调整
            if ratio_change > 0:  # 左距离增大，右距离减小，表示头部向左转
                return "yaw_left"
            else:  # 右距离增大，左距离减小，表示头部向右转
                return "yaw_right"
                
        return None

    def calculate_eyebrow_to_face_contour_distance(self, landmarks):
        """
        计算左眉毛与下巴、左右脸轮廓的距离，用于点头检测
        
        参数:
            landmarks: MediaPipe检测到的面部关键点列表
            
        返回:
            tuple: (eyebrow_to_chin, eyebrow_to_left_contour, eyebrow_to_right_contour)
        """
        # 左眉毛关键点 - 使用更准确的眉毛中心点
        left_eyebrow_center = landmarks[66]  # 左眉毛中心点（更准确）
        
        # 下巴关键点
        chin = landmarks[152]  # 下巴
        
        # 左脸轮廓关键点
        left_face_contour = [
            landmarks[50],   # 左脸颊
            landmarks[123],  # 左脸颊轮廓
            landmarks[116],  # 左脸颊轮廓
            landmarks[117],  # 左脸颊轮廓
            landmarks[118],  # 左脸颊轮廓
            landmarks[119],  # 左脸颊轮廓
            landmarks[120],  # 左脸颊轮廓
            landmarks[121],  # 左脸颊轮廓
            landmarks[122],  # 左脸颊轮廓
        ]
        
        # 右脸轮廓关键点
        right_face_contour = [
            landmarks[280],  # 右脸颊
            landmarks[352],  # 右脸颊轮廓
            landmarks[345],  # 右脸颊轮廓
            landmarks[346],  # 右脸颊轮廓
            landmarks[347],  # 右脸颊轮廓
            landmarks[348],  # 右脸颊轮廓
            landmarks[349],  # 右脸颊轮廓
            landmarks[350],  # 右脸颊轮廓
            landmarks[351],  # 右脸颊轮廓
        ]
        
        # 计算左眉毛到下巴的距离
        eyebrow_to_chin = np.sqrt((left_eyebrow_center.x - chin.x)**2 + (left_eyebrow_center.y - chin.y)**2)
        
        # 计算左眉毛到左脸轮廓的最小距离
        left_eyebrow_to_left_distances = []
        for point in left_face_contour:
            distance = np.sqrt((left_eyebrow_center.x - point.x)**2 + (left_eyebrow_center.y - point.y)**2)
            left_eyebrow_to_left_distances.append(distance)
        eyebrow_to_left_contour = min(left_eyebrow_to_left_distances)
        
        # 计算左眉毛到右脸轮廓的最小距离
        left_eyebrow_to_right_distances = []
        for point in right_face_contour:
            distance = np.sqrt((left_eyebrow_center.x - point.x)**2 + (left_eyebrow_center.y - point.y)**2)
            left_eyebrow_to_right_distances.append(distance)
        eyebrow_to_right_contour = min(left_eyebrow_to_right_distances)
        
        return eyebrow_to_chin, eyebrow_to_left_contour, eyebrow_to_right_contour

    def process_video(self, video_path, output_dir):
        """
        处理视频并保存检测到的动作截图

        参数:
            video_path: 输入视频文件的路径
            output_dir: 输出截图的保存目录
        """
        print(f"开始处理视频: {video_path}")
        print(f"输出目录: {output_dir}")

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir  # 保存输出目录路径

        # 打开视频文件
        if str(video_path).lower() in ["0", "camera"]:
            cap = cv2.VideoCapture(0)
            print("打开摄像头实时检测")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"打开视频文件: {video_path}")

        if not cap.isOpened():
            print(f"错误：无法打开视频文件: {video_path}")
            return

        # 获取视频总帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频总帧数: {frame_count}")

        # 逐帧处理视频
        frame_idx = 0
        while True:
            try:
                # 读取一帧
                ret, frame = cap.read()
                if not ret:
                    print(f"视频处理完成，共处理 {frame_idx} 帧")
                    break
            
                # 将BGR图像转换为RGB格式（MediaPipe要求）
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #检测图像中的人脸，并抽取人脸的关键点，没有就返回None
                results = self.face_mesh.process(image_rgb)

                # 如果没有检测到人脸，就进行下一帧处理
                if not results.multi_face_landmarks:
                    frame_idx += 1
                    self.start_index = frame_idx
                    continue

                # 获取检测到的面部关键点
                landmarks = results.multi_face_landmarks[0].landmark

                # 在图像上绘制面部关键点和网格
                self.mp_drawing.draw_landmarks(
                    image=image_rgb,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                #如果是正脸初始帧，先进行正脸确认，也就是确认正脸的初始的yaw和pitch，方便后续的摇头检测
                if frame_idx < self.initialization_frames + self.start_index:
                    # 计算当前帧的头部姿态
                    pitch, yaw = self.calculate_head_pose(landmarks)
                    self.init_pitch_values.append(pitch)
                    self.init_yaw_values.append(yaw)
                    print(f"将第{frame_idx}帧纳入初始化")

                #如果不是初始帧，则计算正脸的yaw和pitch，并开始检测动作
                else:
                    self.initial_pitch = sum(self.init_pitch_values) / len(self.init_pitch_values)
                    self.initial_yaw = sum(self.init_yaw_values) / len(self.init_yaw_values)
                    self.is_initialized = True
                    print(f"初始化完成 - 初始pitch: {self.initial_pitch:.2f}, 初始yaw: {self.initial_yaw:.2f}")

                    # 检测当前帧的动作
                    current_action_list, image = self.detect_movement(image_bgr,landmarks,frame_idx)
                    print(f"第{frame_idx}帧的动作列表为{current_action_list}")

                    # 显示实时处理画面
                    cv2.imshow('Head Movement Detection', image_bgr)

                    # 按 'q' 键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户手动退出")
                        break

                frame_idx += 1
                # if frame_idx % 20 == 0:
                #     cv2.waitKey(0)

            except Exception as e:
                print(f"处理第 {frame_idx} 帧时发生错误: {str(e)}")
                continue

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n处理完成！")

if __name__ == "__main__":
    try:
        # 让用户选择输入类型
        # input_type = input("输入 'video' 处理视频文件，输入 'camera' 进行实时检测: ").strip().lower()
        # if input_type == "camera":
        #     video_path = "0"
        # else:
        #     video_path = "test4.mp4"  # 或让用户输入文件名
        video_path = "0"

        output_dir = "head_movement_screenshots"
        print(f"程序开始运行...")
        print(f"输入: {video_path}")
        print(f"输出目录: {output_dir}")

        # 创建检测器实例并处理视频
        detector = HeadMovementDetector()
        detector.process_video(video_path, output_dir)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

