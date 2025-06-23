import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import os

class HeadMovementDetector:
    def __init__(self, use_depth=True):
        """初始化头部动作检测器"""
        # 初始化MediaPipe面部网格检测器
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 深度相机支持
        self.use_depth = use_depth
        if self.use_depth:
            try:
                # 尝试初始化深度相机
                self.depth_camera = cv2.VideoCapture(1)  # 假设深度相机是第二个设备
                self.depth_scale = 0.001  # 深度单位转换（毫米到米）
                print("深度相机初始化成功")
            except Exception as e:
                print(f"深度相机初始化失败: {e}")
                self.use_depth = False
        
        # 滑动窗口设置
        self.window_size = 5  # 滑动窗口大小，存储最近5帧的数据
        self.pitch_history = []  # 存储最近N帧的pitch值
        self.yaw_history = []    # 存储最近N帧的yaw值
        
        # 动作状态跟踪
        self.current_action = None  # 当前正在进行的动作
        self.action_frames = 0      # 当前动作持续的帧数
        self.min_frames_threshold = 3  # 最小连续帧数阈值
        
        # 初始状态记录
        self.initial_pitch = None  # 初始pitch值
        self.initial_yaw = None    # 初始yaw值
        self.is_initialized = False  # 是否已初始化
        self.initialization_frames = 10  # 初始化所需的帧数
        self.init_pitch_values = []  # 初始化期间收集的pitch值
        self.init_yaw_values = []    # 初始化期间收集的yaw值
        
        # 动作阈值设置
        self.movement_threshold = 1.2  # 点头动作阈值（度）
        self.yaw_threshold = 1.2      # 摇头动作阈值（度）
        self.eye_closed_threshold = 0.18  # 眼睛闭合阈值
        self.mouth_open_threshold = 0.05  # 张嘴阈值
        
        # 眼睛闭合检测
        self.min_eye_closed_frames = 2  # 最小眼睛闭合帧数
        self.eye_closed_frames = 0      # 当前眼睛闭合持续帧数
        
        # 动作状态
        self.last_pitch = None
        self.last_yaw = None
        self.last_action = None
        self.action_start_frame = None
        self.action_start_pitch = None
        self.action_start_yaw = None
        
        # 相机参数
        self.size = (640, 480)  # 图像大小
        self.focal_length = self.size[1]
        self.center = (self.size[1]/2, self.size[0]/2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.center[0]],
             [0, self.focal_length, self.center[1]],
             [0, 0, 1]], dtype="double"
        )
        self.dist_coeffs = np.zeros((4,1))  # 假设没有镜头畸变
        
        # 嘴部和眼睛的关键点索引
        self.mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
        self.left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # 状态跟踪
        self.last_eye_state = 'open'  # 记录上一帧眼睛状态
        self.last_mouth_state = 'closed'  # 记录上一帧嘴巴状态
        
        # 新增：截图保存目录
        self.save_dir = 'head_movement_screenshots_v1.3'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def calculate_focal_length(self, depth_values):
        """根据深度信息计算焦距"""
        if not depth_values:
            return self.focal_length
            
        avg_depth = np.mean(depth_values)
        # 根据深度调整焦距，深度越大，焦距越小
        return self.focal_length * (1.0 + (avg_depth - 500) / 1000)

    def get_depth_at_point(self, depth_frame, x, y):
        """获取指定点的深度值"""
        if depth_frame is None:
            return None
            
        try:
            # 确保坐标在有效范围内
            x = int(max(0, min(x, depth_frame.shape[1]-1)))
            y = int(max(0, min(y, depth_frame.shape[0]-1)))
            return depth_frame[y, x]
        except:
            return None

    def calculate_head_pose(self, landmarks, depth_frame=None):
        """使用深度信息计算头部姿态"""
        # 获取关键点坐标和深度值
        image_points = []
        depth_values = []
        
        for point in [33, 8, 36, 45, 48, 54]:  # 关键点索引
            x, y = landmarks[point]
            # 获取该点的深度值
            if depth_frame is not None:
                depth = self.get_depth_at_point(depth_frame, x, y)
                depth_values.append(depth if depth is not None else 0)
            image_points.append([x, y])
        
        image_points = np.array(image_points, dtype="double")
        
        if depth_frame is not None and len(depth_values) == 6 and all(depth_values):
            # 使用深度信息构建3D点
            model_points = np.array([
                (0.0, 0.0, depth_values[0]),          # 鼻尖
                (0.0, -330.0, depth_values[1]),       # 下巴
                (-225.0, 170.0, depth_values[2]),     # 左眼左角
                (225.0, 170.0, depth_values[3]),      # 右眼右角
                (-150.0, -150.0, depth_values[4]),    # 左嘴角
                (150.0, -150.0, depth_values[5])      # 右嘴角
            ])
            
            # 根据深度信息调整相机内参
            focal_length = self.calculate_focal_length(depth_values)
            camera_matrix = np.array(
                [[focal_length, 0, self.center[0]],
                 [0, focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )
        else:
            # 使用默认3D模型点
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])
            camera_matrix = self.camera_matrix
        
        # 使用PnP算法求解
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, 
            image_points, 
            camera_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE  # 使用迭代方法提高精度
        )
        
        # 将旋转向量转换为欧拉角
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        return euler_angles[0, 0], euler_angles[1, 0], euler_angles[2, 0]

    def update_history(self, pitch, yaw):
        """更新滑动窗口历史记录"""
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        if len(self.pitch_history) > self.window_size:
            self.pitch_history.pop(0)
            self.yaw_history.pop(0)

    def initialize_state(self, pitch, yaw):
        """初始化头部姿态状态"""
        if not self.is_initialized:
            self.init_pitch_values.append(pitch)
            self.init_yaw_values.append(yaw)
            
            if len(self.init_pitch_values) >= self.initialization_frames:
                # 计算初始状态的平均值
                self.initial_pitch = sum(self.init_pitch_values) / len(self.init_pitch_values)
                self.initial_yaw = sum(self.init_yaw_values) / len(self.init_yaw_values)
                self.is_initialized = True
                print(f"初始化完成 - 初始pitch: {self.initial_pitch:.2f}, 初始yaw: {self.initial_yaw:.2f}")
                return True
        return False

    def calculate_mouth_openness(self, landmarks):
        """计算嘴部开合度"""
        mouth_points = [landmarks[i] for i in self.mouth_indices]
        mouth_points = np.array(mouth_points)
        
        # 计算嘴部高度和宽度
        mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
        mouth_width = np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0])
        
        # 计算开合度（高度/宽度）
        return mouth_height / mouth_width

    def calculate_eye_openness(self, landmarks):
        """计算眼睛开合度"""
        # 计算左眼开合度
        left_eye_points = [landmarks[i] for i in self.left_eye_indices]
        left_eye_points = np.array(left_eye_points)
        left_eye_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
        left_eye_width = np.max(left_eye_points[:, 0]) - np.min(left_eye_points[:, 0])
        left_eye_ratio = left_eye_height / left_eye_width
        
        # 计算右眼开合度
        right_eye_points = [landmarks[i] for i in self.right_eye_indices]
        right_eye_points = np.array(right_eye_points)
        right_eye_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
        right_eye_width = np.max(right_eye_points[:, 0]) - np.min(right_eye_points[:, 0])
        right_eye_ratio = right_eye_height / right_eye_width
        
        # 返回两只眼睛的平均开合度
        return (left_eye_ratio + right_eye_ratio) / 2

    def detect_movement(self, pitch, yaw):
        """检测头部动作，基于与初始状态的相对变化"""
        if not self.is_initialized:
            if self.initialize_state(pitch, yaw):
                self.last_pitch = self.initial_pitch
                self.last_yaw = self.initial_yaw
            return None

        # 计算与初始状态的相对变化
        pitch_change = pitch - self.initial_pitch
        yaw_change = yaw - self.initial_yaw
        
        # 更新历史记录
        self.update_history(pitch, yaw)
        
        # 存储检测到的所有动作
        detected_actions = []
        
        # 检测点头动作（相对于初始状态）
        if abs(pitch_change) > self.movement_threshold:
            # 检查是否在正面位置（yaw接近初始值）
            if abs(yaw_change) < 0.5:  # 允许0.5度的偏差
                if pitch_change > 0:
                    detected_actions.append(("pitch_up", abs(pitch_change)))
                else:
                    detected_actions.append(("pitch_down", abs(pitch_change)))
            else:
                # 如果不在正面位置，使用较小的阈值
                if abs(pitch_change) > self.movement_threshold * 1.5:  # 增加阈值
                    if pitch_change > 0:
                        detected_actions.append(("pitch_up", abs(pitch_change)))
                    else:
                        detected_actions.append(("pitch_down", abs(pitch_change)))
        
        # 检测摇头动作（相对于初始状态）
        if abs(yaw_change) > self.yaw_threshold:
            if yaw_change > 0:
                detected_actions.append(("yaw_right", abs(yaw_change)))
            else:
                detected_actions.append(("yaw_left", abs(yaw_change)))
        
        # 根据优先级选择最终动作
        movement = None
        if detected_actions:
            # 优先选择摇头动作
            yaw_actions = [a for a in detected_actions if a[0].startswith('yaw_')]
            if yaw_actions:
                movement = max(yaw_actions, key=lambda x: x[1])[0]
            else:
                # 如果没有摇头动作，选择点头动作
                pitch_actions = [a for a in detected_actions if a[0].startswith('pitch_')]
                if pitch_actions:
                    movement = max(pitch_actions, key=lambda x: x[1])[0]
        
        # 更新动作状态
        if movement != self.current_action:
            self.current_action = movement
            self.action_frames = 1
            self.action_start_frame = len(self.pitch_history) - 1
            self.action_start_pitch = pitch
            self.action_start_yaw = yaw
        else:
            self.action_frames += 1
        
        # 只有当动作持续足够帧数时才返回
        if self.action_frames >= self.min_frames_threshold:
            return movement
        
        return None

    def draw_action_info(self, frame, movement, eye_openness, mouth_openness, pitch=None, yaw=None, roll=None):
        """在图像上显示动作信息，风格与v1.2一致"""
        # 显示当前动作
        action_str = f"Action: {movement[0] if movement else 'None'}"
        cv2.putText(frame, action_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 显示pitch/yaw/roll
        if pitch is not None and yaw is not None and roll is not None:
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 显示眼睛开合度
        cv2.putText(frame, f"Eyes: {eye_openness:.3f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 显示嘴巴开合度
        cv2.putText(frame, f"Mouth: {mouth_openness:.3f}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def process_frame(self, frame, depth_frame=None):
        """处理单帧图像，显示风格与v1.2一致"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = self.convert_landmarks_to_array(results.multi_face_landmarks[0])
            pitch, yaw, roll = self.calculate_head_pose(landmarks, depth_frame)
            movement = self.detect_movement(pitch, yaw)
            eye_openness = self.calculate_eye_openness(landmarks)
            print(f"帧 {self.frame_count} eye_openness: {eye_openness:.3f}")
            if eye_openness < self.eye_closed_threshold:
                self.eye_closed_frames += 1
                if self.eye_closed_frames >= self.min_eye_closed_frames:
                    movement = ('eyes_closed', eye_openness)
            else:
                self.eye_closed_frames = 0
            mouth_openness = self.calculate_mouth_openness(landmarks)
            if mouth_openness > self.mouth_open_threshold:
                if self.last_mouth_state == 'closed':
                    movement = ('mouth_open', mouth_openness)
                self.last_mouth_state = 'open'
            else:
                self.last_mouth_state = 'closed'
            if movement:
                self.save_action_frame(frame, movement)
            # 绘制面部关键点（仿v1.2）
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
            )
            # 显示信息
            self.draw_action_info(frame, movement, eye_openness, mouth_openness, pitch, yaw, roll)
            self.frame_count += 1
        return frame

    def convert_landmarks_to_array(self, landmarks):
        """将MediaPipe landmarks转换为numpy数组"""
        return np.array([[lm.x * self.size[0], lm.y * self.size[1]] for lm in landmarks.landmark])

    def save_action_frame(self, frame, movement):
        """保存动作帧"""
        if isinstance(movement, tuple):
            action_type, magnitude = movement
        else:
            action_type = movement
            magnitude = 0.0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{action_type}_{self.frame_count:04d}_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"保存了{action_type}动作帧: {filepath}")
        print(f"动作幅度: {magnitude:.2f}")

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    detector = HeadMovementDetector(use_depth=True)
    detector.frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 获取深度帧
        if detector.use_depth:
            ret_depth, depth_frame = detector.depth_camera.read()
            if ret_depth:
                # 处理深度帧
                depth_frame = cv2.convertScaleAbs(depth_frame, alpha=0.03)
            else:
                depth_frame = None
        else:
            depth_frame = None
        
        # 处理帧
        processed_frame = detector.process_frame(frame, depth_frame)
        
        # 显示结果
        cv2.imshow('Head Movement Detection', processed_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    if detector.use_depth:
        detector.depth_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 