import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

class HeadMovementDetector:
    """
    头部动作检测器类
    用于检测视频中的头部动作，包括点头、摇头、眨眼和张嘴等动作
    使用MediaPipe面部网格检测器实现面部关键点的检测和跟踪
    """
    def __init__(self):
        """初始化头部动作检测器"""
        # 初始化MediaPipe面部网格检测器
        # max_num_faces: 最大检测人脸数
        # refine_landmarks: 是否使用更精确的关键点检测
        # min_detection_confidence: 最小检测置信度
        # min_tracking_confidence: 最小跟踪置信度
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 初始化绘图工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # 记录上一帧的头部姿态，用于计算动作变化
        self.last_pitch = None  # 上一帧的俯仰角
        self.last_yaw = None    # 上一帧的偏航角
        
        # 动作检测阈值设置
        self.movement_threshold = 1.0  # 头部动作幅度阈值，用于判断是否发生点头或摇头
        
        # MediaPipe 468点面部网格的关键点索引定义
        # 嘴部轮廓关键点：用于检测张嘴动作
        self.mouth_indices = [61, 291, 0, 17, 291, 405, 17, 314, 405, 314, 17, 291, 405, 314]
        
        # 左眼轮廓关键点：用于检测眨眼动作
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # 右眼轮廓关键点：用于检测眨眼动作
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # 头部姿态计算的关键点索引
        self.nose_tip = 1        # 鼻尖：用于计算头部姿态
        self.left_eye_center = 33    # 左眼中心：用于计算头部姿态
        self.right_eye_center = 263  # 右眼中心：用于计算头部姿态
        self.chin = 152         # 下巴：用于计算头部姿态
        
        # 嘴部开合度计算的关键点索引
        self.upper_lip = 13     # 上唇中点：用于计算嘴部开合度
        self.lower_lip = 14     # 下唇中点：用于计算嘴部开合度
        
        # 状态记录
        self.last_eye_state = 'open'     # 记录上一帧眼睛状态（开/闭）
        self.last_mouth_state = 'closed'  # 记录上一帧嘴巴状态（开/闭）

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
        计算眼睛开合程度
        
        参数:
            landmarks: MediaPipe检测到的面部关键点列表
            
        返回:
            float: 眼睛开合度，值越小表示眼睛闭得越紧
        """
        # 获取眼睛上下关键点坐标
        left_eye_upper = landmarks[self.left_eye_indices[1]]  # 左眼上眼睑中点
        left_eye_lower = landmarks[self.left_eye_indices[5]]  # 左眼下眼睑中点
        right_eye_upper = landmarks[self.right_eye_indices[1]]  # 右眼上眼睑中点
        right_eye_lower = landmarks[self.right_eye_indices[5]]  # 右眼下眼睑中点
        
        # 计算左右眼的高度（上下眼睑之间的距离）
        left_height = abs(left_eye_upper.y - left_eye_lower.y)
        right_height = abs(right_eye_upper.y - right_eye_lower.y)
        
        # 计算左右眼的宽度（眼角之间的距离）
        left_width = abs(landmarks[self.left_eye_indices[0]].x - landmarks[self.left_eye_indices[3]].x)
        right_width = abs(landmarks[self.right_eye_indices[0]].x - landmarks[self.right_eye_indices[3]].x)
        
        # 计算眼睛开合比例（高度/宽度）
        # 使用高度与宽度的比值可以消除距离的影响
        left_ratio = left_height / left_width if left_width > 0 else 0
        right_ratio = right_height / right_width if right_width > 0 else 0
        
        # 返回两只眼睛的平均开合比例
        return (left_ratio + right_ratio) / 2

    def calculate_head_pose(self, landmarks):
        """
        计算头部姿态（pitch和yaw）
        
        参数:
            landmarks: MediaPipe检测到的面部关键点列表
            
        返回:
            tuple: (pitch, yaw) 俯仰角和偏航角（单位：度）
            pitch: 正值表示抬头，负值表示低头
            yaw: 正值表示右转，负值表示左转
        """
        # 获取关键点坐标
        nose = landmarks[self.nose_tip]        # 鼻尖
        left_eye = landmarks[self.left_eye_center]   # 左眼中心
        right_eye = landmarks[self.right_eye_center] # 右眼中心
        chin = landmarks[self.chin]            # 下巴

        # 计算头部姿态
        dx = right_eye.x - left_eye.x  # 水平方向的变化
        dy = nose.y - chin.y           # 垂直方向的变化（正值表示抬头）
        dz = nose.z                    # 深度信息

        # 使用arctan2计算角度，可以处理所有象限
        pitch = np.arctan2(dy, dz)  # 俯仰角（点头）
        yaw = np.arctan2(dx, dz)    # 偏航角（摇头）

        # 将弧度转换为角度
        return np.degrees(pitch), np.degrees(yaw)

    def detect_movement(self, image, frame_idx):
        """
        检测视频帧中的头部动作
        
        参数:
            image: 输入的视频帧图像
            frame_idx: 当前帧的索引号
            
        返回:
            tuple: (movement, image)
                movement: 检测到的动作信息，格式为(action_type, magnitude)或None
                image: 绘制了关键点和状态信息的图像
        """
        # 将BGR图像转换为RGB格式（MediaPipe要求）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        # 如果没有检测到人脸，返回None
        if not results.multi_face_landmarks:
            return None, image
            
        # 获取检测到的面部关键点
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 计算当前帧的头部姿态（俯仰角和偏航角）
        pitch, yaw = self.calculate_head_pose(landmarks)
        
        # 计算嘴部和眼睛的开合程度
        mouth_openness = self.calculate_mouth_openness(landmarks)
        eye_openness = self.calculate_eye_openness(landmarks)
        print(f"帧 {frame_idx} eye_openness: {eye_openness:.3f}")  # 调试输出
        
        # 在图像上显示各种状态值
        cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Mouth: {mouth_openness:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Eyes: {eye_openness:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 初始化动作检测结果
        movement = None
        
        # 检测头部动作（点头和摇头）
        if self.last_pitch is not None and self.last_yaw is not None:
            # 计算与上一帧的角度变化
            pitch_change = pitch - self.last_pitch  # 正值表示抬头，负值表示低头
            yaw_change = yaw - self.last_yaw       # 正值表示右转，负值表示左转
            
            # 检测点头动作
            if abs(pitch_change) > self.movement_threshold:
                if pitch_change > 0:  # 俯仰角增大，表示抬头
                    movement = ('pitch_up', pitch_change)
                    print(f"检测到抬头动作: {pitch_change:.2f} (帧 {frame_idx})")
                else:  # 俯仰角减小，表示低头
                    movement = ('pitch_down', abs(pitch_change))
                    print(f"检测到低头动作: {abs(pitch_change):.2f} (帧 {frame_idx})")
            
            # 检测摇头动作
            if abs(yaw_change) > self.movement_threshold:
                if yaw_change < 0:  # 偏航角减小，表示左摇头
                    movement = ('yaw_left', abs(yaw_change))
                    print(f"检测到左摇头动作: {abs(yaw_change):.2f} (帧 {frame_idx})")
                else:  # 偏航角增大，表示右摇头
                    movement = ('yaw_right', yaw_change)
                    print(f"检测到右摇头动作: {yaw_change:.2f} (帧 {frame_idx})")
        
        # 检测张嘴动作（只在闭->开瞬间保存）
        if mouth_openness > 0.05:  # 张嘴阈值
            if self.last_mouth_state == 'closed':
                movement = ('mouth_open', mouth_openness)
                print(f"检测到张嘴动作: {mouth_openness:.3f} (帧 {frame_idx})")
            self.last_mouth_state = 'open'
        else:
            self.last_mouth_state = 'closed'
            
        # 检测眨眼动作（只在开->闭瞬间保存）
        if eye_openness < 0.22:  # 眨眼阈值
            if self.last_eye_state == 'open':
                movement = ('eyes_closed', eye_openness)
                print(f"检测到眨眼动作: {eye_openness:.3f} (帧 {frame_idx})")
            self.last_eye_state = 'closed'
        else:
            self.last_eye_state = 'open'
        
        # 更新上一帧的头部姿态值
        self.last_pitch = pitch
        self.last_yaw = yaw
        
        # 在图像上绘制面部关键点和网格
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.drawing_spec
        )
        
        return movement, image

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
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
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
                
                # 检测当前帧的动作
                movement, image = self.detect_movement(frame, frame_idx)
                
                # 如果检测到动作，保存截图
                if movement is not None:
                    movement_type, change = movement
                    # 生成包含时间戳的文件名
                    filename = f"{movement_type}_{frame_idx:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), image)
                    print(f"保存了{movement_type}动作帧: {filename}")
                    print(f"动作幅度: {change:.2f}")
                
                # 显示实时处理画面
                cv2.imshow('Head Movement Detection', image)
                
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户手动退出")
                    break
                
                frame_idx += 1
                    
            except Exception as e:
                print(f"处理第 {frame_idx} 帧时发生错误: {str(e)}")
                continue
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n处理完成！")

if __name__ == "__main__":
    try:
        # 设置输入输出路径
        video_path = "test2.mp4"  # 输入视频文件
        output_dir = "head_movement_screenshots"  # 输出目录
        
        print(f"程序开始运行...")
        print(f"视频路径: {video_path}")
        print(f"输出目录: {output_dir}")
        
        # 创建检测器实例并处理视频
        detector = HeadMovementDetector()
        detector.process_video(video_path, output_dir)
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()