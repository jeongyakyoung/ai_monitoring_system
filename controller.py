import cv2
import random
from ultralytics import YOLO
import numpy as np
import datetime
import copy
import telegram
import asyncio
from collections import defaultdict
import threading
import os
import sys
from threading import Lock
import atexit
import time

# Load the YOLO model
class Messenger:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Messenger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        # 첫 초기화 시에만 실행
        if not hasattr(self, 'initialized'):
            self.token = None
            self.chat_id = None
            self.initialized = True
            atexit.register(self.cleanup_images)
            
    def set_telegram(self, token, chat_id):
        self.token = token # "7535814762:AAHmB72JRqiRcDxdyHrUoBtbuC573FhyXq0"
        self.chat_id = chat_id # "-4519677286"
        
    def send_photo(self, img_path):
        if not self.token or not self.chat_id:
            raise ValueError("Telegram token or chat_id is not set.")
        
        async def main():
            try:
                bot = telegram.Bot(self.token)
                with open(img_path, 'rb') as photo:
                    await bot.send_photo(chat_id=self.chat_id, photo=photo)
                print(f"이미지 전송 완료: {img_path}")
            except Exception as e:
                print(f"이미지 전송 중 오류 발생: {e}")
            finally:
                time.sleep(1)
                self.remove_file(img_path)
                
        try:    
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(main())
        except Exception as e:
            print(f"비동기 실행 중 오류 발생: {e}")
            self.remove_file(img_path)
    
    def remove_file(self, img_path):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f"파일 삭제 완료: {img_path}")
                    break
            except Exception as e:
                print(f"파일 삭제 중 오류 발생: {e}")
                if attempt < max_attempts - 1:
                    print(f"파일 삭제 실패, 재시도 중... 남은 시도 횟수: {max_attempts - attempt - 1}")
                    time.sleep(0.2)
                else:
                    print(f"파일 삭제 실패, 최대 시도 횟수 도달: {img_path}")
                    
                
    def cleanup_images(self):
        """프로그램 종료 시 img 폴더의 falling 이미지들 정리"""
        try:
            if os.path.exists("img"):
                for file in os.listdir("img"):
                    if file.endswith(".jpg"):
                        file_path = os.path.join("img", file)
                        max_attempts = 3
                        for attempt in range(max_attempts):
                            try:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                                    print(f"Cleanup: 남은 이미지 파일 삭제 - {file}")
                                    break
                            except Exception as e:
                                print(f"Cleanup: 파일 삭제 실패 - {file}: {e}")
                                if attempt < max_attempts - 1:
                                    print(f"파일 삭제 실패, 재시도 중... 남은 시도 횟수: {max_attempts - attempt - 1}")
                                    time.sleep(1.0)
                                else:
                                    print(f"파일 삭제 실패, 최대 시도 횟수 도달: {file_path}")
                                    
        except Exception as e:
            print(f"Cleanup: 정리 중 오류 발생 - {e}")
            
    def send_message(self, text):
        if not self.token or not self.chat_id:
            raise ValueError("Telegram token or chat_id is not set.")
        async def main():
            bot = telegram.Bot(self.token)
            await bot.send_message(chat_id=self.chat_id, text=text)
            
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
        
class Detector:
    def __new__(cls, conf, th, fps):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Detector, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, conf, th, fps) -> None:
        self.telegram = Messenger()
        self.model = None
        self.cap = None
        self.save_img_path = None
        self.save_video_path = None
        
        self.model_conf = conf
        self.model_img_w = 640
        self.model_img_h = 640
        
        self.fps = fps
        self.original_th = th
        self.tracking_th = int(th * fps)
        
        self.font_scale = None
        self.font = None
        self.font_thickness = None
        
        self.origin_img = None
        
        self.track_history = defaultdict(lambda: [])
        self.continuous_count = defaultdict(int) # 연속 카운트를 저장할 딕셔너리
        
        self.file_lock = threading.Lock()
        
        self.set_model()
        self.set_font()
        self.set_skeleton_point_bgr()
        
        # self._make_img_folder()
        
        self.out = None
        # self.set_save_video()
        
        self.join_parts = [
                            (0, 1), (1, 2), (1, 3), (2, 4), (3, 4),    # 얼굴 연결
                            (5, 6),                                     # 어깨 연결
                            (5, 7), (7, 9),                             # 왼팔: 어깨, 팔꿈치, 손목
                            (6, 8), (8, 10),                            # 오른팔: 어깨, 팔꿈치, 손목
                            (11, 12),                                   # 엉덩이 연결
                            (11, 13), (13, 15),                         # 왼다리: 엉덩이, 무릎, 발목
                            (12, 14), (14, 16),                         # 오른다리: 엉덩이, 무릎, 발목
                            ('chest', 11), ('chest', 12),               # 가슴에서 엉덩이로
                            ('chest', 0)                                # 가슴에서 목으로
                            ]
    
    def adjust_tracking_threshold(self, actual_fps):
        self.tracking_th = int(self.original_th * actual_fps)
        
    def change_fps(self, fps):
        self.fps = fps
        
    def change_value(self, ai_model_conf, tracking_th):
        self.model_conf = ai_model_conf
        self.original_th = tracking_th
        self.tracking_th = int(tracking_th * self.fps)
        
    def resource_path(self, relative_path):
        base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

    def set_model(self, model_kind="./yolo11n-pose_safety.pt"):
        # model_dir = './model'
        # os.makedirs(model_dir, exist_ok=True)
        # model_name = "yolo11n-pose.pt"
        # model_path = self.resource_path(os.path.join(model_dir, model_name))
        model_path = self.resource_path(model_kind)
        self.model = YOLO(model_path)
    
    def set_font(self):
        self.font_scale = 0.5
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_thickness = 2
    
    def draw_falling_result(self, img, x1, y1, x2, y2):
        label = "Falling"
                    
        text_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        text_w, text_h = text_size
    
        # 텍스트 배경 사각형 그리기
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)

        # 텍스트 그리기
        cv2.putText(img, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # 위험한 사람에 대해 빨간색 바운딩 박스 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    def draw_normal_result(self, img, x1, y1, x2, y2):
        label = "Normal"
        text_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        text_w, text_h = text_size
        # 텍스트 배경 사각형 그리기
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)

        # 텍스트 그리기
        cv2.putText(img, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # 정상 상태일 때 녹색 바운딩 박스 표시
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    def set_skeleton_point_bgr(self):
        self.nose_bgr = (255, 0, 0)
        self.left_pelvis_bgr = (0, 255, 0)
        self.right_pelvis_bgr = (0, 0, 255)
        self.left_knee_bgr = (255, 255, 0)
        self.right_knee_bgr = (0, 255, 255)
        
    def draw_skeleton(self, img, point_x, point_y, color):
        cv2.circle(img, (point_x, point_y), 5, color, -1, cv2.LINE_AA)
    
    def _change_fps_to_second(self, fps, th):
        result = int(fps * th)
        return result
    
    def tracker_update(self, frame, track_id, x1, y1, x2, y2, telegram_flag):
        track = self.track_history[track_id]
        track.append((x1, y1))
        
        # 이전 좌표와 비교
        if len(track) >= 2:
            prev_x, prev_y = track[-2]
            curr_x, curr_y = track[-1]
            
            if abs(curr_x - prev_x) < 30 and abs(curr_y - prev_y) < 30: # 좌표가 비슷한 경우
                self.continuous_count[track_id] += 1
            else:
                self.continuous_count[track_id] = 0
                
            if self.continuous_count[track_id] >= self.tracking_th:
                self.continuous_count[track_id] = 0
                del(self.track_history[track_id])
                img_path = self.save_falling_img(frame, x1, y1, x2, y2)
                full_img_path = self.save_full_img(self.origin_img)
 
                if telegram_flag:
                    telegram_thread_crop = threading.Thread(target=self.telegram.send_photo, args=(img_path,), daemon=True)
                    telegram_thread_crop.start()
                    telegram_thread_full = threading.Thread(target=self.telegram.send_photo, args=(full_img_path,), daemon=True)
                    telegram_thread_full.start()
                    
        if len(track) > 30: # 최대 30개의 좌표만 저장
            track.pop(0)
    
    def predict(self, frame, telegram_flag=True, visualize_falg=True):
        results = self.model.track(frame, conf=self.model_conf, persist=True, verbose=False)
        self.origin_img = copy.deepcopy(frame) #results[0].orig_img
        keypoints = results[0].keypoints
        boxes = results[0].boxes
        
        self.visualize_skeleton_bbox(boxes, keypoints, mode=visualize_falg)
        
        for (keypoint, box) in zip(keypoints, boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            human_width = x2-x1
            human_height = y2-y1
                
            nose_conf = keypoint.conf[0][0].to('cpu').numpy()
            nose = keypoint.xy[0][0].to('cpu').numpy()
            nose_x, nose_y = int(nose[0]), int(nose[1])
            
            
            left_pelvis_conf = keypoint.conf[0][11].to('cpu').numpy()
            left_pelvis = keypoint.xy[0][11].to('cpu').numpy() # 왼쪽 어깨
            left_pelvis_x, left_pelvis_y = int(left_pelvis[0]), int(left_pelvis[1])
            
            
            right_pelvis_conf = keypoint.conf[0][12].to('cpu').numpy()
            right_pelvis = keypoint.xy[0][12].to('cpu').numpy() # 오른쪽 어깨
            right_pelvis_x, right_pelvis_y = int(right_pelvis[0]), int(right_pelvis[1])
            
            
            left_knee_conf = keypoint.conf[0][13].to('cpu').numpy()
            left_knee = keypoint.xy[0][13].to('cpu').numpy() # 왼쪽 무릎
            left_knee_x, left_knee_y = int(left_knee[0]), int(left_knee[1])
            
            
            right_knee_conf = keypoint.conf[0][14].to('cpu').numpy()
            right_knee = keypoint.xy[0][14].to('cpu').numpy() # 오른쪽 무릎
            right_knee_x, right_knee_y = int(right_knee[0]), int(right_knee[1])
            
            left_shoulder_conf = keypoint.conf[0][5].to('cpu').numpy()
            left_shoulder = keypoint.xy[0][5].to('cpu').numpy()
            left_shoulder_x, left_shoulder_y = int(left_shoulder[0]), int(left_shoulder[1])
            
            right_shoulder_conf = keypoint.conf[0][6].to('cpu').numpy()
            right_shoulder = keypoint.xy[0][6].to('cpu').numpy()
            right_shoulder_x, right_shoulder_y = int(right_shoulder[0]), int(right_shoulder[1])
            
            
            if (nose_conf >= 0.5 and left_pelvis_conf >= 0.5 and right_pelvis_conf >= 0.5 and left_knee_conf >= 0.5 and right_knee_conf >= 0.5
                and left_shoulder_conf >= 0.5 and right_shoulder_conf >= 0.5):
                # self.draw_skeleton(self.origin_img, nose_x, nose_y, self.nose_bgr)
                # self.draw_skeleton(self.origin_img, left_pelvis_x, left_pelvis_y, self.left_pelvis_bgr)
                # self.draw_skeleton(self.origin_img, left_knee_x, left_knee_y, self.left_knee_bgr)
                # self.draw_skeleton(self.origin_img, right_pelvis_x, right_pelvis_y, self.right_pelvis_bgr)
                # self.draw_skeleton(self.origin_img, right_knee_x, right_knee_y, self.right_knee_bgr)

                if (left_pelvis_y >= left_knee_y or right_pelvis_y >= right_knee_y or left_pelvis_y <= nose_y or right_pelvis_y <= nose_y or
                    left_shoulder_y <= nose_y or right_shoulder_y <= nose_y):
                    if box.id is not None:
                        track_id = box.id[0].int().cpu().tolist()
                        self.tracker_update(frame, track_id, x1, y1, x2, y2, telegram_flag)
                    self.draw_falling_result(self.origin_img, x1, y1, x2, y2)
                    
                else:
                    self.draw_normal_result(self.origin_img, x1, y1, x2, y2)
                    
            elif human_width >= human_height:
                if box.id is not None:
                    track_id = box.id[0].int().cpu().tolist()
                    self.tracker_update(frame, track_id, x1, y1, x2, y2, telegram_flag)
                self.draw_falling_result(self.origin_img, x1, y1, x2, y2)
                
            else:
                self.draw_normal_result(self.origin_img, x1, y1, x2, y2)
                
        return self.origin_img
    
    def visualize_skeleton_bbox(self, boxes, keypoints, mode=False):
        boxes = boxes.xyxy.cpu().numpy()
        if mode:
            for box, keypoint_set in zip(boxes, keypoints):
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box)

                # 바운딩 박스 그리기
                # cv2.rectangle(self.origin_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                points = []
                
                for coord, conf in zip(keypoint_set.xy[0], keypoint_set.conf[0]):
                    x, y = map(int, coord)
                    if conf >= self.model_conf:
                        points.append((x, y))
                        cv2.circle(self.origin_img, (x, y), 5, (0, 0, 255), -1)
                    else:
                        points.append(None)
                        
                if points[5] is not None and points[6] is not None:
                    chest_x = (points[5][0] + points[6][0]) // 2
                    chest_y = (points[5][1] + points[6][1]) // 2
                    chest_point = (chest_x, chest_y)
                    points.append(chest_point)  # 가슴 좌표 추가
                    cv2.circle(self.origin_img, chest_point, 5, (255, 0, 0), -1)  # 파란색으로 가슴 표시
                else:
                    chest_point = None
                    points.append(None)

                # 관절 연결 그리기
                for start, end in self.join_parts:
                    # 가슴(chest)은 따로 처리
                    if start == 'chest':
                        start_point = chest_point
                    else:
                        start_point = points[start]

                    if end == 'chest':
                        end_point = chest_point
                    else:
                        end_point = points[end]

                    if start_point is not None and end_point is not None:
                        cv2.line(self.origin_img, start_point, end_point, (255, 0, 0), 2)
                        
    def _make_img_folder(self):
        os.makedirs('img', exist_ok=True)
        
    def save_falling_img(self, img, x1, y1, x2, y2):
        with self.file_lock:  # Lock을 사용하여 파일 접근 동기화
            today = datetime.datetime.now()
            timestamp = today.strftime('%Y%m%d_%H%M%S_%f')  # 밀리초까지 포함
            
            crop_img = img[y1:y2, x1:x2]
            save_img_path = f'crop_img_{timestamp}.jpg'
            result_path = self.resource_path(os.path.join('img', save_img_path))
            
            cv2.imwrite(result_path, crop_img)
            return result_path

    def save_full_img(self, img):
        with self.file_lock:  # Lock을 사용하여 파일 접근 동기화
            today = datetime.datetime.now()
            timestamp = today.strftime('%Y%m%d_%H%M%S_%f')  # 밀리초까지 포함
            
            save_img_path = f'full_img_{timestamp}.jpg'
            result_path = self.resource_path(os.path.join('img', save_img_path))
            
            cv2.imwrite(result_path, img)
            return result_path
        
    def set_save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # self.out = cv2.VideoWriter('output.avi', fourcc, fps, (self.model_img_w, self.model_img_h))
        
    def model_run(self, frame, telegram_flag, visualize_flag):
        resize_frame = cv2.resize(frame, dsize=(self.model_img_w, self.model_img_h), interpolation=cv2.INTER_LINEAR)
        result_img = self.predict(resize_frame, telegram_flag, visualize_flag)
        return result_img
    
    def run(self):
        count = 0
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()
            count += 1
            if count % 5 != 0:
                continue
            if success:
                resize_frame = cv2.resize(frame, dsize=(self.model_img_w, self.model_img_h), interpolation=cv2.INTER_LINEAR)
                result_img = self.predict(resize_frame)
                cv2.imshow('result', result_img)
                # self.out.write(result_img)
                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    detector = Detector()
    detector.run()