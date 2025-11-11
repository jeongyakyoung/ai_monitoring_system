import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QDate, QObject, Qt, QThread
from PyQt5.QtGui import QImage, QPixmap, QFontDatabase, QFont, QCursor, QPainter, QColor
from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
import sys
from PyQt5.QtWidgets import QWidget
import cv2
from controller import Detector, Messenger, WarningLight
from setting import FileController
from PyQt5.QtCore import QFile, QTextStream, pyqtSignal
from functools import partial
import time
import gc
import subprocess

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"        # UDP 대신 TCP로 전송(패킷손실 줄임)
    "fflags;nobuffer|"           # 버퍼 최소화
    "flags;low_delay|"           # 저지연
    "probesize;32|"              # 분석 최소화
    "analyzeduration;0|"         # 분석 최소화
    "max_delay;0|"               # 최대 지연 0
    "stimeout;2000000"           # 소켓 타임아웃 2초(μs 단위)
)

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path('main.ui')
form_class = uic.loadUiType(form)[0]

form_telegram = resource_path('telegram.ui')
form_telegram_window = uic.loadUiType(form_telegram)[0]

form_model = resource_path('model_conf.ui')
form_model_window = uic.loadUiType(form_model)[0]

form_port = resource_path('relay_port.ui')
form_port_window = uic.loadUiType(form_port)[0]

import subprocess
import numpy as np
import cv2
import time
import gc

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter

import subprocess
import numpy as np
import cv2
import time
import gc

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter

import subprocess
import numpy as np
import cv2
import time
import gc

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter

import subprocess
import numpy as np
import cv2
import time
import gc
from threading import Thread
from queue import Queue, Empty

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter


# 라즈베리파이용 #
# class CameraThread(QThread):
#     frame_signal = pyqtSignal(QPixmap, bool)
#     error_signal = pyqtSignal(str)

#     def __init__(self, port, ai_conf, tr_th, messenger):
#         super().__init__()

#         self.rtsp_url = "rtsp://admin:Cctv8324%21@192.168.1.101:554/trackID=1" # 라즈베리파이용
#         self.width = 640
#         self.height = 360
#         self.frame_size = self.width * self.height * 3

#         self.fps = 30
#         self.running = False
#         self.frame_queue = Queue(maxsize=1)

#         self.telegram_flag = True
#         self.skeleton_visualize_flag = True
#         self.messenger = messenger

#         try:
#             self.model = Detector(ai_conf, tr_th, self.fps)
#             if self.model.model is None:
#                 raise RuntimeError("YOLO 모델 로드 실패")
#         except Exception as e:
#             self.error_signal.emit(f"AI 모델 초기화 실패: {e}")
#             self.model = None
#             return

#     def start_frame_reader(self):
#         def reader_loop():
#             cmd = [
#                 "ffmpeg",
#                 "-rtsp_transport", "tcp",
#                 "-fflags", "nobuffer",
#                 "-flags", "low_delay",
#                 "-an",
#                 "-i", self.rtsp_url,
#                 "-vf", "scale=640:360",                # 해상도 축소 추가
#                 "-f", "image2pipe",
#                 "-pix_fmt", "bgr24",
#                 "-vcodec", "rawvideo",
#                 "-"
#             ]
#             try:
#                 pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
#                 while self.running:
#                     raw_frame = pipe.stdout.read(self.frame_size)
#                     if len(raw_frame) != self.frame_size:
#                         continue
#                     if self.frame_queue.full():
#                         try:
#                             self.frame_queue.get_nowait()  # 이전 프레임 제거
#                         except:
#                             pass
#                     self.frame_queue.put_nowait(raw_frame)
#             except Exception as e:
#                 self.error_signal.emit(f"프레임 수신 오류: {e}")

#         Thread(target=reader_loop, daemon=True).start()

#     def run(self):
#         if self.model is None or self.model.model is None:
#             return

#         self.running = True
#         self.start_frame_reader() #라즈베리파이용
#         frame_count = 0

#         while self.running:
#             try:
#                 raw_frame = self.frame_queue.get(timeout=2)
#                 frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))

#                 start_time = time.time()

#                 result_img = self.model.model_run(frame, self.telegram_flag, self.skeleton_visualize_flag)

#                 resized = cv2.resize(result_img, (400, 300))
#                 rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#                 h, w, ch = rgb.shape
#                 image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
#                 pixmap = QPixmap.fromImage(image)

#                 fps_actual = 1.0 / (time.time() - start_time + 1e-6)
#                 self.model.adjust_tracking_threshold(fps_actual)

#                 self.frame_signal.emit(pixmap, True)

#                 frame_count += 1
#                 if frame_count >= 5000:
#                     self.model.reset_tracking()
#                     frame_count = 0
#                     time.sleep(1)

#                 gc.collect()

#             except Empty:
#                 self.send_black_frame()
#             except Exception as e:
#                 print(f"[카메라 처리 오류] {e}")
#                 self.send_black_frame()
#                 time.sleep(1)

#     def send_black_frame(self):
#         black = QImage(400, 300, QImage.Format_RGB888)
#         black.fill(QColor('black'))
#         painter = QPainter(black)
#         painter.setPen(QColor('white'))
#         painter.drawText(black.rect(), Qt.AlignCenter, "카메라 연결 실패")
#         painter.end()
#         self.frame_signal.emit(QPixmap.fromImage(black), False)

#     def stop(self):
#         self.running = False
#         self.wait()


# 윈도우 용 #
class CameraThread(QThread):
    frame_signal = pyqtSignal(QPixmap, bool)
    error_signal = pyqtSignal(str)

    def __init__(self, port, ai_conf, tr_th, messenger):
        super().__init__()
        url = "rtsp://admin:Cctv8324%21@192.168.1.101:554/trackID=1"
        
        self.port = port # url # IDIS 카메라 예시: "rtsp://admin:1234@192.168.0.101:554/trackID=2"
        self.running = False
        self.cap = None
        self.fps = 30
        self.frame_start_time = None
        
        self.telegram_flag = True
        self.skeleton_visualize_flag = True
        self.model = Detector(ai_conf, tr_th, self.fps)
        
        self.messenger = messenger
        
        try:
            self.model = Detector(ai_conf, tr_th, self.fps)
            
            # ✅ 모델이 None이면 실행 중지
            if self.model.model is None:
                raise RuntimeError("🚨 AI 모델 초기화 실패! YOLO 모델이 로드되지 않았습니다.")
            
        except Exception as e:
            print(f"🚨 AI 모델 초기화 오류: {e}")
            self.error_signal.emit(f"AI 모델 초기화 실패: {e}")
            self.model = None
            return  # ✅ 초기화 실패 시 실행 중단
    
    def run(self):
        if self.model is None or self.model.model is None:
            print("🚨 YOLO 모델이 로드되지 않음. 카메라 실행 중단.")
            return

        # --- 캡처 오픈 ---
        self.cap = cv2.VideoCapture(self.port, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        def reopen():
            # 재연결 헬퍼
            try:
                if self.cap:
                    self.cap.release()
            except: pass
            time.sleep(0.2)
            self.cap = cv2.VideoCapture(self.port, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 최초 프레임 확인
        for _ in range(8):  # 시작 시 버퍼 비우기
            self.cap.grab()
        ok, frame = self.cap.retrieve()
        if not ok:
            self.send_black_frame()
            return

        self.running = True
        frame_count = 0
        consecutive_fail = 0

        # 추론 스킵(지연 방지) — 필요 시 2~3으로 올리세요
        INFER_EVERY = 2

        while self.running:
            try:
                loop_start = time.time()

                # --- 오래된 프레임 버리고 최신만 디코딩 ---
                for _ in range(5):  # 상황에 따라 3~8 조절
                    self.cap.grab()
                ok, frame = self.cap.retrieve()
                if not ok or frame is None:
                    consecutive_fail += 1
                    if consecutive_fail >= 10:
                        # 디코딩 에러가 반복되면 재연결
                        reopen()
                        consecutive_fail = 0
                    self.send_black_frame()
                    continue
                else:
                    consecutive_fail = 0

                # --- FPS 변경 감지 시 모델에 전달(선택) ---
                fps_prop = self.cap.get(cv2.CAP_PROP_FPS) or 0
                if fps_prop > 0:
                    self.model.change_fps(fps_prop)

                # --- 추론 스킵으로 지연 억제 ---
                if frame_count % INFER_EVERY == 0:
                    try:
                        result_img = self.model.model_run(frame, self.telegram_flag, self.skeleton_visualize_flag)
                    except Exception as e:
                        print(f"AI 모델 실행 오류: {e}")
                        result_img = frame
                    last_result = result_img
                else:
                    # 직전 결과 재사용
                    result_img = last_result if 'last_result' in locals() else frame

                # --- 디스플레이 변환 ---
                resized_frame = cv2.resize(result_img, (640, 360),interpolation=cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.frame_signal.emit(pixmap, True)

                # --- 실제 처리 FPS로 트래킹 임계값 보정 ---
                proc_time = time.time() - loop_start
                actual_fps = 1.0 / proc_time if proc_time > 0 else fps_prop
                self.model.adjust_tracking_threshold(actual_fps)

                # --- 주기적 트래킹 히스토리 리셋(슬립 금지!) ---
                frame_count += 1
                if frame_count >= 5000:
                    self.model.reset_tracking()
                    frame_count = 0

                gc.collect()

            except Exception as e:
                print(f"카메라 스레드 실행 오류: {e}")
                self.send_black_frame()
                time.sleep(0.1)  # 짧게만

        # 종료
        try:
            self.cap.release()
        except: pass
    
    # def run(self):
    #     if self.model is None or self.model.model is None:
    #         print("🚨 YOLO 모델이 로드되지 않음. 카메라 실행 중단.")
    #         return
    #     self.cap = cv2.VideoCapture(self.port, cv2.CAP_FFMPEG)
    #     self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #     # self.cap = cv2.VideoCapture(self.port, cv2.CAP_GSTREAMER) # GSTREAMER는 라즈베리파이용
    #     # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    #     for _ in range(5):
    #         self.cap.grab()
    #     ok, frame = self.cap.retrieve()

    #     if not ok:
    #         # If the camera is not available, send a black frame
    #         self.send_black_frame()
    #         return

    #     self.running = True
    #     frame_count = 0
        
    #     while self.running:
    #         try:
    #             self.frame_start_time = time.time()
                
    #             ret, frame = self.cap.read()
    #             if not ret:
    #                 self.send_black_frame()
    #                 break
                
    #             fps = self.cap.get(cv2.CAP_PROP_FPS)
    #             self.model.change_fps(fps)
    #             try:
    #                 if self.model and hasattr(self.model, 'model_run'):
    #                     result_img = self.model.model_run(frame, self.telegram_flag, self.skeleton_visualize_flag)
    #                 else:
    #                     print("AI 모델이 초기화되지 않았습니다.")
    #                     result_img = frame
    #             except Exception as e:
    #                 print(f"AI 모델 실행 오류 발생: {(e)}")
    #                 result_img = frame
                    
    #             gc.collect()
                
    #             resized_frame = cv2.resize(result_img, (400, 300)) # 원래 코드는 result_img
    #             rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    #             h, w, ch = rgb_frame.shape
    #             bytes_per_line = ch * w
    #             qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    #             pixmap = QPixmap.fromImage(qt_image)
                
    #             processing_time = time.time() - self.frame_start_time
    #             actual_fps = 1.0 / processing_time if processing_time > 0 else fps
            
    #             self.model.adjust_tracking_threshold(actual_fps)
                
    #             self.frame_signal.emit(pixmap, True)
                
    #             frame_count += 1
    #             if frame_count >= 5000:
    #                 self.model.reset_tracking()
    #                 frame_count = 0
                    
                    
    #         except Exception as e:
    #             print(f"카메라 스레드 실행 중 오류 발생: {e}")
    #             self.send_black_frame()
    #             time.sleep(1)
                
    #     self.cap.release()
        
    def send_black_frame(self):
        """Send a black frame with a 'Camera Unavailable' message."""
        black_image = QImage(400, 300, QImage.Format_RGB888)
        black_image.fill(QColor('black'))
        painter = QPainter(black_image)
        painter.setPen(QColor('white'))
        painter.setFont(painter.font())
        painter.drawText(black_image.rect(), Qt.AlignCenter, "카메라 연결 실패")
        painter.end()
        black_pixmap = QPixmap.fromImage(black_image)
        self.frame_signal.emit(black_pixmap, False)
        
    def stop(self):
        self.running = False
        self.wait()
#--------------------------------------------------------------------------------------------#

    # def run(self): # 동영상파일 실행시(데모 버전)
    #     self.cap = cv2.VideoCapture('./soruce/falling_6.mp4')
        
    #     while self.cap.isOpened():
    #         ret ,frame = self.cap.read()
    #         if not ret:
    #             self.stop()
    #         else:
    #             last_frame = self.cvimage_to_label(frame)
    #             self.frame_signal.emit(last_frame)

    def change_telegram_func(self, flag):
        self.telegram_flag = flag
    
    def change_skeleton_visualize_func(self, flag):
        self.skeleton_visualize_flag = flag
        
class TelegramwindowClass(QDialog, QWidget, form_telegram_window): 
    def __init__(self):
        super(TelegramwindowClass, self).__init__()
        self.initUi()
        self.setting = FileController()
        self.json_data = self.setting.load_json()
        self.set_telegram_api_label()
        self.set_telegram_id_label()
        self.save_btn.clicked.connect(self.save_value)
    
    def initUi(self):
        self.setupUi(self)
        self.setWindowTitle("텔레그램 정보 등록")
    
    def get_telegram_token(self):
        self.json_data = self.setting.load_json()
        return self.json_data['telegram_api']
    
    def get_telegram_bot_id(self):
        self.json_data = self.setting.load_json()
        return self.json_data['telegram_bot_id']
    
    def set_telegram_api_label(self):
        api = self.json_data['telegram_api']
        if api is None or api == "":
            self.telegram_api_label.setText("")
        else:
            self.telegram_api_label.setText(str(api))
        
    def set_telegram_id_label(self):
        id = self.json_data['telegram_bot_id']
        if id is None or id == "":
            self.telegram_id_label.setText("")
        else:
            self.telegram_id_label.setText(str(id))
        
    # def enroll_telegram_info(self):
    #     self.messenger.set_telegram(self.telegram_api_label.text(), self.telegram_id_label.text())
        
    def get_telegram_api_label(self):
        return self.telegram_api_label.text()
    
    def get_telegram_id_label(self):
        return self.telegram_id_label.text()

    def save_telegram_id_label(self):
        id = self.telegram_id_label.text()
        self.setting.revise_str_json('telegram_bot_id', str(id))
        
    def save_telegram_api_label(self):
        api = self.telegram_api_label.text()
        self.setting.revise_str_json('telegram_api', str(api))

    def save_value(self):
        self.save_telegram_api_label()
        self.save_telegram_id_label()
        self.save_popup()
        
    def save_popup(self):
        QMessageBox.information(self, "OK", "저장 완료")
        
    def open(self):
        self.exec_()
        
class ModelwindowClass(QDialog, QWidget, form_model_window): 
    def __init__(self):
        super(ModelwindowClass, self).__init__()
        self.initUi()
        self.setting = FileController()
        self.json_data = self.setting.load_json()
        self.set_model_conf()
        self.set_tracking_th()
        self.save_btn.clicked.connect(self.save_value)
        
        self.ai_conf = 0
        self.tr_th = 0
        
    def initUi(self):
        self.setupUi(self)
        self.setWindowTitle("모델 추론 임계값 변경")
        
    def save_model_conf(self):
        self.ai_conf = float(self.model_conf_label.text())
        self.setting.revise_str_json('ai_model_conf', float(self.ai_conf))
        
    def set_model_conf(self):
        self.ai_conf = self.json_data['ai_model_conf']
        self.model_conf_label.setText(str(self.ai_conf))
    
    def save_tracking_th(self):
        self.tr_th = float(self.tracking_th.text())
        self.setting.revise_str_json('tracking_th', int(self.tr_th))
        
    def set_tracking_th(self):
        self.tr_th = self.json_data['tracking_th']
        self.tracking_th.setText(str(self.tr_th))
        
    def save_popup(self):
        QMessageBox.information(self, "OK", "저장 완료")
    
    def save_value(self):
        self.save_model_conf()
        self.save_tracking_th()
        self.save_popup()
    
    def get_ai_conf(self):
        return self.ai_conf
    
    def get_tracking_threshold(self):
        return self.tr_th
    
    def open(self):
        self.exec_()

class RelayPortwindowClass(QDialog, QWidget, form_port_window): 
    def __init__(self):
        super(RelayPortwindowClass, self).__init__()
        self.initUi()

        self.port_num = ""

        self.save_btn.clicked.connect(self.save_port_num)
        self.setting = FileController()
        self.json_data = self.setting.load_json()

        self.set_relay_module_port_num()

    def initUi(self):
        self.setupUi(self)
        self.setWindowTitle("릴레이 모듈 포트 번호 변경")
    
    def set_relay_module_port_num(self):
        self.port_num = self.json_data['relay_module_port']
        self.port_value.setText(self.port_num) # 값 설정

    def save_popup(self):
        QMessageBox.information(self, "OK", "포트 변경 완료")

    def save_port_num(self):
        self.port_num =  self.port_value.text()
        self.setting.revise_str_json('relay_module_port', str(self.port_num).upper())
        self.save_popup()
        #self.port_value.setText(text) # 값 설정

    def get_port_num(self):
        return self.port_num
    
    def open(self):
        self.exec_()

class CameraPreview(QWidget):
    closed = pyqtSignal(str)
    
    def __init__(self, camera_name, parent=None):
        super().__init__(parent)
        self.w = 1280
        self.h = 720
        self.camera_name = camera_name
        
        self.setWindowTitle(f"{camera_name} - 확대 보기")
        self.setGeometry(100, 100, self.w, self.h)  # 확대 창의 기본 크기

        self.layout = QVBoxLayout(self)
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black; background-color: #000;")
        self.layout.addWidget(self.video_label)

    def update_frame(self, pixmap):
        """Update the frame in the preview window."""
        self.video_label.setPixmap(pixmap.scaled(self.w, self.h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def closeEvent(self, event):
        """Signal when the preview window is closed."""
        self.closed.emit(self.camera_name)
        super().closeEvent(event)
                
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super(WindowClass, self).__init__()
        self.setupUi(self)
        self.load_stylesheet()
        self.set_cursor()
        self.setWindowTitle("AI기반 위험감지 모니터링 시스템")

        self.setting = FileController()
        self.json_data = self.setting.load_json()
        
        self.camera_threads = {}
        self.video_labels = {}
        
        self.active_thread = None
        
        self.telegram_winodw_class = TelegramwindowClass()
        self.telegram_enroll_btn.clicked.connect(self.open_telegram_info_dialog)
        
        self.model_window_class = ModelwindowClass()
        self.ai_model_conf_btn.clicked.connect(self.open_model_conf_dialog)
        
        self.relay_port_window_class = RelayPortwindowClass()

        self.messenger = Messenger()
        self.api_token = self.telegram_winodw_class.get_telegram_api_label()
        self.chat_id = self.telegram_winodw_class.get_telegram_id_label()
        
        self.change_relay_port_btn.clicked.connect(self.open_relay_port_dialog)

        self.warning_light = WarningLight(self.json_data['relay_module_port'])
        if self.warning_light is None:
            QMessageBox.warning(self, "OK", "경광등 포트를 확인 해 주세요.")
        
        self.alarm_btn.clicked.connect(self.pause_alarm) # 라즈베리파이 경광등 종료
        self.alarm_test_btn.clicked.connect(self.test_alarm)

        if self.api_token and self.chat_id:
            self.messenger.set_telegram(self.api_token, self.chat_id)
            
        self.active_radio_btn.clicked.connect(self.telegram_active)
        self.unactive_radio_btn.clicked.connect(self.telegram_unactive)
        
        self.sk_active_radio_btn.clicked.connect(self.skelton_active)
        self.sk_unactive_radio_btn.clicked.connect(self.skelton_unactive)

        self.camera_add_btn.clicked.connect(self.add_camera)
        self.camera_data = self.json_data['camera_list']
        
        self.telegram_test_btn.clicked.connect(self.test_telegram)
        
        self.ai_conf = self.json_data['ai_model_conf']
        self.tr_th = self.json_data['tracking_th']
        
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.current_row = 0
        self.current_col = 0
        self.max_cols = 3
        
        self.relay_module_port_num = self.json_data['relay_module_port']

        self.selected_camera = None
        
        self.init_camera_list()
        self.check_telegram_info()
    
    def test_alarm(self):
        self.warning_light.on(auto_off_seconds=5)
        QMessageBox.information(self, "경보 테스트", "경보 실행 완료")

    def pause_alarm(self):
        self.warning_light.off()
        QMessageBox.information(self, "경보", "경보 종료")
        
    def test_telegram(self):
        self.messenger.test_telegram_msg()
        
    def check_telegram_info(self):
        if self.api_token and self.chat_id:
            self.telegram_active()
        else:
            self.telegram_unactive()
            
    def update_json_value(self):
        self.json_data = self.setting.load_json()
        
    def open_camera_preview(self, camera_name):
        """Open a new window to display the camera preview."""
        if camera_name not in self.video_labels or camera_name not in self.camera_threads:
            QMessageBox.warning(self, "오류", f"카메라 '{camera_name}'를 찾을 수 없습니다.")
            return

        # 이미 존재하는 경우 새로 만들지 않음
        if not hasattr(self, 'camera_preview_windows'):
            self.camera_preview_windows = {}

        if camera_name not in self.camera_preview_windows:
            preview_window = CameraPreview(camera_name)
            self.camera_preview_windows[camera_name] = preview_window
            
            # 확대 창 닫힐 때 처리 연결
            preview_window.closed.connect(self.restore_camera_signal)
            
            preview_window.show()
    
        # 연결 재설정
        camera_thread = self.camera_threads[camera_name]
        camera_thread.frame_signal.disconnect()  # 기존 연결 해제
        camera_thread.frame_signal.connect(
            lambda pixmap, available, name=camera_name: self.update_preview(name, pixmap)
        )

    def restore_camera_signal(self, camera_name):
        """Restore the camera signal to the original QLabel."""
        if camera_name not in self.camera_threads or camera_name not in self.video_labels:
            return

        camera_thread = self.camera_threads[camera_name]

        # 확대 창 제거
        if hasattr(self, 'camera_preview_windows') and camera_name in self.camera_preview_windows:
            del self.camera_preview_windows[camera_name]

        # 신호 복원
        camera_thread.frame_signal.disconnect()
        camera_thread.frame_signal.connect(
            lambda pixmap, available, name=camera_name: self.update_frame(name, pixmap, available)
        )
        print(f"Camera signal restored to QLabel for '{camera_name}'.")
        
    def update_preview(self, camera_name, pixmap):
        if hasattr(self, 'camera_preview_windows') and camera_name in self.camera_preview_windows:
            self.camera_preview_windows[camera_name].update_frame(pixmap)
    
    def init_camera_list(self):
        camera_list_dict = self.json_data['camera_list']
        if len(camera_list_dict) >= 1:
            for camera_name, port in camera_list_dict.items():
                video_label = QLabel(f"{camera_name} - {port}")
                video_label.setFixedSize(400, 300)
                video_label.setAlignment(Qt.AlignCenter)
                video_label.setStyleSheet("border: 1px solid black; background-color: #000; color: white;")
                video_label.setContextMenuPolicy(Qt.CustomContextMenu)
                
                # 레이블에 이름 저장
                video_label.camera_name = camera_name
                
                # 연결
                video_label.mouseDoubleClickEvent = partial(self.handle_double_click, camera_name)
                video_label.customContextMenuRequested.connect(
                    partial(self.handle_right_click, video_label)
                )
                self.video_labels[camera_name] = video_label

                # Grid Layout에 레이블 추가
                self.grid_layout.addWidget(video_label, self.current_row, self.current_col)

                # 다음 열/행 계산
                self.current_col += 1
                if self.current_col >= self.max_cols:
                    self.current_col = 0
                    self.current_row += 1

                # 카메라 스레드 생성 및 시작
                camera_thread = CameraThread(port, self.ai_conf, self.tr_th, self.messenger)
                camera_thread.frame_signal.connect(lambda pixmap, available, name=camera_name: self.update_frame(name, pixmap, available))
                self.camera_threads[camera_name] = camera_thread
                camera_thread.start()
                print(f"Added and started camera: {camera_name} with port {port}")
    
        else:
            print(f"No data")
            
    def handle_double_click(self, camera_name, event):
        """Handle double-click on a QLabel to open the camera preview."""
        self.open_camera_preview(camera_name)
        
    def add_camera(self):
        """카메라 추가"""
        camera_name, ok = QInputDialog.getText(self, "카메라 추가", "카메라 이름을 입력하세요:")
        if not ok or not camera_name.strip():
            return
        
        if camera_name in self.video_labels:
            QMessageBox.warning(self, "오류", "이미 존재하는 카메라 이름입니다.")
            return
        
        # 카메라 포트 번호 입력 받기
        port, ok = QInputDialog.getText(self, "포트 설정", "카메라 주소를 입력하세요:")
        if not ok:
            return
            
        try:
            # 새로운 카메라 스레드 생성
            camera_thread = CameraThread(port, self.ai_conf, self.tr_th, self.messenger)
            if camera_thread.model is None or camera_thread.model.model is None:
                QMessageBox.warning(self, "오류", "AI 모델을 로드할 수 없습니다. 모델 파일을 확인하세요.")
                return

            camera_thread.frame_signal.connect(
                lambda pixmap, available, name=camera_name: self.update_frame(name, pixmap, available)
            )
            
            # UI 레이블 생성
            video_label = QLabel()
            video_label.setFixedSize(400, 300)
            video_label.setAlignment(Qt.AlignCenter)
            video_label.setText("카메라 연결을 확인해 주세요")
            video_label.setStyleSheet("border: 1px solid black; background-color: #000; color: white;")
            video_label.mousePressEvent = lambda event: self.handle_right_click(video_label) if event.button() == Qt.RightButton else None
            video_label.mouseDoubleClickEvent = lambda event: self.open_camera_preview(camera_name)
            video_label.camera_name = camera_name
            
            # 그리드에 레이블 추가
            self.grid_layout.addWidget(video_label, self.current_row, self.current_col)
            
            # 다음 위치 계산
            self.current_col += 1
            if self.current_col >= self.max_cols:
                self.current_col = 0
                self.current_row += 1
            
            # 카메라 정보 저장
            self.video_labels[camera_name] = video_label
            self.camera_threads[camera_name] = camera_thread
            
            # JSON 파일에 카메라 정보 저장
            self.setting.add_dict_json('camera_list', camera_name, port)
            
            # 카메라 스레드 시작
            camera_thread.start()
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"카메라 추가 중 오류 발생: {e}")
            if camera_name in self.video_labels:
                self.video_labels[camera_name].deleteLater()
                del self.video_labels[camera_name]
            if camera_name in self.camera_threads:
                self.camera_threads[camera_name].stop()
                del self.camera_threads[camera_name]
    
    def handle_right_click(self, widget):
        """우클릭 이벤트 처리"""
        for name, label in self.video_labels.items():
            if label == widget:
                self.selected_camera = name
                context_menu = QMenu(self)
                rename_action = context_menu.addAction("이름 변경")
                port_change_action = context_menu.addAction("포트 변경")
                delete_action = context_menu.addAction("삭제")

                # Connect actions to their respective methods
                rename_action.triggered.connect(self.rename_camera)
                port_change_action.triggered.connect(self.change_camera_port)
                delete_action.triggered.connect(self.delete_camera)

                # Show context menu at the cursor's position
                context_menu.exec_(QCursor.pos())
                break
    
    def rename_camera(self):
        """카메라 이름 변경"""
        if not self.selected_camera or self.selected_camera not in self.video_labels:
            return

        new_name, ok = QInputDialog.getText(self, "이름 변경", "새로운 이름을 입력하세요:", text=self.selected_camera)
        if ok and new_name.strip():
            if new_name in self.video_labels:
                QMessageBox.warning(self, "오류", "중복된 이름입니다. 다른 이름을 사용하세요.")
                return

            # 기존 정보 가져오기
            video_label = self.video_labels.pop(self.selected_camera)
            camera_thread = self.camera_threads.pop(self.selected_camera)

            # UI 및 속성 업데이트
            video_label.setText(f"{new_name} - {camera_thread.port}")
            video_label.camera_name = new_name  # 새로운 이름으로 레이블 속성 업데이트

            # 더블 클릭 이벤트 핸들러 재설정
            video_label.mouseDoubleClickEvent = lambda event: self.open_camera_preview(new_name)

            # 새로운 이름으로 매핑 업데이트
            self.video_labels[new_name] = video_label
            self.camera_threads[new_name] = camera_thread

            # JSON 파일 업데이트
            self.setting.revise_key_dict_json('camera_list', self.selected_camera, new_name)

            # 확대 창 매핑 갱신
            if hasattr(self, 'camera_preview_windows') and self.selected_camera in self.camera_preview_windows:
                preview_window = self.camera_preview_windows.pop(self.selected_camera)
                preview_window.setWindowTitle(f"{new_name} - 확대 보기")
                self.camera_preview_windows[new_name] = preview_window

            # 기존 시그널 연결 해제 및 재연결
            camera_thread.frame_signal.disconnect()
            camera_thread.frame_signal.connect(lambda pixmap, available, name=new_name: self.update_frame(name, pixmap, available))

            print(f"카메라 이름 변경: {self.selected_camera} -> {new_name}")
            self.selected_camera = new_name
   
    def change_camera_port(self):
        """카메라 포트 변경"""
        if not self.selected_camera or self.selected_camera not in self.camera_threads:
            return

        new_port, ok = QInputDialog.getText(self, "카메라 주소 변경", "새로운 카메라 주소를 입력하세요:")
        if ok:
            camera_thread = self.camera_threads[self.selected_camera]
            camera_thread.stop()  # 기존 스레드 종료
            new_thread = CameraThread(new_port, self.ai_conf, self.tr_th, self.messenger)
            new_thread.frame_signal.connect(lambda pixmap, available, name=self.selected_camera: self.update_frame(name, pixmap, available))
            self.camera_threads[self.selected_camera] = new_thread
            new_thread.start()
            
            # 기존 json 파일 매핑 업데이트
            self.setting.revise_val_dict_json('camera_list', self.selected_camera, new_port)
            
            # UI 업데이트
            video_label = self.video_labels[self.selected_camera]
            video_label.setText(f"{self.selected_camera} - {new_port}")

            # 확대 창 연결 재설정
            if hasattr(self, 'camera_preview_windows') and self.selected_camera in self.camera_preview_windows:
                camera_thread.frame_signal.disconnect()  # 기존 연결 해제
                new_thread.frame_signal.connect(
                    lambda pixmap, available, name=self.selected_camera: self.update_preview(name, pixmap)
                )

            print(f"카메라 포트 변경: {self.selected_camera} -> {new_port}")
            self.selected_camera = None
            
    def delete_camera(self):
        """카메라 삭제"""
        if not self.selected_camera or self.selected_camera not in self.camera_threads:
            QMessageBox.warning(self, "오류", "삭제할 카메라를 선택하세요.")
            return

        # 사용자 확인
        reply = QMessageBox.question(
            self,
            "삭제 확인",
            f"'{self.selected_camera}' 카메라를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        # 스레드 정리
        camera_thread = self.camera_threads[self.selected_camera]
        camera_thread.frame_signal.disconnect()  # 시그널 연결 해제
        camera_thread.stop()
        del self.camera_threads[self.selected_camera]

        # UI 요소 삭제
        video_label = self.video_labels[self.selected_camera]
        self.grid_layout.removeWidget(video_label)
        video_label.deleteLater()
        del self.video_labels[self.selected_camera]

        # json 요소 삭제
        self.setting.remove_dict_json(self.selected_camera)
        
        # 레이아웃 정리
        self.rearrange_grid_layout()

        print(f"카메라 삭제 완료: {self.selected_camera}")
        self.selected_camera = None
    
    def rearrange_grid_layout(self):
        row, col = 0, 0
        widgets = []

        # 현재 GridLayout에 있는 모든 위젯 수집
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if isinstance(widget, QLabel):
                widgets.append(widget)

        # 모든 위젯 제거
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)

        # 위젯들을 한 칸씩 당겨 재배치
        for widget in widgets:
            self.grid_layout.addWidget(widget, row, col)
            col += 1
            if col >= self.max_cols:
                col = 0
                row += 1

        # 현재 위치 업데이트
        self.current_row = row
        self.current_col = col
                        
    def show_error(self, error_message):
        """Show error messages."""
        QMessageBox.warning(self, "오류", error_message)

    def set_cursor(self):
        cusor_path = resource_path('style/cursor/jyy.png')
        cursor_pixmap = QPixmap(cusor_path)
        scaled_pixmap = cursor_pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        custom_cursor = QCursor(scaled_pixmap, 10, 10)
        self.setCursor(custom_cursor)
        
    def load_external_font(self, font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            print(f"폰트를 로드하지 못했습니다: {font_path}")
            return ""
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        if font_families:
            return font_families[0]
        else:
            print("등록된 폰트 이름을 가져오지 못했습니다.")
            return ""

    def load_stylesheet(self):
        # stylesheet.qss 파일 로드
        self.setStyleSheet("")
        qss_file_path = resource_path('style/stylesheet.qss')
        qss_file = QFile(qss_file_path)
        qss_file.open(QFile.ReadOnly | QFile.Text)
        qss_stream = QTextStream(qss_file)
        qss_stream_all = qss_stream.readAll()
        qss_file.close()
        
        font_path = resource_path("style/font/Maplestory Light.ttf")  # 폰트 파일 경로
        font_name = self.load_external_font(font_path)
        
        if font_name:
            # QSS에서 폰트 이름 대체
            qss_stream_all = qss_stream_all.replace('Maple Story', font_name)

            # 기존 스타일 초기화 후 새 스타일 적용
            self.setStyleSheet("")
            self.setStyleSheet(qss_stream_all)

            # 프로그램적으로 폰트 설정 (필요시)
            font = QFont(font_name)
            font.setPointSize(16)
            self.setFont(font)
        else:
            self.setStyleSheet(qss_stream_all)

    def skelton_active(self):
        for camera_thread in self.camera_threads.values():
            camera_thread.change_skeleton_visualize_func(flag=True)
    
    def skelton_unactive(self):
        for camera_thread in self.camera_threads.values():
            camera_thread.change_skeleton_visualize_func(flag=False)
        
    def telegram_active(self):
        if self.telegram_winodw_class.get_telegram_api_label() == "" or self.telegram_winodw_class.get_telegram_id_label() == "":
            QMessageBox.warning(self, "오류", "텔레그램 정보를 찾을 수 없습니다.")
            self.unactive_radio_btn.setChecked(True)
        else:
            if len(self.video_labels) > 0:
                for camera_thread in self.camera_threads.values():
                    camera_thread.change_telegram_func(flag=True)
                self.active_radio_btn.setChecked(True)
        
    def telegram_unactive(self):
        for camera_thread in self.camera_threads.values():
            camera_thread.change_telegram_func(flag=False)
        
    def open_model_conf_dialog(self):
        self.model_window_class.open()
        self.ai_conf = self.model_window_class.get_ai_conf()
        self.tr_th = self.model_window_class.get_tracking_threshold()
        for camera_thread in self.camera_threads.values():
            camera_thread.model.change_value(self.ai_conf, self.tr_th)
            
    def open_telegram_info_dialog(self):
        self.telegram_winodw_class.open()
        self.api_token = self.telegram_winodw_class.get_telegram_api_label()
        self.chat_id = self.telegram_winodw_class.get_telegram_id_label()
        self.messenger.set_telegram(self.api_token, self.chat_id)
        self.check_telegram_info()
    
    def open_relay_port_dialog(self):
        self.relay_port_window_class.open()
        self.relay_module_port_num = self.relay_port_window_class.get_port_num()
        self.warning_light.change_port(self.relay_module_port_num)

    def update_frame(self, camera_name, pixmap, available):
        if camera_name not in self.video_labels:
            return  # 카메라가 삭제된 경우 처리 중단

        self.video_labels[camera_name].setPixmap(pixmap)
        if not available:
            self.video_labels[camera_name].setStyleSheet("border: 1px solid red; background-color: #000; color: white;")
        else:
            self.video_labels[camera_name].setStyleSheet("border: 1px solid black; background-color: #000; color: white;")
    
    def closeEvent(self, event):
        """Stop all camera threads when the application is closed."""
        for thread in self.camera_threads.values():
            thread.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = WindowClass()
    main_window.setWindowState(Qt.WindowMaximized)
    main_window.show()
    
    app.exec_()