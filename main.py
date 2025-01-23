import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QDate, QObject, Qt, QThread
from PyQt5.QtGui import QImage, QPixmap, QFontDatabase, QFont, QCursor, QPainter, QColor
from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
import sys
from PyQt5.QtWidgets import QWidget
import cv2
from controller import Detector, Messenger
from setting import FileController
from PyQt5.QtCore import QFile, QTextStream, pyqtSignal
from functools import partial
import time
def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form = resource_path('main.ui')
form_class = uic.loadUiType(form)[0]

form_telegram = resource_path('telegram.ui')
form_telegram_window = uic.loadUiType(form_telegram)[0]

form_model = resource_path('model_conf.ui')
form_model_window = uic.loadUiType(form_model)[0]


class CameraThread(QThread):
    frame_signal = pyqtSignal(QPixmap, bool)
    error_signal = pyqtSignal(str)

    def __init__(self, port, ai_conf, tr_th):
        super().__init__()
        self.port = port
        self.running = False
        self.cap = None
        self.fps = 30
        self.frame_start_time = None
        
        self.telegram_flag = False
        self.skeleton_visualize_flag = True
        self.model = Detector(ai_conf, tr_th, self.fps)
        
        self.messenger = Messenger()
        
    def run(self):
        #Todo : 옆 제안 의견 반영하여 실제 초 단위 계산으로 변경
        self.cap = cv2.VideoCapture(self.port)
            
        if not self.cap.isOpened():
            # If the camera is not available, send a black frame
            self.send_black_frame()
            return

        self.running = True
        
        while self.running:
            try:
                self.frame_start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    self.send_black_frame()
                    break
                
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.model.change_fps(fps)
                try:
                    if self.model and hasattr(self.model, 'model_run'):
                        result_img = self.model.model_run(frame, self.telegram_flag, self.skeleton_visualize_flag)
                    else:
                        print("AI 모델이 초기화되지 않았습니다.")
                        result_img = frame
                except Exception as e:
                    print(f"AI 모델 실행 오류 발생: {(e)}")
                    result_img = frame
                resized_frame = cv2.resize(result_img, (400, 300))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                processing_time = time.time() - self.frame_start_time
                actual_fps = 1.0 / processing_time if processing_time > 0 else fps
                
                self.model.adjust_tracking_threshold(actual_fps)
                
                self.frame_signal.emit(pixmap, True)
            except Exception as e:
                print(f"카메라 스레드 실행 중 오류 발생: {e}")
                self.send_black_frame()
                time.sleep(1)
                
        self.cap.release()
        
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
        
        self.messenger = Messenger()
        self.api_token = self.telegram_winodw_class.get_telegram_api_label()
        self.chat_id = self.telegram_winodw_class.get_telegram_id_label()
        
        if self.api_token and self.chat_id:
            self.messenger.set_telegram(self.api_token, self.chat_id)
            
        self.active_radio_btn.clicked.connect(self.telegram_active)
        self.unactive_radio_btn.clicked.connect(self.telegram_unactive)
        
        self.sk_active_radio_btn.clicked.connect(self.skelton_active)
        self.sk_unactive_radio_btn.clicked.connect(self.skelton_unactive)

        self.camera_add_btn.clicked.connect(self.add_camera)
        self.camera_data = self.json_data['camera_list']
        
        self.ai_conf = self.json_data['ai_model_conf']
        self.tr_th = self.json_data['tracking_th']
        
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.current_row = 0
        self.current_col = 0
        self.max_cols = 3
        
        self.selected_camera = None
        
        self.init_camera_list()
        self.check_telegram_info()
        
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
                camera_thread = CameraThread(port, self.ai_conf, self.tr_th)
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
        port, ok = QInputDialog.getInt(self, "포트 설정", "카메라 포트 번호를 입력하세요:", min=0)
        if not ok:
            return
            
        try:
            # 새로운 카메라 스레드 생성
            camera_thread = CameraThread(port, self.ai_conf, self.tr_th)
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

        new_port, ok = QInputDialog.getInt(self, "포트 변경", "새로운 포트를 입력하세요:", min=0, max=65535)
        if ok:
            camera_thread = self.camera_threads[self.selected_camera]
            camera_thread.stop()  # 기존 스레드 종료
            new_thread = CameraThread(new_port, self.ai_conf, self.tr_th)
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
    main_window.show()
    app.exec_()