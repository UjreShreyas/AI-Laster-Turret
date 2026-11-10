import sys
import os
import json
import math
import time
import cv2
import numpy as np
import pygame
import serial
import serial.tools.list_ports
from ultralytics import YOLO

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QLinearGradient, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QFrame, QGraphicsDropShadowEffect,
    QMessageBox
)

# --- COCO CLASS MAP ---
COCO_CLASSES = {
    "All Objects": -1,
    "Person": 0, "Cat": 15, "Dog": 16, "Sports Ball": 32,
    "Bottle": 39, "Cup": 41, "Cell Phone": 67, "Laptop": 63,
    "Chair": 56, "Car": 2, "Motorcycle": 3, "Bicycle": 1,
    "Backpack": 24, "Umbrella": 25, "Book": 73, "Teddy Bear": 77
}
CLASS_NAMES_LIST = list(COCO_CLASSES.keys())

# --- PASTEL THEMES CONFIGURATION ---
THEMES = {
    "Sakura Blossom": [QColor("#FFC3A0"), QColor("#FFAFBD"), QColor("#E6E6FA"), QColor("#B0E0E6")],
    "Pastel Sunset": [QColor("#FFDAC1"), QColor("#FFB7B2"), QColor("#E2F0CB"), QColor("#B5EAD7")],
    "Aurora Mint": [QColor("#A8E6CF"), QColor("#DCEDC1"), QColor("#FFD3B6"), QColor("#FFAAA5")],
    "Cyber Lavender": [QColor("#D8B4F8"), QColor("#CAEDFF"), QColor("#FBF0B2"), QColor("#FFC7EA")],
    "Matcha Dew": [QColor("#D4E7C5"), QColor("#BFD8AF"), QColor("#E1F0DA"), QColor("#99BC85")]
}


# --- ANIMATED BACKGROUND WIDGET ---
class AnimatedGradientWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0.0
        self.current_theme = "Sakura Blossom"
        self.colors = THEMES[self.current_theme]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(33)

    def set_theme(self, theme_name):
        if theme_name in THEMES:
            self.current_theme = theme_name
            self.colors = THEMES[theme_name]
            self.update()

    def update_animation(self):
        self.angle = (self.angle + 0.5) % 360.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rad = math.radians(self.angle)
        x1 = self.width() * 0.5 + math.cos(rad) * self.width() * 0.5
        y1 = self.height() * 0.5 + math.sin(rad) * self.height() * 0.5
        x2 = self.width() * 0.5 - math.cos(rad) * self.width() * 0.5
        y2 = self.height() * 0.5 - math.sin(rad) * self.height() * 0.5

        gradient = QLinearGradient(float(x1), float(y1), float(x2), float(y2))
        gradient.setColorAt(0.0, self.colors[0])
        gradient.setColorAt(0.33, self.colors[1])
        gradient.setColorAt(0.66, self.colors[2])
        gradient.setColorAt(1.0, self.colors[3])

        painter.fillRect(self.rect(), gradient)


# --- CLICKABLE CAMERA DISPLAY WIDGET ---
class ClickableVideoLabel(QLabel):
    point_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            w_disp = self.width()
            h_disp = self.height()
            if w_disp > 0 and h_disp > 0:
                scale_x = 1280.0 / w_disp
                scale_y = 720.0 / h_disp
                real_x = int(pos.x() * scale_x)
                real_y = int(pos.y() * scale_y)
                self.point_clicked.emit(real_x, real_y)
        super().mousePressEvent(event)


# --- CAMERA & TRACKING WORKER THREAD ---
class TrackingThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, object, int, int)
    status_updated = pyqtSignal(str)
    class_changed = pyqtSignal(str)

    def __init__(self, config_path="config.json"):
        super().__init__()
        self.config_path = config_path
        self.load_config()

        self.running = True
        self.is_calibrating = True
        self.calib_stages = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.calib_index = 0

        self.screen_points = []
        self.servo_points = []
        self.homography_matrix = None

        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        self.smoothing = 0.8
        self.target_class_id = -1
        self.class_idx = 0
        self.conf_threshold = 0.45

        # Target Locks
        self.manual_target_pt = None
        self.locked_box_pt = None

        # Virtual Crosshair (Moved by Right Stick)
        self.crosshair_x = 640.0
        self.crosshair_y = 360.0
        self.crosshair_speed = 14.0

        # Laser State
        self.laser_enabled = True

        # Serial Connection
        self.arduino = None
        self.init_serial(self.config.get("serial_port", "COM4"))

        # Pygame Joystick (Xbox Controller)
        pygame.init()
        pygame.joystick.init()
        self.controller = None
        self.init_controller()

        # Load YOLO Model
        model_file = self.config.get("model_path", "yolov8n.pt")
        self.model = YOLO(model_file)

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "serial_port": "COM4", "baud_rate": 115200, "camera_index": 0,
                "model_path": "yolov8n.pt"
            }

    def init_controller(self):
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()
            print(f"Controller Connected: {self.controller.get_name()}")

    def init_serial(self, requested_port="COM4"):
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.close()
            except Exception:
                pass

        baud = self.config.get("baud_rate", 115200)

        try:
            self.arduino = serial.Serial(port=requested_port, baudrate=baud, timeout=0.1)
            time.sleep(1.5)
            self.status_updated.emit(f"Serial: Connected ({requested_port})")
            return
        except Exception:
            pass

        available_ports = serial.tools.list_ports.comports()
        for p in available_ports:
            try:
                self.arduino = serial.Serial(port=p.device, baudrate=baud, timeout=0.1)
                time.sleep(1.5)
                self.config["serial_port"] = p.device
                self.status_updated.emit(f"Serial: Auto-Connected ({p.device})")
                return
            except Exception:
                continue

        self.arduino = None
        self.status_updated.emit(f"Serial: Disconnected (No Port Found)")

    def toggle_laser(self):
        self.laser_enabled = not self.laser_enabled
        if self.arduino and self.arduino.is_open:
            cmd = "L1\n" if self.laser_enabled else "L0\n"
            try:
                self.arduino.write(cmd.encode())
            except Exception:
                pass

    def run(self):
        cap = cv2.VideoCapture(self.config.get("camera_index", 0), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        rt_released = True
        lt_released = True
        btn_x_released = True
        btn_y_released = True
        btn_a_released = True
        lb_released = True
        rb_released = True

        joy_speed = self.config.get("joy_speed", 2.0)

        while self.running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)

            pygame.event.pump()
            joy_x, joy_y = 0.0, 0.0
            rx, ry = 0.0, 0.0
            rt_pulled = False
            lt_pulled = False
            btn_x = False
            btn_y = False
            btn_a = False
            btn_lb = False
            btn_rb = False

            if not self.controller and pygame.joystick.get_count() > 0:
                self.init_controller()

            if self.controller:
                # Left Stick (Pan/Tilt Steering)
                joy_x = self.controller.get_axis(0)
                joy_y = self.controller.get_axis(1)

                # Right Stick (Virtual Crosshair Movement)
                if self.controller.get_numaxes() >= 4:
                    rx = self.controller.get_axis(2)
                    ry = self.controller.get_axis(3)

                # Triggers (LT = Axis 4, RT = Axis 5)
                if self.controller.get_numaxes() >= 6:
                    lt_pulled = self.controller.get_axis(4) > 0.4
                    rt_pulled = self.controller.get_axis(5) > 0.4

                # Buttons (A=0, B=1, X=2, Y=3, LB=4, RB=5)
                num_b = self.controller.get_numbuttons()
                if num_b > 0: btn_a = bool(self.controller.get_button(0))
                if num_b > 2: btn_x = bool(self.controller.get_button(2))
                if num_b > 3: btn_y = bool(self.controller.get_button(3))
                if num_b > 4: btn_lb = bool(self.controller.get_button(4))
                if num_b > 5: btn_rb = bool(self.controller.get_button(5))

            if not rt_pulled: rt_released = True
            if not lt_pulled: lt_released = True
            if not btn_x: btn_x_released = True
            if not btn_y: btn_y_released = True
            if not btn_a: btn_a_released = True
            if not btn_lb: lb_released = True
            if not btn_rb: rb_released = True

            # --- VIRTUAL CROSSHAIR POSITION UPDATE ---
            if abs(rx) > 0.15: self.crosshair_x += rx * self.crosshair_speed
            if abs(ry) > 0.15: self.crosshair_y += ry * self.crosshair_speed
            self.crosshair_x = max(0.0, min(1280.0, self.crosshair_x))
            self.crosshair_y = max(0.0, min(720.0, self.crosshair_y))

            # --- CONTROLLER BUTTON ACTION CONTROLS ---
            # X Button: Toggle Laser
            if btn_x and btn_x_released:
                btn_x_released = False
                self.toggle_laser()

            # Y Button: Start Calibration
            if btn_y and btn_y_released:
                btn_y_released = False
                self.start_calibration()

            # A Button: Reset Target Locks
            if btn_a and btn_a_released:
                btn_a_released = False
                self.manual_target_pt = None
                self.locked_box_pt = None

            # LB / RB Buttons: Cycle Target Class
            if (btn_lb and lb_released) or (btn_rb and rb_released):
                if btn_lb: lb_released = False; self.class_idx = (self.class_idx - 1) % len(CLASS_NAMES_LIST)
                if btn_rb: rb_released = False; self.class_idx = (self.class_idx + 1) % len(CLASS_NAMES_LIST)
                cname = CLASS_NAMES_LIST[self.class_idx]
                self.target_class_id = COCO_CLASSES[cname]
                self.class_changed.emit(cname)

            target_x, target_y = None, None

            # --- YOLO DETECTION ---
            if self.target_class_id == -1:
                results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            else:
                results = self.model.predict(frame, classes=[self.target_class_id], conf=self.conf_threshold,
                                             verbose=False)

            detected_boxes = []
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{self.model.names[cls_id]} {conf:.2f}"
                    detected_boxes.append((x1, y1, x2, y2, label))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Left Trigger Action: Lock object inside crosshair or set static manual point
            if lt_pulled and lt_released:
                lt_released = False
                cx_int, cy_int = int(self.crosshair_x), int(self.crosshair_y)
                hit_box = None
                for box in detected_boxes:
                    bx1, by1, bx2, by2, _ = box
                    if bx1 <= cx_int <= bx2 and by1 <= cy_int <= by2:
                        hit_box = box
                        break

                if hit_box is not None:
                    # Lock onto moving object box center
                    box_cx = (hit_box[0] + hit_box[2]) // 2
                    box_cy = (hit_box[1] + hit_box[3]) // 2
                    self.locked_box_pt = (box_cx, box_cy)
                    self.manual_target_pt = None
                else:
                    # Lock static screen point
                    self.manual_target_pt = (cx_int, cy_int)
                    self.locked_box_pt = None

            # Determine Active Target
            if self.locked_box_pt is not None:
                # Dynamic Object Tracking Lock: Find box closest to locked_box_pt
                if len(detected_boxes) > 0:
                    best_box = None
                    min_dist = float('inf')
                    for b in detected_boxes:
                        bcx = (b[0] + b[2]) // 2
                        bcy = (b[1] + b[3]) // 2
                        dist = (bcx - self.locked_box_pt[0]) ** 2 + (bcy - self.locked_box_pt[1]) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_box = b

                    if best_box is not None:
                        target_x = (best_box[0] + best_box[2]) // 2
                        target_y = (best_box[1] + best_box[3]) // 2
                        self.locked_box_pt = (target_x, target_y)
                        cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (255, 0, 255), 3)
                        cv2.circle(frame, (target_x, target_y), 8, (255, 0, 255), -1)
                        cv2.putText(frame, "TRACKING OBJECT LOCK", (target_x + 12, target_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 255), 2)
            elif self.manual_target_pt is not None:
                # Fixed Point Lock
                target_x, target_y = self.manual_target_pt
                cv2.circle(frame, (target_x, target_y), 8, (0, 255, 255), -1)
                cv2.putText(frame, "POINT LOCK", (target_x + 12, target_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)
            elif len(detected_boxes) > 0:
                # Default Auto Tracking
                x1, y1, x2, y2, _ = detected_boxes[0]
                target_x = (x1 + x2) // 2
                target_y = (y1 + y2) // 2
                cv2.circle(frame, (target_x, target_y), 5, (0, 0, 255), -1)

            # --- DRAW VIRTUAL CROSSHAIR HUD ---
            ch_x, ch_y = int(self.crosshair_x), int(self.crosshair_y)
            cv2.circle(frame, (ch_x, ch_y), 16, (255, 255, 0), 2)
            cv2.line(frame, (ch_x - 22, ch_y), (ch_x - 6, ch_y), (255, 255, 0), 2)
            cv2.line(frame, (ch_x + 6, ch_y), (ch_x + 22, ch_y), (255, 255, 0), 2)
            cv2.line(frame, (ch_x, ch_y - 22), (ch_x, ch_y - 6), (255, 255, 0), 2)
            cv2.line(frame, (ch_x, ch_y + 6), (ch_x, ch_y + 22), (255, 255, 0), 2)

            # --- CALIBRATION VS TRACKING MODE ---
            if self.is_calibrating:
                stage_name = self.calib_stages[self.calib_index]

                # Steering during calibration
                if abs(joy_x) > 0.1: self.pan_angle += joy_x * joy_speed
                if abs(joy_y) > 0.1: self.tilt_angle += joy_y * joy_speed

                self.pan_angle = max(0.0, min(180.0, self.pan_angle))
                self.tilt_angle = max(0.0, min(180.0, self.tilt_angle))

                # RT to lock calibration stage
                if rt_pulled and rt_released:
                    rt_released = False

                    lock_x = target_x if target_x is not None else ch_x
                    lock_y = target_y if target_y is not None else ch_y

                    self.screen_points.append([lock_x, lock_y])
                    self.servo_points.append([self.pan_angle, self.tilt_angle])

                    self.calib_index += 1

                    if self.calib_index >= 4:
                        src_pts = np.array(self.screen_points, dtype=np.float32)
                        dst_pts = np.array(self.servo_points, dtype=np.float32)
                        self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
                        self.is_calibrating = False
                        self.status_updated.emit("Matrix Generated! Auto-Tracking Started.")

            else:
                # --- OG HOMOGRAPHY TRACKING ---
                if target_x is not None and self.homography_matrix is not None:
                    pt = np.array([[[target_x, target_y]]], dtype=np.float32)
                    mapped_pt = cv2.perspectiveTransform(pt, self.homography_matrix)

                    target_pan = mapped_pt[0][0][0]
                    target_tilt = mapped_pt[0][0][1]

                    self.pan_angle = (self.pan_angle * self.smoothing) + (target_pan * (1 - self.smoothing))
                    self.tilt_angle = (self.tilt_angle * self.smoothing) + (target_tilt * (1 - self.smoothing))

            # Constrain and send
            pan_out = int(max(0, min(180, self.pan_angle)))
            tilt_out = int(max(0, min(180, self.tilt_angle)))

            if self.arduino and self.arduino.is_open:
                try:
                    self.arduino.write(f"{pan_out},{tilt_out}\n".encode())
                except Exception:
                    pass

            self.frame_ready.emit(frame, detected_boxes, pan_out, tilt_out)
            time.sleep(0.015)

        cap.release()

    def start_calibration(self):
        self.screen_points = []
        self.servo_points = []
        self.calib_index = 0
        self.is_calibrating = True
        self.status_updated.emit("4-Point Calibration Started! Aim with Stick & Pull RT to Lock.")

    def stop(self):
        self.running = False
        self.wait()
        if self.arduino:
            self.arduino.close()


# --- MAIN GUI WINDOW ---
class LaserTurretApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Laser Turret Studio • Full Xbox Controller Edition")
        self.resize(1340, 860)

        self.bg_widget = AnimatedGradientWidget(self)
        self.setCentralWidget(self.bg_widget)

        main_layout = QHBoxLayout(self.bg_widget)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # --- LEFT PANEL: VIEWPORT ---
        self.viewport_card = QFrame()
        self.viewport_card.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.28);
                border: 1px solid rgba(255, 255, 255, 0.45);
                border-radius: 24px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setYOffset(8)
        self.viewport_card.setGraphicsEffect(shadow)

        vp_layout = QVBoxLayout(self.viewport_card)
        vp_layout.setContentsMargins(16, 16, 16, 16)

        header_layout = QHBoxLayout()
        title_label = QLabel("🎯 LIVE TARGETING VIEWPORT (Right-Stick Crosshair Active)")
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1A252C; background: transparent; border: none;")

        self.status_pill = QLabel("Status: Initializing...")
        self.status_pill.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.55);
            border-radius: 12px;
            padding: 4px 12px;
            font-weight: bold;
            font-size: 11px;
            color: #2C3E50;
            border: 1px solid rgba(255, 255, 255, 0.6);
        """)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_pill)
        vp_layout.addLayout(header_layout)

        self.video_display = ClickableVideoLabel()
        self.video_display.setMinimumSize(850, 520)
        self.video_display.setScaledContents(True)
        self.video_display.setStyleSheet("border-radius: 16px; background-color: #0F172A; border: none;")
        self.video_display.point_clicked.connect(self.on_video_clicked)
        vp_layout.addWidget(self.video_display, stretch=1)

        self.calib_info_label = QLabel("CALIBRATION MODE: Stage 1/4 (Top-Left)")
        self.calib_info_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.calib_info_label.setStyleSheet("""
            background-color: rgba(255, 230, 0, 0.85);
            color: #1E1E1E;
            border-radius: 10px;
            padding: 8px;
            text-align: center;
            border: none;
        """)
        vp_layout.addWidget(self.calib_info_label)

        main_layout.addWidget(self.viewport_card, stretch=3)

        # --- RIGHT PANEL: CONTROLS & THEMES ---
        self.control_card = QFrame()
        self.control_card.setMinimumWidth(360)
        self.control_card.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.32);
                border: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 24px;
            }
        """)
        shadow2 = QGraphicsDropShadowEffect()
        shadow2.setBlurRadius(25)
        shadow2.setColor(QColor(0, 0, 0, 40))
        shadow2.setYOffset(8)
        self.control_card.setGraphicsEffect(shadow2)

        ctrl_layout = QVBoxLayout(self.control_card)
        ctrl_layout.setContentsMargins(20, 24, 20, 24)
        ctrl_layout.setSpacing(16)

        ctrl_title = QLabel("⚙️ TURRET CONTROL")
        ctrl_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        ctrl_title.setStyleSheet("color: #1A252C; background: transparent; border: none;")
        ctrl_layout.addWidget(ctrl_title)

        # Serial Port Selector
        port_box_layout = QVBoxLayout()
        port_label = QLabel("🔌 Serial COM Port:")
        port_label.setStyleSheet("font-weight: bold; color: #2C3E50; background: transparent; border: none;")

        port_row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.style_combo(self.port_combo)

        self.btn_reconnect = QPushButton("Connect")
        self.btn_reconnect.setStyleSheet("""
            QPushButton {
                background: #007AFF; color: white; font-weight: bold;
                border-radius: 10px; padding: 8px 14px; border: none;
            }
            QPushButton:hover { background: #0056B3; }
        """)
        self.btn_reconnect.clicked.connect(self.reconnect_serial)

        port_row.addWidget(self.port_combo, stretch=1)
        port_row.addWidget(self.btn_reconnect)
        port_box_layout.addWidget(port_label)
        port_box_layout.addLayout(port_row)
        ctrl_layout.addLayout(port_box_layout)

        # Theme Switcher
        theme_box_layout = QVBoxLayout()
        theme_label = QLabel("🎨 Background Theme:")
        theme_label.setStyleSheet("font-weight: bold; color: #2C3E50; background: transparent; border: none;")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEMES.keys()))
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        self.style_combo(self.theme_combo)
        theme_box_layout.addWidget(theme_label)
        theme_box_layout.addWidget(self.theme_combo)
        ctrl_layout.addLayout(theme_box_layout)

        # Target Class Selector
        class_box_layout = QVBoxLayout()
        class_label = QLabel("🔍 Target Object Class (LB/RB to Cycle):")
        class_label.setStyleSheet("font-weight: bold; color: #2C3E50; background: transparent; border: none;")
        self.class_combo = QComboBox()
        self.class_combo.addItems(CLASS_NAMES_LIST)
        self.class_combo.currentTextChanged.connect(self.on_class_changed)
        self.style_combo(self.class_combo)
        class_box_layout.addWidget(class_label)
        class_box_layout.addWidget(self.class_combo)
        ctrl_layout.addLayout(class_box_layout)

        # Confidence Slider
        conf_layout = QVBoxLayout()
        conf_hdr = QHBoxLayout()
        conf_title = QLabel("🎯 Confidence Threshold:")
        conf_title.setStyleSheet("font-weight: bold; color: #2C3E50; background: transparent; border: none;")
        self.conf_val_lbl = QLabel("0.45")
        self.conf_val_lbl.setStyleSheet("font-weight: bold; color: #007AFF; background: transparent; border: none;")
        conf_hdr.addWidget(conf_title)
        conf_hdr.addStretch()
        conf_hdr.addWidget(self.conf_val_lbl)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(15, 90)
        self.conf_slider.setValue(45)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        self.conf_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: rgba(255,255,255,0.6); border-radius: 3px; }
            QSlider::handle:horizontal { background: #007AFF; width: 18px; margin: -6px 0; border-radius: 9px; }
        """)
        conf_layout.addLayout(conf_hdr)
        conf_layout.addWidget(self.conf_slider)
        ctrl_layout.addLayout(conf_layout)

        # Controller Shortcuts Info Box
        shortcuts_box = QFrame()
        shortcuts_box.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.45);
                border-radius: 14px;
                padding: 10px;
                border: 1px solid rgba(255, 255, 255, 0.6);
            }
        """)
        sc_layout = QVBoxLayout(shortcuts_box)
        sc_title = QLabel("🎮 CONTROLLER SHORTCUTS")
        sc_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        sc_title.setStyleSheet("color: #0F172A; background: transparent; border: none;")

        sc_text = QLabel(
            "• R-Stick: Virtual Crosshair\n• LT: Lock Point / Bounding Box\n• X Button: Toggle Laser ON/OFF\n• Y Button: Start 4-Point Calib\n• A Button: Clear Manual Target\n• LB/RB: Cycle Target Class")
        sc_text.setStyleSheet("font-size: 11px; color: #334155; background: transparent; border: none;")
        sc_layout.addWidget(sc_title)
        sc_layout.addWidget(sc_text)
        ctrl_layout.addWidget(shortcuts_box)

        # 4-Point Calibration Button
        self.btn_calib = QPushButton("📐 Start 4-Point Calibration (Y)")
        self.btn_calib.setStyleSheet("""
            QPushButton {
                background: linear-gradient(135deg, #007AFF, #00C6FF);
                color: white; font-weight: bold; font-size: 13px;
                border-radius: 14px; padding: 10px; border: none;
            }
            QPushButton:hover { background: linear-gradient(135deg, #0056B3, #00A3E0); }
        """)
        self.btn_calib.clicked.connect(self.start_4point_calibration)
        ctrl_layout.addWidget(self.btn_calib)

        # Telemetry Card
        telemetry_box = QFrame()
        telemetry_box.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.4);
                border-radius: 16px; padding: 8px 12px;
                border: 1px solid rgba(255, 255, 255, 0.5);
            }
        """)
        telem_layout = QVBoxLayout(telemetry_box)
        self.lbl_pan = QLabel("Pan Angle: 90°")
        self.lbl_tilt = QLabel("Tilt Angle: 90°")
        self.lbl_pan.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #0F172A; background: transparent; border: none;")
        self.lbl_tilt.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #0F172A; background: transparent; border: none;")
        telem_layout.addWidget(self.lbl_pan)
        telem_layout.addWidget(self.lbl_tilt)
        ctrl_layout.addWidget(telemetry_box)

        ctrl_layout.addStretch()
        main_layout.addWidget(self.control_card, stretch=1)

        self.worker = TrackingThread()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_updated.connect(self.update_status)
        self.worker.class_changed.connect(self.on_worker_class_changed)
        self.worker.start()

    def refresh_ports(self):
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            ports = ["COM4", "COM5", "COM3"]
        self.port_combo.addItems(ports)
        if "COM4" in ports:
            self.port_combo.setCurrentText("COM4")

    def reconnect_serial(self):
        selected_port = self.port_combo.currentText()
        self.worker.init_serial(selected_port)

    def style_combo(self, combo):
        combo.setStyleSheet("""
            QComboBox {
                background: rgba(255, 255, 255, 0.65);
                border: 1px solid rgba(255, 255, 255, 0.8);
                border-radius: 12px; padding: 8px 12px;
                font-weight: bold; color: #0F172A;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF; border: 1px solid #CBD5E1;
                border-radius: 10px; color: #0F172A;
                selection-background-color: #007AFF; selection-color: #FFFFFF; padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 28px; color: #0F172A; background-color: #FFFFFF;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #007AFF; color: #FFFFFF;
            }
        """)

    def on_theme_changed(self, theme_name):
        self.bg_widget.set_theme(theme_name)

    def on_class_changed(self, class_name):
        cls_id = COCO_CLASSES.get(class_name, -1)
        self.worker.target_class_id = cls_id
        if class_name in CLASS_NAMES_LIST:
            self.worker.class_idx = CLASS_NAMES_LIST.index(class_name)

    def on_worker_class_changed(self, class_name):
        self.class_combo.blockSignals(True)
        self.class_combo.setCurrentText(class_name)
        self.class_combo.blockSignals(False)

    def on_conf_changed(self, val):
        conf = val / 100.0
        self.conf_val_lbl.setText(f"{conf:.2f}")
        self.worker.conf_threshold = conf

    def start_4point_calibration(self):
        self.worker.start_calibration()

    def clear_manual_target(self):
        self.worker.manual_target_pt = None

    def on_video_clicked(self, x, y):
        self.worker.manual_target_pt = (x, y)

    def update_frame(self, frame, detected_boxes, pan_angle, tilt_angle):
        self.lbl_pan.setText(f"Pan Angle: {pan_angle}°")
        self.lbl_tilt.setText(f"Tilt Angle: {tilt_angle}°")
        if self.worker.is_calibrating:
            stage_num = self.worker.calib_index + 1
            stage_name = self.worker.calib_stages[self.worker.calib_index]
            self.calib_info_label.setText(
                f"CALIBRATION STAGE {stage_num}/4: Stand {stage_name} (Aim with Stick, RT to Lock)")
            self.calib_info_label.setVisible(True)
        else:
            self.calib_info_label.setVisible(False)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_display.setPixmap(pixmap)

    def update_status(self, msg):
        self.status_pill.setText(msg)

    def closeEvent(self, event):
        self.worker.stop()
        pygame.quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LaserTurretApp()
    window.show()
    sys.exit(app.exec())