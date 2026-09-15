"""
Stratum Showcase Application: Real-time Computer Vision Camera
=============================================================
High-performance NanoDet Object Detection with OpenCV Zoo ONNX support.
Optimized for mobile real-time inference via Stratum & Chaquopy.
"""

import os
import sys
import time
import queue
import threading
import traceback
from collections import deque

import stratum

# ── Stratum Android View & Graphics Wrappers ──────────────────────────────────
from stratum.android.view.View import View
from stratum.android.view.TextureView import TextureView
from stratum.android.view.Surface import Surface
from stratum.android.view.MotionEvent import MotionEvent
from stratum.android.widget.FrameLayout import FrameLayout
from stratum.android.widget.LinearLayout import LinearLayout
from stratum.android.widget.Button import Button
from stratum.android.widget.TextView import TextView
from stratum.android.widget.ImageView import ImageView
from stratum.android.graphics.Matrix import Matrix
from stratum.android.graphics.Rect import Rect
from stratum.android.graphics.SurfaceTexture import SurfaceTexture
from stratum.android.graphics.Bitmap_CompressFormat import Bitmap_CompressFormat
from stratum.android.os.Looper import Looper
from stratum.android.os.Handler import Handler

# ── Camera2 Hardware Wrappers ────────────────────────────────────────────────
from stratum.android.hardware.camera2.CameraManager import CameraManager
from stratum.android.hardware.camera2.CameraDevice import CameraDevice
from stratum.android.hardware.camera2.CameraCharacteristics import CameraCharacteristics
from stratum.android.hardware.camera2.CaptureRequest import CaptureRequest
from stratum.android.hardware.camera2.CameraCaptureSession import CameraCaptureSession
from stratum.android.hardware.camera2.params.StreamConfigurationMap import StreamConfigurationMap
from stratum.android.hardware.camera2.params.MeteringRectangle import MeteringRectangle
from stratum.android.util.Size import Size

# ── Java Reflection for Native Array Instantiation ────────────────────────────
from stratum.java.lang.Class import Class
from stratum.java.lang.reflect.Array import Array

# ── Hardware Sensor Wrappers ─────────────────────────────────────────────────
from stratum.android.hardware.SensorManager import SensorManager
from stratum.android.hardware.Sensor import Sensor

# ── MediaStore Persistence Wrappers ──────────────────────────────────────────
from stratum.android.content.ContentValues import ContentValues
from stratum.android.provider.MediaStore_Images_Media import MediaStore_Images_Media

# ── Computer Vision Libraries ────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception as err:
    OPENCV_AVAILABLE = False
    print(f"[CV ERROR] OpenCV/NumPy unavailable: {err}")

stratum.set_log_enabled(False)

# ── NANODET CONFIGURATION CONSTANTS ──────────────────────────────────────────
CANDIDATE_MODEL_NAMES = [
    "object_detection_nanodet_2022nov.onnx",
]

NANODET_INPUT_SIZE = 416
NANODET_STRIDES = (8, 16, 32)
NANODET_REG_MAX = 7
NANODET_DETECT_INTERVAL = 2

# Thresholds for mobile real-time NanoDet
NANODET_SCORE_THRESHOLD = 0.35
NANODET_NMS_THRESHOLD = 0.50

# Normalization constants (BGR format)
NANODET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
NANODET_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def _cast(obj, cls):
    """Safely casts a native JNI global reference (_ptr) into a typed Stratum wrapper."""
    if obj is None:
        return None
    return cls.from_ptr(obj)


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class SlidingFpsTracker:
    def __init__(self, window_size: int = 20, stall_timeout: float = 0.8):
        self.timestamps = deque(maxlen=window_size)
        self.stall_timeout = stall_timeout
        self._lock = threading.Lock()

    def tick(self) -> float | None:
        now = time.perf_counter()
        with self._lock:
            self.timestamps.append(now)
            return self._calc_fps(now)

    def _calc_fps(self, now: float) -> float | None:
        if not self.timestamps or (now - self.timestamps[-1]) > self.stall_timeout:
            return 0.0
        if len(self.timestamps) < 3:
            return None
        try:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed <= 0.0001:
                return None
            return (len(self.timestamps) - 1) / elapsed
        except IndexError:
            return None

    def reset(self):
        with self._lock:
            self.timestamps.clear()


class CameraApp:
    def __init__(self, activity):
        self.activity = activity
        self.res = activity.getResources()
        self.pkg = activity.getPackageName()

        self.cam_mgr = None
        self.camera_device = None
        self.capture_session = None
        self.builder = None
        self.handler = None

        self.camera_ids = []
        self.current_cam_idx = 0
        self.sensor_orient = 90
        self.facing_front = False

        self.zoom_level = 1.0
        self.max_zoom = 1.0
        self.active_array_size = None

        self.preview_width = 1280
        self.preview_height = 720

        self.cv_mode = 0
        self.mode_labels = [
            "Normal (Raw)", "ORB Features", "White Edges", "Quad Tracker", "B&W Threshold", "NanoDet AI"
        ]
        self.mode_icons = ["⚪ Normal", "⚡ ORB", "◈ Edges", "⛶ Quad", "◐ B&W", "🎯 NanoDet"]

        self.running = True
        self.is_switching = False
        self.worker_busy = False
        self.first_frame_rendered = False
        self.frame_event = threading.Event()

        self.hw_fps_tracker = SlidingFpsTracker(window_size=25)
        self.cv_fps_tracker = SlidingFpsTracker(window_size=15)
        self.last_hud_update = 0.0

        self.last_sensor_time = 0.0
        self.current_ui_rotation = 0.0

        self.save_queue = queue.Queue(maxsize=8)
        self.bmp_lock = threading.Lock()

        # NanoDet Neural Network State
        self.nanodet_net = None
        self.nanodet_error_msg = None
        self.nanodet_error_detail = None
        self.nanodet_frame_counter = 0
        self.nanodet_last_detections = []
        self._nanodet_grids = {}
        self._project = np.arange(NANODET_REG_MAX + 1, dtype=np.float32)

        self._init_nanodet_dnn()
        self._inflate_and_bind_ui()
        self._init_orientation_sensor()

        self.processing_thread = threading.Thread(target=self._cv_worker_loop, daemon=True)
        self.processing_thread.start()

        self.saver_thread = threading.Thread(target=self._save_worker_loop, daemon=True)
        self.saver_thread.start()

    # ── NANODET DNN LOADER & ASSET EXTRACTION ────────────────────────────────
    def _extract_asset_if_needed(self, model_name: str, target_path: str) -> bool:
        from com.chaquo.python import Python
        from java.io import FileOutputStream
        from java import jbyte, jarray

        app_context = Python.getPlatform().getApplication()
        try:
            in_stream = app_context.getAssets().open(model_name)
        except Exception:
            return False

        need_copy = True
        if os.path.exists(target_path) and os.path.getsize(target_path) > 50000:
            need_copy = False

        if need_copy:
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass

            print(f"[NanoDet] Extracting {model_name} to {target_path} ...")
            out_stream = FileOutputStream(target_path)
            buf = jarray(jbyte)(65536)
            total_bytes = 0
            while True:
                n = in_stream.read(buf)
                if n <= 0:
                    break
                out_stream.write(buf, 0, n)
                total_bytes += n
            out_stream.flush()
            out_stream.close()
            print(f"[NanoDet] Extracted {model_name} successfully ({total_bytes} bytes).")

        in_stream.close()
        return os.path.exists(target_path) and os.path.getsize(target_path) > 50000

    def _init_nanodet_dnn(self):
        if not OPENCV_AVAILABLE:
            self.nanodet_error_msg = "OpenCV unavailable"
            return

        files_dir = None
        try:
            files_dir = str(self.activity.getFilesDir().getAbsolutePath())
        except Exception:
            try:
                from com.chaquo.python import Python
                files_dir = str(Python.getPlatform().getApplication().getFilesDir().getAbsolutePath())
            except Exception:
                files_dir = os.path.expanduser("~")

        os.makedirs(files_dir, exist_ok=True)

        target_path = None
        found_model = None
        for cand in CANDIDATE_MODEL_NAMES:
            cand_path = os.path.join(files_dir, cand)
            if self._extract_asset_if_needed(cand, cand_path):
                found_model = cand
                target_path = cand_path
                break

        if not target_path or not os.path.exists(target_path):
            self.nanodet_error_msg = "Model asset not found"
            self.nanodet_error_detail = f"Add {CANDIDATE_MODEL_NAMES[0]} to assets"
            print(f"[NanoDet] {self.nanodet_error_msg} — {self.nanodet_error_detail}")
            return

        try:
            print(f"[NanoDet] Loading {found_model} into cv2.dnn ({os.path.getsize(target_path)} bytes)...")
            net = cv2.dnn.readNet(target_path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            self.nanodet_net = net

            # Precompute coordinate grids for strides
            for stride in NANODET_STRIDES:
                feat_h = NANODET_INPUT_SIZE // stride
                feat_w = NANODET_INPUT_SIZE // stride
                xv, yv = np.meshgrid(np.arange(feat_w), np.arange(feat_h))
                grid = np.stack((xv, yv), axis=-1).reshape(-1, 2).astype(np.float32)
                self._nanodet_grids[stride] = grid

            self.nanodet_error_msg = None
            self.nanodet_error_detail = None
            print(f"[NanoDet] {found_model} successfully loaded and initialized!")

        except Exception as err:
            err_type = type(err).__name__
            err_str = str(err)
            print(f"[NanoDet INIT ERROR] {err_type}: {err_str}")
            traceback.print_exc()
            self.nanodet_error_msg = f"{err_type}"
            self.nanodet_error_detail = err_str[:120]

    # ── UI INFLATION & VIEW BINDING ───────────────────────────────────────────
    def _res_id(self, name: str, res_type: str = "id") -> int:
        res_id = self.res.getIdentifier(name, res_type, self.pkg)
        if res_id == 0:
            raise KeyError(f"Resource '{name}' of type '{res_type}' not found in package '{self.pkg}'")
        return res_id

    def _bind(self, name: str, cls):
        return _cast(self.root.findViewById(self._res_id(name, "id")), cls)

    def _inflate_and_bind_ui(self):
        inflater = self.activity.getLayoutInflater()
        layout_id = self._res_id("activity_camera", "layout")
        self.root = _cast(inflater.inflate(layout_id, None, False), FrameLayout)
        stratum.setContentView(self.activity, self.root)

        self.texture_view = self._bind("texture_view", TextureView)
        self.image_view = self._bind("image_view", ImageView)
        self.boot_cover = self._bind("boot_cover", FrameLayout)
        self.flash_overlay = self._bind("flash_overlay", View)
        self.focus_ring = self._bind("focus_ring", View)

        self.texture_view.setOnTouchListener(self._on_viewfinder_touch)
        self.texture_view.setSurfaceTextureListener({
            "onSurfaceTextureAvailable":   self._on_surface_available,
            "onSurfaceTextureSizeChanged": self._on_surface_size_changed,
            "onSurfaceTextureDestroyed":   self._on_surface_destroyed,
            "onSurfaceTextureUpdated":     self._on_surface_updated,
        })

        self.hud_badge = self._bind("hud_badge", LinearLayout)
        self.txt_fps = self._bind("txt_fps", TextView)
        self.txt_mode = self._bind("txt_mode", TextView)
        self.save_toast = self._bind("save_toast", TextView)

        self.dock = self._bind("dock", LinearLayout)
        self.btn_zoom_out = self._bind("btn_zoom_out", Button)
        self.btn_zoom_out.setOnClickListener(self._on_zoom_out)

        self.btn_mode = self._bind("btn_mode", Button)
        self.btn_mode.setOnClickListener(self._on_toggle_mode)

        self.shutter_container = self._bind("shutter_container", FrameLayout)
        self.btn_capture = self._bind("btn_capture", Button)
        self.btn_capture.setOnClickListener(self._on_capture)

        self.btn_switch = self._bind("btn_switch", Button)
        self.btn_switch.setOnClickListener(self._on_switch)

        self.btn_zoom_in = self._bind("btn_zoom_in", Button)
        self.btn_zoom_in.setOnClickListener(self._on_zoom_in)

        self.rotating_views = [
            self.btn_zoom_out, self.btn_mode, self.btn_capture,
            self.btn_switch, self.btn_zoom_in, self.hud_badge, self.save_toast,
        ]

    # ── TOUCH TO AUTOFOCUS & EXPOSURE ─────────────────────────────────────────
    def _on_viewfinder_touch(self, view, motion_event) -> bool:
        raw_event = _cast(motion_event, MotionEvent)
        if raw_event.getAction() != 0:
            return False
        touch_x = raw_event.getX()
        touch_y = raw_event.getY()
        self._animate_focus_ring(touch_x, touch_y)
        self._trigger_touch_focus(touch_x, touch_y)
        return True

    def _animate_focus_ring(self, x: float, y: float):
        if not self.handler:
            return
        ring_half = 36.0
        self.focus_ring.setX(x - ring_half)
        self.focus_ring.setY(y - ring_half)
        self.focus_ring.setScaleX(1.3)
        self.focus_ring.setScaleY(1.3)
        self.focus_ring.setAlpha(1.0)

        def pulse(step=0):
            if step >= 5:
                self.handler.postDelayed(lambda: self.focus_ring.setAlpha(0.0), 400)
                return
            scale = max(1.0, 1.3 - (step * 0.06))
            self.focus_ring.setScaleX(scale)
            self.focus_ring.setScaleY(scale)
            self.handler.postDelayed(lambda pulse_step=step+1: pulse(pulse_step), 20)

        self.handler.post(pulse)

    def _trigger_touch_focus(self, touch_x: float, touch_y: float):
        if not self.capture_session or not self.builder or not self.active_array_size:
            return
        try:
            vw = max(float(self.texture_view.getWidth()), 1.0)
            vh = max(float(self.texture_view.getHeight()), 1.0)
            array_w = float(self.active_array_size.width())
            array_h = float(self.active_array_size.height())

            crop_w = array_w / self.zoom_level
            crop_h = array_h / self.zoom_level
            crop_x = (array_w - crop_w) / 2.0
            crop_y = (array_h - crop_h) / 2.0

            norm_x = max(0.0, min(1.0, touch_x / vw))
            norm_y = max(0.0, min(1.0, touch_y / vh))

            if self.facing_front:
                sensor_x = crop_x + (norm_y * crop_w)
                sensor_y = crop_y + (norm_x * crop_h)
            else:
                sensor_x = crop_x + (norm_y * crop_w)
                sensor_y = crop_y + ((1.0 - norm_x) * crop_h)

            box_size_w = crop_w * 0.12
            box_size_h = crop_h * 0.12

            left = int(max(crop_x, sensor_x - (box_size_w / 2.0)))
            top = int(max(crop_y, sensor_y - (box_size_h / 2.0)))
            right = int(min(crop_x + crop_w, left + box_size_w))
            bottom = int(min(crop_y + crop_h, top + box_size_h))

            focus_rect = Rect(left, top, right, bottom)
            metering_rect = MeteringRectangle(focus_rect, 1000)

            if not hasattr(self, "_metering_rect_class"):
                self._metering_rect_class = Class.forName("android.hardware.camera2.params.MeteringRectangle")

            metering_array = Array.newInstance(self._metering_rect_class, 1)
            Array.set(metering_array, 0, metering_rect)

            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 2)
            self.capture_session.capture(self.builder.build(), None, self.handler)

            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_REGIONS(), metering_array)
            self.builder.set(CaptureRequest.sf_get_CONTROL_AE_REGIONS(), metering_array)
            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_MODE(), 1)
            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 0)

            self.capture_session.setRepeatingRequest(self.builder.build(), None, self.handler)

            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 1)
            self.capture_session.capture(self.builder.build(), None, self.handler)
            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 0)
        except Exception as err:
            print(f"[TOUCH FOCUS ERROR] {err}")

    # ── HARDWARE SENSORS & DYNAMIC ROTATION ───────────────────────────────────
    def _init_orientation_sensor(self):
        try:
            sensor_svc = self.activity.getSystemService("sensor")
            self.sensor_mgr = _cast(sensor_svc, SensorManager)
            if self.sensor_mgr:
                accel = self.sensor_mgr.getDefaultSensor(1)
                if accel:
                    self.sensor_mgr.registerListener({
                        "onSensorChanged": self._on_sensor_changed,
                        "onAccuracyChanged": lambda s, a: None,
                    }, accel, 3)
        except Exception as err:
            print(f"[SENSOR INIT ERROR] {err}")

    def _on_sensor_changed(self, event):
        now = time.time()
        if (now - self.last_sensor_time) < 0.120:
            return
        self.last_sensor_time = now
        try:
            vals = event.f_get_values()
            if not vals or len(vals) < 2:
                return
            x, y = float(vals[0]), float(vals[1])
            target_deg = 0.0
            if abs(x) > abs(y):
                target_deg = 90.0 if x > 3.0 else (270.0 if x < -3.0 else self.current_ui_rotation)
            else:
                target_deg = 180.0 if y < -3.0 else 0.0
            if target_deg != self.current_ui_rotation:
                self.current_ui_rotation = target_deg
                if self.handler:
                    self.handler.post(lambda deg=target_deg: self._apply_ui_rotation(deg))
        except Exception:
            pass

    def _apply_ui_rotation(self, deg: float):
        for v in self.rotating_views:
            try:
                v.setRotation(deg)
            except Exception:
                pass

    # ── TRANSITIONS & FEEDBACK OVERLAYS ──────────────────────────────────────
    def _dismiss_boot_cover(self):
        if not self.handler or self.first_frame_rendered:
            return
        self.first_frame_rendered = True

        def fade(step=0):
            if step >= 5:
                self.boot_cover.setVisibility(8)
                return
            self.boot_cover.setAlpha(max(0.0, 1.0 - (step * 0.22)))
            self.handler.postDelayed(lambda s=step+1: fade(s), 25)

        self.handler.post(fade)

    def _trigger_shutter_flash(self):
        if not self.handler:
            return
        self.flash_overlay.setAlpha(0.7)

        def fade_flash(step=0):
            if step >= 4:
                self.flash_overlay.setAlpha(0.0)
                return
            self.flash_overlay.setAlpha(max(0.0, 0.7 - (step * 0.2)))
            self.handler.postDelayed(lambda s=step+1: fade_flash(s), 25)

        self.handler.post(fade_flash)

    def _show_save_toast(self):
        if not self.handler:
            return
        self.save_toast.setAlpha(1.0)
        self.handler.postDelayed(lambda: self.save_toast.setAlpha(0.0), 1300)

    # ── CAPTURE & ASYNC STORAGE ──────────────────────────────────────────────
    def _on_capture(self, view):
        self._trigger_shutter_flash()
        snapshot = None
        with self.bmp_lock:
            if self.cv_mode == 0:
                snapshot = self.texture_view.getBitmap(self.preview_width, self.preview_height)
            else:
                active_bmp = getattr(self, "out_bmp", None)
                if active_bmp is not None:
                    try:
                        snapshot = active_bmp.copy(active_bmp.getConfig(), False)
                    except Exception as err:
                        print(f"[CAPTURE SNAPSHOT ERROR] {err}")
        if snapshot is not None:
            try:
                self.save_queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def _save_worker_loop(self):
        while self.running:
            try:
                bmp = self.save_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                resolver = self.activity.getContentResolver()
                values = ContentValues()
                fname = f"stratum_{int(time.time() * 1000)}.jpg"
                values.put("_display_name", fname)
                values.put("mime_type", "image/jpeg")
                values.put("relative_path", "Pictures/Stratum")

                uri = resolver.insert(MediaStore_Images_Media.sf_get_EXTERNAL_CONTENT_URI(), values)
                if uri:
                    out_stream = resolver.openOutputStream(uri)
                    if out_stream:
                        jpeg_format = Bitmap_CompressFormat.sf_get_JPEG()
                        bmp.compress(jpeg_format, 95, out_stream)
                        out_stream.close()
                        if self.handler:
                            self.handler.post(self._show_save_toast)
            except Exception as err:
                print(f"[DISK SAVE ERROR] {err}")
            finally:
                self.save_queue.task_done()

    # ── CAMERA2 TRANSFORM & GEOMETRY ─────────────────────────────────────────
    def _configure_transform(self, view_w: float, view_h: float):
        if view_w <= 0 or view_h <= 0 or self.preview_width <= 0 or self.preview_height <= 0:
            return
        try:
            matrix = Matrix()
            cx = view_w / 2.0
            cy = view_h / 2.0
            buf_w = float(self.preview_width)
            buf_h = float(self.preview_height)
            scale = max(view_w / buf_h, view_h / buf_w)
            scale_x = (buf_h / view_w) * scale
            scale_y = (buf_w / view_h) * scale

            if self.facing_front:
                matrix.postRotate(270.0, cx, cy)
                matrix.postScale(-scale_x, scale_y, cx, cy)
            else:
                matrix.postRotate(90.0, cx, cy)
                matrix.postScale(scale_x, scale_y, cx, cy)

            self.texture_view.setTransform(matrix)
        except Exception as err:
            print(f"[TRANSFORM ERROR] {err}")

    def _clear_buffers(self):
        with self.bmp_lock:
            for attr in ("in_bmp", "out_bmp", "bb_wrapper", "arr", "gray_buf"):
                if hasattr(self, attr):
                    delattr(self, attr)
        self.nanodet_frame_counter = 0
        self.nanodet_last_detections = []

    def _apply_zoom(self):
        if not self.capture_session or not self.builder or not self.active_array_size:
            return
        try:
            full_w = self.active_array_size.width()
            full_h = self.active_array_size.height()
            zoom = max(1.0, min(float(self.zoom_level), float(self.max_zoom)))

            crop_w = int(full_w / zoom)
            crop_h = int(full_h / zoom)
            crop_x = (full_w - crop_w) // 2
            crop_y = (full_h - crop_h) // 2

            crop_rect = Rect(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
            if crop_rect:
                crop_key = CaptureRequest.sf_get_SCALER_CROP_REGION()
                self.builder.set(crop_key, crop_rect)

            req = self.builder.build()
            self.capture_session.setRepeatingRequest(req, None, self.handler)
        except Exception as err:
            print(f"[ZOOM ERROR] {err}")

    def _on_zoom_in(self, view):
        self.zoom_level = min(float(self.max_zoom), round(self.zoom_level + 0.5, 1))
        self._apply_zoom()

    def _on_zoom_out(self, view):
        self.zoom_level = max(1.0, round(self.zoom_level - 0.5, 1))
        self._apply_zoom()

    def _on_switch(self, view):
        if len(self.camera_ids) < 2 or self.is_switching:
            return
        self.is_switching = True
        self.frame_event.set()
        wait_start = time.time()
        while self.worker_busy and (time.time() - wait_start) < 0.4:
            time.sleep(0.005)
        try:
            self._shutdown_camera()
            self._clear_buffers()
            self.current_cam_idx = (self.current_cam_idx + 1) % len(self.camera_ids)
            self.zoom_level = 1.0
            self._open_camera()
        except Exception as err:
            print(f"[SWITCH ERROR] {err}")
            self.is_switching = False

    def _choose_optimal_size(self, cam_id, view_w, view_h):
        try:
            chars = self.cam_mgr.getCameraCharacteristics(cam_id)
            map_obj = chars.get(CameraCharacteristics.sf_get_SCALER_STREAM_CONFIGURATION_MAP())
            stream_map = _cast(map_obj, StreamConfigurationMap)

            sizes_array = stream_map.getOutputSizes(34)
            if sizes_array is None:
                sizes_array = stream_map.getOutputSizes(256)

            if sizes_array is not None:
                best_w, best_h = 1280, 720
                min_diff = float("inf")
                screen_ratio = max(view_w, view_h) / max(1, min(view_w, view_h))

                for sz_obj in sizes_array:
                    if sz_obj is None:
                        continue
                    sz = _cast(sz_obj, Size)
                    w, h = sz.getWidth(), sz.getHeight()
                    if w * h <= 1920 * 1080 and w > h:
                        diff = abs((w / h) - screen_ratio)
                        if diff < min_diff:
                            min_diff = diff
                            best_w, best_h = w, h

                self.preview_width, self.preview_height = best_w, best_h
        except Exception:
            self.preview_width, self.preview_height = 1280, 720

    def _read_camera_info(self, cam_id):
        try:
            chars = self.cam_mgr.getCameraCharacteristics(cam_id)
            orient_obj = chars.get(CameraCharacteristics.sf_get_SENSOR_ORIENTATION())
            facing_obj = chars.get(CameraCharacteristics.sf_get_LENS_FACING())
            mz_obj = chars.get(CameraCharacteristics.sf_get_SCALER_AVAILABLE_MAX_DIGITAL_ZOOM())
            ar_obj = chars.get(CameraCharacteristics.sf_get_SENSOR_INFO_ACTIVE_ARRAY_SIZE())

            self.sensor_orient = int(orient_obj) if isinstance(orient_obj, (int, float)) else 90
            self.facing_front = (int(facing_obj) == 0) if isinstance(facing_obj, (int, float)) else False
            self.max_zoom = max(1.0, float(mz_obj)) if isinstance(mz_obj, (int, float)) else 5.0
            self.active_array_size = _cast(ar_obj, Rect) if ar_obj is not None else None
        except Exception as err:
            print(f"[CAM INFO ERROR] {err}")

    def _shutdown_camera(self):
        if self.capture_session:
            try:
                self.capture_session.close()
            except Exception:
                pass
            self.capture_session = None
        if self.camera_device:
            try:
                self.camera_device.close()
            except Exception:
                pass
            self.camera_device = None

    def _open_camera(self):
        try:
            cam_id = self.camera_ids[self.current_cam_idx]
            self._read_camera_info(cam_id)

            vw = max(float(self.texture_view.getWidth()), 1.0)
            vh = max(float(self.texture_view.getHeight()), 1.0)
            self._choose_optimal_size(cam_id, vw, vh)

            st_raw = self.texture_view.getSurfaceTexture()
            if st_raw:
                st = _cast(st_raw, SurfaceTexture)
                st.setDefaultBufferSize(self.preview_width, self.preview_height)

            self._configure_transform(vw, vh)

            self.cam_mgr.openCamera(cam_id, {
                "onOpened":       self._on_camera_opened,
                "onDisconnected": lambda dev: None,
                "onError":        lambda dev, err: setattr(self, "is_switching", False),
            }, self.handler)
        except Exception as err:
            print(f"[OPEN CAMERA ERROR] {err}")

    def _on_surface_available(self, st, w, h):
        try:
            sys_svc = self.activity.getSystemService("camera")
            self.cam_mgr = _cast(sys_svc, CameraManager)

            raw_ids = self.cam_mgr.getCameraIdList()
            self.camera_ids = [str(cid) for cid in raw_ids] if raw_ids else ["0"]

            looper = Looper.getMainLooper()
            self.handler = self.handler or Handler(looper)

            self._open_camera()
        except Exception as err:
            print(f"[SURFACE AVAILABLE ERROR] {err}")

    def _on_surface_size_changed(self, st, w, h):
        self.is_switching = True
        self._clear_buffers()
        if w > 0 and h > 0:
            self._configure_transform(float(w), float(h))
        self.is_switching = False

    def _on_surface_destroyed(self, st):
        self.shutdown()
        return True

    def _on_surface_updated(self, st):
        if self.is_switching:
            return

        current_hw_fps = self.hw_fps_tracker.tick()

        if self.cv_mode == 0:
            if self.image_view.getVisibility() != 8:
                self.image_view.setVisibility(8)
            if not self.first_frame_rendered:
                self._dismiss_boot_cover()

            now = time.perf_counter()
            if current_hw_fps is not None and (now - self.last_hud_update >= 0.25):
                self.last_hud_update = now
                self.txt_fps.setText(f"FPS: {round(current_hw_fps)}")
            return

        if self.image_view.getVisibility() != 0:
            self.image_view.setVisibility(0)

        if not self.worker_busy:
            self.worker_busy = True
            self.frame_event.set()

    def _on_camera_opened(self, raw_device):
        try:
            self.camera_device = _cast(raw_device, CameraDevice)
            st_raw = self.texture_view.getSurfaceTexture()
            st = _cast(st_raw, SurfaceTexture)
            self.surface = Surface(st)

            self.builder = self.camera_device.createCaptureRequest(1)
            self.builder.addTarget(self.surface)
            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_MODE(), 4)

            self.camera_device.createCaptureSession([self.surface], {
                "onConfigured":      self._on_session_configured,
                "onConfigureFailed": lambda s: setattr(self, "is_switching", False),
            }, self.handler)
        except Exception as err:
            print(f"[SESSION CREATE ERROR] {err}")

    def _on_session_configured(self, raw_session):
        try:
            self.capture_session = _cast(raw_session, CameraCaptureSession)
            self._apply_zoom()
            self.is_switching = False
        except Exception as err:
            print(f"[SESSION CONFIG ERROR] {err}")

    def _on_toggle_mode(self, view):
        self.cv_mode = (self.cv_mode + 1) % len(self.mode_labels)
        self.btn_mode.setText(self.mode_icons[self.cv_mode])
        self.txt_mode.setText(self.mode_labels[self.cv_mode])

        self.hw_fps_tracker.reset()
        self.cv_fps_tracker.reset()
        self.txt_fps.setText("FPS: --")
        self.nanodet_frame_counter = 0
        self.nanodet_last_detections = []

    # ── NANODET INFERENCE & DFL POST-PROCESSING ──────────────────────────────
    def _run_nanodet_detection(self):
        if self.nanodet_net is None:
            msg = self.nanodet_error_msg or "Model not loaded"
            detail = self.nanodet_error_detail or ""
            cv2.putText(self.arr, msg, (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60, 255), 2)
            if detail:
                cv2.putText(self.arr, detail[:38], (15, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 180, 255), 1)
                if len(detail) > 38:
                    cv2.putText(self.arr, detail[38:76], (15, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 180, 255), 1)
            return

        self.nanodet_frame_counter += 1
        should_run = (self.nanodet_frame_counter % NANODET_DETECT_INTERVAL == 0) \
            or not self.nanodet_last_detections

        if should_run:
            try:
                self.nanodet_last_detections = self._nanodet_forward_pass()
            except Exception as err:
                print(f"[NanoDet DETECTION ERROR] {err}")
                traceback.print_exc()

        self._draw_cached_detections()

    def _nanodet_forward_pass(self):
        """
        Preprocesses frame -> runs NanoDet DNN -> decodes DFL bounding boxes and scores -> runs NMS.
        Returns: [(bx, by, bw, bh, cls_id, score), ...]
        """
        img_h, img_w = self.arr.shape[:2]
        in_size = NANODET_INPUT_SIZE

        # Letterbox maintain aspect ratio
        r = min(in_size / img_h, in_size / img_w)
        new_w, new_h = int(img_w * r), int(img_h * r)

        bgr = cv2.cvtColor(self.arr, cv2.COLOR_RGBA2BGR)
        resized_img = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad canvas and apply BGR mean/std normalization required by NanoDet
        padded_img = np.zeros((in_size, in_size, 3), dtype=np.float32)
        norm_sub = (resized_img.astype(np.float32) - NANODET_MEAN) / NANODET_STD
        padded_img[:new_h, :new_w] = norm_sub

        # Convert HWC to NCHW blob (swapRB=False since we already normalized in BGR)
        blob = cv2.dnn.blobFromImage(padded_img, scalefactor=1.0, size=(in_size, in_size), swapRB=False)
        self.nanodet_net.setInput(blob)

        out_names = self.nanodet_net.getUnconnectedOutLayersNames()
        outputs = self.nanodet_net.forward(out_names)

        # Parse the outputs into cls_preds and dis_preds
        # NanoDet exports produce 6 outputs (3 cls strides + 3 dis strides)
        cls_preds = {}
        dis_preds = {}
        for name, tensor in zip(out_names, outputs):
            for s in NANODET_STRIDES:
                if f"stride_{s}" in name or f"_{s}" in name:
                    if "cls" in name:
                        cls_preds[s] = tensor[0] if tensor.ndim == 3 else tensor
                    elif "dis" in name:
                        dis_preds[s] = tensor[0] if tensor.ndim == 3 else tensor

        # If layer names are numeric/unlabeled, sort by spatial dimensions
        if len(cls_preds) < 3 or len(dis_preds) < 3:
            cls_list, dis_list = [], []
            for t in outputs:
                t_sq = t[0] if t.ndim == 3 else t
                if t_sq.shape[-1] == 80:  # 80 classes
                    cls_list.append(t_sq)
                elif t_sq.shape[-1] == (NANODET_REG_MAX + 1) * 4:  # 32 regression values
                    dis_list.append(t_sq)

            # Sort descending by point count (2704 -> stride 8, 676 -> stride 16, 169 -> stride 32)
            cls_list.sort(key=lambda x: x.shape[0], reverse=True)
            dis_list.sort(key=lambda x: x.shape[0], reverse=True)

            for idx, s in enumerate(NANODET_STRIDES):
                if idx < len(cls_list):
                    cls_preds[s] = cls_list[idx]
                if idx < len(dis_list):
                    dis_preds[s] = dis_list[idx]

        candidate_boxes = []
        candidate_scores = []
        candidate_cls = []

        # Decode each stride
        for stride in NANODET_STRIDES:
            if stride not in cls_preds or stride not in dis_preds:
                continue

            cls_out = cls_preds[stride]  # (N, 80)
            dis_out = dis_preds[stride]  # (N, 32)
            grid = self._nanodet_grids[stride]  # (N, 2)

            # Apply sigmoid to raw logits if necessary
            if np.min(cls_out) < 0.0 or np.max(cls_out) > 1.0:
                scores_all = 1.0 / (1.0 + np.exp(-np.clip(cls_out, -15.0, 15.0)))
            else:
                scores_all = cls_out

            max_scores = np.max(scores_all, axis=1)
            cls_ids = np.argmax(scores_all, axis=1)

            pos_mask = max_scores >= NANODET_SCORE_THRESHOLD
            if not np.any(pos_mask):
                continue

            pos_dis = dis_out[pos_mask]
            pos_grid = grid[pos_mask]
            pos_scores = max_scores[pos_mask]
            pos_cls_ids = cls_ids[pos_mask]

            # Decode Distribution Focal Loss (DFL): 4 coordinates with (REG_MAX + 1) bins
            n_pos = pos_dis.shape[0]
            dis_reshaped = pos_dis.reshape(n_pos, 4, NANODET_REG_MAX + 1)

            # Numerical stable Softmax over the last dimension
            exp_d = np.exp(dis_reshaped - np.max(dis_reshaped, axis=-1, keepdims=True))
            softmax_d = exp_d / np.sum(exp_d, axis=-1, keepdims=True)
            dist = np.sum(softmax_d * self._project, axis=-1) * stride  # (n_pos, 4) -> l, t, r, b

            # Calculate box coordinates on input canvas
            x1 = (pos_grid[:, 0] * stride - dist[:, 0]) / r
            y1 = (pos_grid[:, 1] * stride - dist[:, 1]) / r
            x2 = (pos_grid[:, 0] * stride + dist[:, 2]) / r
            y2 = (pos_grid[:, 1] * stride + dist[:, 3]) / r

            # Clip within original image frame
            x1 = np.clip(x1, 0, img_w - 1)
            y1 = np.clip(y1, 0, img_h - 1)
            x2 = np.clip(x2, 0, img_w - 1)
            y2 = np.clip(y2, 0, img_h - 1)

            w = np.maximum(1.0, x2 - x1)
            h = np.maximum(1.0, y2 - y1)

            for i in range(len(pos_scores)):
                candidate_boxes.append([int(x1[i]), int(y1[i]), int(w[i]), int(h[i])])
                candidate_scores.append(float(pos_scores[i]))
                candidate_cls.append(int(pos_cls_ids[i]))

        if not candidate_boxes:
            return []

        # Multi-class Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            candidate_boxes,
            candidate_scores,
            NANODET_SCORE_THRESHOLD,
            NANODET_NMS_THRESHOLD
        )

        if indices is None or len(indices) == 0:
            return []

        results = []
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = candidate_boxes[idx]
            results.append((bx, by, bw, bh, candidate_cls[idx], candidate_scores[idx]))
        return results

    def _draw_cached_detections(self):
        palette = [
            (0, 229, 255, 255), (0, 255, 163, 255), (255, 170, 0, 255),
            (255, 82, 82, 255), (179, 136, 255, 255),
        ]
        for bx, by, bw, bh, cls_id, score in self.nanodet_last_detections:
            color = palette[cls_id % len(palette)]
            name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else 'object'
            label = f"{name} {int(score * 100)}%"

            cv2.rectangle(self.arr, (bx, by), (bx + bw, by + bh), color, 2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            tag_y = max(by, th + 8)
            cv2.rectangle(self.arr, (bx, tag_y - th - 6), (bx + tw + 6, tag_y + baseline - 2), (20, 20, 25, 220), -1)
            cv2.putText(self.arr, label, (bx + 3, tag_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        if not self.nanodet_last_detections and self.nanodet_net is not None:
            cv2.putText(self.arr, "Searching for objects...", (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200, 255), 1)

    # ── COMPUTER VISION WORKER LOOP ──────────────────────────────────────────
    def _cv_worker_loop(self):
        orb = cv2.ORB_create(nfeatures=200) if OPENCV_AVAILABLE else None

        while self.running:
            self.frame_event.wait()
            self.frame_event.clear()

            if not self.running:
                break
            if self.is_switching or self.cv_mode == 0 or not OPENCV_AVAILABLE:
                continue

            self.worker_busy = True
            try:
                if not hasattr(self, "in_bmp"):
                    vw = max(self.texture_view.getWidth(), 360)
                    vh = max(self.texture_view.getHeight(), 640)

                    proc_w = 360
                    proc_h = int(360 * (vh / vw))
                    proc_w = proc_w if proc_w % 2 == 0 else proc_w + 1
                    proc_h = proc_h if proc_h % 2 == 0 else proc_h + 1

                    frame = self.texture_view.getBitmap(proc_w, proc_h)
                    if frame is None:
                        continue

                    byte_count = proc_w * proc_h * 4
                    with self.bmp_lock:
                        self.in_bmp = frame
                        self.out_bmp = frame.copy(frame.getConfig(), True)
                        self.bb_wrapper = stratum.allocate_direct_buffer(byte_count)
                        raw_mv = stratum._stratum.bytebuffer_to_memoryview(self.bb_wrapper._ptr)
                        self.arr = np.frombuffer(raw_mv, dtype=np.uint8).reshape((proc_h, proc_w, 4))
                        self.gray_buf = np.empty((proc_h, proc_w), dtype=np.uint8)
                else:
                    if self.texture_view.getBitmap(self.in_bmp) is None:
                        continue

                self.bb_wrapper.rewind()
                self.in_bmp.copyPixelsToBuffer(self.bb_wrapper)

                if self.facing_front:
                    cv2.flip(self.arr, 1, dst=self.arr)

                if self.cv_mode == 1:
                    cv2.cvtColor(self.arr, cv2.COLOR_RGBA2GRAY, dst=self.gray_buf)
                    kps = orb.detect(self.gray_buf, None)
                    cv2.drawKeypoints(
                        self.arr, kps, self.arr,
                        color=(0, 229, 255, 255),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )

                elif self.cv_mode == 2:
                    cv2.cvtColor(self.arr, cv2.COLOR_RGBA2GRAY, dst=self.gray_buf)
                    edges = cv2.Canny(self.gray_buf, 60, 150)
                    self.arr[:, :, 0] = edges
                    self.arr[:, :, 1] = edges
                    self.arr[:, :, 2] = edges
                    self.arr[:, :, 3] = 255

                elif self.cv_mode == 3:
                    cv2.cvtColor(self.arr, cv2.COLOR_RGBA2GRAY, dst=self.gray_buf)
                    blurred = cv2.GaussianBlur(self.gray_buf, (5, 5), 0)
                    edges = cv2.Canny(blurred, 40, 140)
                    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if len(approx) == 4 and cv2.contourArea(c) > 2000:
                            cv2.polylines(self.arr, [approx], True, (0, 255, 163, 255), 4)
                            break

                elif self.cv_mode == 4:
                    cv2.cvtColor(self.arr, cv2.COLOR_RGBA2GRAY, dst=self.gray_buf)
                    _, thresh = cv2.threshold(self.gray_buf, 115, 255, cv2.THRESH_BINARY)
                    self.arr[:, :, 0] = thresh
                    self.arr[:, :, 1] = thresh
                    self.arr[:, :, 2] = thresh
                    self.arr[:, :, 3] = 255

                elif self.cv_mode == 5:
                    self._run_nanodet_detection()

                with self.bmp_lock:
                    self.bb_wrapper.rewind()
                    self.out_bmp.copyPixelsFromBuffer(self.bb_wrapper)

                cv_fps = self.cv_fps_tracker.tick()

                if self.handler:
                    def update_ui(fps=cv_fps):
                        if self.cv_mode != 0 and hasattr(self, "out_bmp"):
                            self.image_view.setImageBitmap(self.out_bmp)
                            if not self.first_frame_rendered:
                                self._dismiss_boot_cover()
                            now = time.perf_counter()
                            if fps is not None and (now - self.last_hud_update >= 0.25):
                                self.last_hud_update = now
                                self.txt_fps.setText(f"FPS: {round(fps)}")

                    self.handler.post(update_ui)

            except Exception as err:
                print(f"[WORKER ERROR] {err}")
                self._clear_buffers()
            finally:
                self.worker_busy = False

    def shutdown(self):
        self.running = False
        self.frame_event.set()
        self._shutdown_camera()


# ── STRATUM LIFECYCLE HOOKS ──────────────────────────────────────────────────
app = None

def onCreate():
    global app
    activity = stratum.getActivity()
    try:
        app = CameraApp(activity)
    except Exception:
        err_msg = f"Stratum Init Error:\n{traceback.format_exc()}"
        print(f"[FATAL] {err_msg}")
        try:
            tv = TextView(activity)
            tv.setText(err_msg)
            tv.setTextColor(0xFFFF5252)
            tv.setBackgroundColor(0xFF1E1E1E)
            tv.setPadding(40, 60, 40, 60)
            stratum.setContentView(activity, tv)
        except Exception:
            pass


def onResume():
    global app
    if app and app.camera_device is None:
        st_raw = app.texture_view.getSurfaceTexture()
        if st_raw:
            if app.cam_mgr is None:
                app._on_surface_available(st_raw, app.texture_view.getWidth(), app.texture_view.getHeight())
            else:
                app._open_camera()


def onPause():
    pass


def onStop():
    pass


def onDestroy():
    global app
    if app:
        app.shutdown()
        app = None