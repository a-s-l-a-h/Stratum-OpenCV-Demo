"""
Stratum Showcase Application: Aesthetic Real-time Computer Vision Camera
========================================================================
Demonstrates:
  - XML Layout Inflation & Safe Downcasting via Stratum v9
  - Camera2 Hardware Pipeline with Dynamic Digital Zoom & Facing Flip
  - Direct ByteBuffers for Zero-Copy OpenCV Image Processing
  - Hardware Sensor Integration (Gyroscope/Accelerometer UI Rotation)
  - Async MediaStore Photo Storage
"""

import sys
import time
import queue
import threading
import traceback
import stratum

# ── Stratum Android View & Graphics Wrappers ──────────────────────────────────
from stratum.android.view.View import View
from stratum.android.view.TextureView import TextureView
from stratum.android.view.Surface import Surface
from stratum.android.widget.FrameLayout import FrameLayout
from stratum.android.widget.LinearLayout import LinearLayout
from stratum.android.widget.Button import Button
from stratum.android.widget.TextView import TextView
from stratum.android.widget.ImageView import ImageView
from stratum.android.widget.ImageView_ScaleType import ImageView_ScaleType
from stratum.android.graphics.Matrix import Matrix
from stratum.android.graphics.Rect import Rect
from stratum.android.graphics.SurfaceTexture import SurfaceTexture
from stratum.android.graphics.drawable.GradientDrawable import GradientDrawable
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
from stratum.android.util.Size import Size

# ── Hardware Sensor Wrappers ─────────────────────────────────────────────────
from stratum.android.hardware.SensorManager import SensorManager
from stratum.android.hardware.Sensor import Sensor

# ── Storage & MediaStore Wrappers ────────────────────────────────────────────
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

# Disable verbose native logging for production performance
stratum.set_log_enabled(False)


def _cast(obj, cls):
    """Safely cast a native JNI reference into a typed Stratum wrapper."""
    if obj is None:
        return None
    return cls.from_ptr(obj)


def make_glass_drawable(bg_color_argb: int, corner_radius: float, stroke_color_argb: int = 0, stroke_width: int = 0):
    """Generates an Android GradientDrawable matching glassmorphism design."""
    drawable = GradientDrawable()
    drawable.setShape(0)  # RECTANGLE
    drawable.setColor(bg_color_argb)
    drawable.setCornerRadius(corner_radius)
    if stroke_width > 0 and stroke_color_argb != 0:
        drawable.setStroke(stroke_width, stroke_color_argb)
    return drawable


class AestheticCameraApp:
    def __init__(self, activity):
        self.activity = activity
        self.res = activity.getResources()
        self.pkg = activity.getPackageName()

        # Camera2 Hardware Handles
        self.cam_mgr = None
        self.camera_device = None
        self.capture_session = None
        self.builder = None
        self.handler = None

        # Camera Geometry & Config
        self.camera_ids = []
        self.current_cam_idx = 0
        self.sensor_orient = 90
        self.facing_front = False

        self.zoom_level = 1.0
        self.max_zoom = 1.0
        self.active_array_size = None

        self.preview_width = 1280
        self.preview_height = 720

        # Vision Modes
        self.cv_mode = 0
        self.mode_labels = ["ORB Features", "White Edges", "Doc Scanner", "B&W Threshold"]
        self.mode_icons = ["⚡ ORB", "◈ Edges", "⌗ Scan", "◐ B&W"]

        # Threading & Synchronization
        self.running = True
        self.is_switching = False
        self.worker_ready = True
        self.first_frame_rendered = False
        self.frame_event = threading.Event()
        self.last_fps_time = time.time()
        self.last_sensor_time = 0.0

        self.current_ui_rotation = 0.0
        self.save_queue = queue.Queue(maxsize=8)
        self.bmp_lock = threading.Lock()

        # ── 1. Inflate Layout & Apply Styling ────────────────────────────────
        self._inflate_and_style_ui()

        # ── 2. Hardware Sensors ──────────────────────────────────────────────
        self._init_orientation_sensor()

        # ── 3. Start Background Processing & Saver Loops ─────────────────────
        self.processing_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.processing_thread.start()

        self.saver_thread = threading.Thread(target=self._save_worker_loop, daemon=True)
        self.saver_thread.start()

    # ── UI INFLATION & STYLING ───────────────────────────────────────────────
    def _res_id(self, name: str, res_type: str = "id") -> int:
        res_id = self.res.getIdentifier(name, res_type, self.pkg)
        if res_id == 0:
            raise KeyError(f"Android Resource '{name}' ({res_type}) not found in package '{self.pkg}'")
        return res_id

    def _bind(self, name: str, cls):
        return _cast(self.root.findViewById(self._res_id(name, "id")), cls)

    def _inflate_and_style_ui(self):
        inflater = self.activity.getLayoutInflater()
        layout_id = self._res_id("activity_camera", "layout")
        self.root = _cast(inflater.inflate(layout_id, None, False), FrameLayout)
        stratum.setContentView(self.activity, self.root)

        # Bind Core Viewfinder Elements
        self.texture_view = self._bind("texture_view", TextureView)
        self.image_view = self._bind("image_view", ImageView)
        self.boot_cover = self._bind("boot_cover", FrameLayout)
        self.flash_overlay = self._bind("flash_overlay", View)

        fit_xy = ImageView_ScaleType.sf_get_FIT_XY()
        self.image_view.setScaleType(fit_xy)

        # Attach Viewfinder Lifecycle
        self.texture_view.setSurfaceTextureListener({
            "onSurfaceTextureAvailable":   self._on_surface_available,
            "onSurfaceTextureSizeChanged": self._on_surface_size_changed,
            "onSurfaceTextureDestroyed":   self._on_surface_destroyed,
            "onSurfaceTextureUpdated":     self._on_surface_updated,
        })

        # Bind HUD Status Elements
        self.hud_badge = self._bind("hud_badge", LinearLayout)
        self.dot_live = self._bind("dot_live", TextView)
        self.txt_fps = self._bind("txt_fps", TextView)
        self.txt_mode = self._bind("txt_mode", TextView)
        self.save_toast = self._bind("save_toast", TextView)

        self.hud_badge.setBackground(make_glass_drawable(0xB30A0D14, 40.0, 0x3300E5FF, 1))
        self.save_toast.setBackground(make_glass_drawable(0xD90D111A, 30.0, 0x6600FFA3, 1))

        # Bind Dock & Action Buttons
        self.dock = self._bind("dock", LinearLayout)
        self.dock.setBackground(make_glass_drawable(0xB80E121B, 50.0, 0x26FFFFFF, 1))

        btn_glass = make_glass_drawable(0x2EFFFFFF, 35.0)

        self.btn_zoom_out = self._bind("btn_zoom_out", Button)
        self.btn_zoom_out.setBackground(btn_glass)
        self.btn_zoom_out.setOnClickListener(self._on_zoom_out)

        self.btn_mode = self._bind("btn_mode", Button)
        self.btn_mode.setBackground(make_glass_drawable(0x2600E5FF, 35.0, 0x8000E5FF, 1))
        self.btn_mode.setOnClickListener(self._on_toggle_mode)

        # Authentic Circular Shutter Styling (Outer Glass Ring + Solid White Core)
        self.shutter_container = self._bind("shutter_container", FrameLayout)
        self.shutter_container.setBackground(make_glass_drawable(0x33FFFFFF, 70.0, 0x80FFFFFF, 2))

        self.btn_capture = self._bind("btn_capture", Button)
        self.btn_capture.setBackground(make_glass_drawable(0xFFFFFFFF, 60.0))
        self.btn_capture.setOnClickListener(self._on_capture)

        self.btn_switch = self._bind("btn_switch", Button)
        self.btn_switch.setBackground(btn_glass)
        self.btn_switch.setOnClickListener(self._on_switch)

        self.btn_zoom_in = self._bind("btn_zoom_in", Button)
        self.btn_zoom_in.setBackground(btn_glass)
        self.btn_zoom_in.setOnClickListener(self._on_zoom_in)

        # Views to dynamically re-orient with the accelerometer
        self.rotating_views = [
            self.btn_zoom_out,
            self.btn_mode,
            self.btn_capture,
            self.btn_switch,
            self.btn_zoom_in,
            self.hud_badge,
            self.save_toast,
        ]

        self._start_live_indicator_pulse()

    # ── HARDWARE SENSORS & DYNAMIC ROTATION ───────────────────────────────────
    def _init_orientation_sensor(self):
        try:
            sensor_svc = self.activity.getSystemService("sensor")
            self.sensor_mgr = _cast(sensor_svc, SensorManager)
            if self.sensor_mgr:
                accel = self.sensor_mgr.getDefaultSensor(1)  # TYPE_ACCELEROMETER
                if accel:
                    self.sensor_mgr.registerListener({
                        "onSensorChanged": self._on_sensor_changed,
                        "onAccuracyChanged": lambda s, a: None,
                    }, accel, 3)  # SENSOR_DELAY_NORMAL: Prevents GIL saturation
        except Exception as err:
            print(f"[SENSOR INIT ERROR] {err}")

    def _on_sensor_changed(self, event):
        # Throttle sensor events to maximum 8 updates/sec to prevent UI/CV stutters
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
                if x > 3.0:
                    target_deg = 90.0   # Landscape (Turned Left)
                elif x < -3.0:
                    target_deg = 270.0  # Reverse Landscape (Turned Right)
            else:
                if y < -3.0:
                    target_deg = 180.0  # Inverted Portrait
                else:
                    target_deg = 0.0    # Standard Portrait

            if target_deg != self.current_ui_rotation:
                self.current_ui_rotation = target_deg
                if self.handler:
                    self.handler.post(lambda: self._apply_ui_rotation(target_deg))
        except Exception:
            pass

    def _apply_ui_rotation(self, deg: float):
        for v in self.rotating_views:
            try:
                v.setRotation(deg)
            except Exception:
                pass

    # ── ANIMATION & TRANSITIONS ──────────────────────────────────────────────
    def _dismiss_boot_cover(self):
        """Fades out the boot cover once the first CV frame is active."""
        if not self.handler or self.first_frame_rendered:
            return
        self.first_frame_rendered = True

        def fade(step=0):
            if step >= 6:
                self.boot_cover.setVisibility(8)  # View.GONE
                return
            alpha = max(0.0, 1.0 - (step * 0.18))
            self.boot_cover.setAlpha(alpha)
            self.handler.postDelayed(lambda: fade(step + 1), 25)

        self.handler.post(fade)

    def _start_live_indicator_pulse(self):
        looper = Looper.getMainLooper()
        self.handler = self.handler or Handler(looper)

        def pulse(on=True):
            if not self.running:
                return
            self.dot_live.setAlpha(1.0 if on else 0.3)
            self.handler.postDelayed(lambda: pulse(not on), 600)

        pulse()

    def _trigger_shutter_flash(self):
        if not self.handler:
            return
        self.flash_overlay.setAlpha(0.7)

        def fade_flash(step=0):
            if step >= 4:
                self.flash_overlay.setAlpha(0.0)
                return
            self.flash_overlay.setAlpha(max(0.0, 0.7 - (step * 0.2)))
            self.handler.postDelayed(lambda: fade_flash(step + 1), 25)

        self.handler.post(fade_flash)

    def _show_save_toast(self):
        if not self.handler:
            return
        self.save_toast.setAlpha(1.0)
        self.handler.postDelayed(lambda: self.save_toast.setAlpha(0.0), 1300)

    # ── CAPTURE & ASYNC STORAGE ──────────────────────────────────────────────
    def _on_capture(self, view):
        self._trigger_shutter_flash()

        # Capture a thread-safe snapshot of the output bitmap
        snapshot = None
        with self.bmp_lock:
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

    # ── CAMERA2 PIPELINE ─────────────────────────────────────────────────────
    def _configure_transform(self, view_w, view_h):
        if view_w <= 0 or view_h <= 0 or self.preview_width <= 0:
            return
        try:
            matrix = Matrix()
            pw, ph = float(self.preview_width), float(self.preview_height)
            cx, cy = view_w / 2.0, view_h / 2.0

            # Scale to completely fill the screen without letterboxing
            scale = max(view_w / ph, view_h / pw)
            matrix.setScale((ph / view_w) * scale, (pw / view_h) * scale, cx, cy)

            if self.facing_front:
                matrix.postRotate(270.0, cx, cy)
            else:
                matrix.postRotate(90.0, cx, cy)

            self.texture_view.setTransform(matrix)
        except Exception as err:
            print(f"[TRANSFORM ERROR] {err}")

    def _clear_buffers(self):
        self.worker_ready = False
        with self.bmp_lock:
            for attr in ("in_bmp", "out_bmp", "bb_wrapper", "arr", "gray_buf"):
                if hasattr(self, attr):
                    delattr(self, attr)

    def _apply_zoom(self):
        if not self.capture_session or not self.builder:
            return
        try:
            if self.active_array_size:
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
        except Exception as err:
            print(f"[ZOOM ERROR] {err}")

        try:
            req = self.builder.build()
            self.capture_session.setRepeatingRequest(req, None, self.handler)
        except Exception as err:
            print(f"[CAM REQ ERROR] {err}")

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
        while not self.worker_ready and (time.time() - wait_start) < 1.0:
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

            vw = max(self.texture_view.getWidth(), 1)
            vh = max(self.texture_view.getHeight(), 1)
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
            print(f"[SURFACE ERROR] {err}")

    def _on_surface_size_changed(self, st, w, h):
        self.is_switching = True
        self._clear_buffers()
        if w > 0 and h > 0:
            self._configure_transform(w, h)
        self.is_switching = False
        self.worker_ready = True

    def _on_surface_destroyed(self, st):
        self.shutdown()
        return True

    def _on_surface_updated(self, st):
        if self.is_switching or not OPENCV_AVAILABLE:
            return
        if self.worker_ready:
            # Display the processed frame on the UI thread without holding locks
            if hasattr(self, "out_bmp"):
                self.image_view.setImageBitmap(self.out_bmp)
                if not self.first_frame_rendered:
                    self._dismiss_boot_cover()

            self.worker_ready = False
            self.frame_event.set()

    def _on_camera_opened(self, raw_device):
        try:
            self.camera_device = _cast(raw_device, CameraDevice)
            st_raw = self.texture_view.getSurfaceTexture()
            st = _cast(st_raw, SurfaceTexture)
            self.surface = Surface(st)

            self.builder = self.camera_device.createCaptureRequest(1)
            self.builder.addTarget(self.surface)

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
            self.worker_ready = True
        except Exception as err:
            print(f"[SESSION CONFIG ERROR] {err}")

    def _on_toggle_mode(self, view):
        self.cv_mode = (self.cv_mode + 1) % len(self.mode_labels)
        self.btn_mode.setText(self.mode_icons[self.cv_mode])
        self.txt_mode.setText(self.mode_labels[self.cv_mode])

    # ── COMPUTER VISION WORKER LOOP ──────────────────────────────────────────
    def _worker_loop(self):
        orb = cv2.ORB_create(nfeatures=250)
        smoothed_fps = 30.0

        while self.running:
            self.frame_event.wait()
            self.frame_event.clear()

            if not self.running:
                break
            if self.is_switching:
                self.worker_ready = True
                continue

            try:
                # 1. Initialize Direct Buffers and Numpy Views Once
                if not hasattr(self, "in_bmp"):
                    view_w = max(self.texture_view.getWidth(), 360)
                    view_h = max(self.texture_view.getHeight(), 360)

                    PROC_W = 480
                    PROC_H = int(480 * (view_h / view_w))
                    PROC_W = PROC_W if PROC_W % 2 == 0 else PROC_W + 1
                    PROC_H = PROC_H if PROC_H % 2 == 0 else PROC_H + 1

                    frame = self.texture_view.getBitmap(PROC_W, PROC_H)
                    if frame is None:
                        self.worker_ready = True
                        continue

                    size = PROC_W * PROC_H * 4
                    with self.bmp_lock:
                        self.in_bmp = frame
                        self.out_bmp = frame.copy(frame.getConfig(), True)
                        self.bb_wrapper = stratum.allocate_direct_buffer(size)
                        raw_mv = stratum._stratum.bytebuffer_to_memoryview(self.bb_wrapper._ptr)
                        self.arr = np.frombuffer(raw_mv, dtype=np.uint8).reshape((PROC_H, PROC_W, 4))
                        self.gray_buf = np.empty((PROC_H, PROC_W), dtype=np.uint8)
                else:
                    try:
                        if self.texture_view.getBitmap(self.in_bmp) is None:
                            self.worker_ready = True
                            continue
                    except Exception:
                        self._clear_buffers()
                        self.worker_ready = True
                        continue

                # 2. Copy Texture to Direct ByteBuffer
                self.bb_wrapper.rewind()
                self.in_bmp.copyPixelsToBuffer(self.bb_wrapper)

                if self.facing_front:
                    cv2.flip(self.arr, 1, dst=self.arr)

                # 3. Apply Active OpenCV Filter
                cv2.cvtColor(self.arr, cv2.COLOR_RGBA2GRAY, dst=self.gray_buf)

                if self.cv_mode == 0:
                    # ORB Features (Cyan Keypoints)
                    keypoints = orb.detect(self.gray_buf, None)
                    cv2.drawKeypoints(
                        self.arr, keypoints, self.arr,
                        color=(0, 229, 255, 255),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )
                elif self.cv_mode == 1:
                    # Clean Pure White Edges
                    edges = cv2.Canny(self.gray_buf, 60, 150)
                    self.arr[:, :, 0] = edges
                    self.arr[:, :, 1] = edges
                    self.arr[:, :, 2] = edges
                    self.arr[:, :, 3] = 255

                elif self.cv_mode == 2:
                    # Document Edge Quad Finder
                    blurred = cv2.GaussianBlur(self.gray_buf, (5, 5), 0)
                    edges = cv2.Canny(blurred, 40, 140)
                    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    for cur_c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                        peri = cv2.arcLength(cur_c, True)
                        approx = cv2.approxPolyDP(cur_c, 0.02 * peri, True)
                        if len(approx) == 4 and cv2.contourArea(cur_c) > 3000:
                            cv2.polylines(self.arr, [approx], True, (0, 255, 163, 255), 4)
                            break

                elif self.cv_mode == 3:
                    # Black & White Thresholding
                    _, thresh = cv2.threshold(self.gray_buf, 115, 255, cv2.THRESH_BINARY)
                    self.arr[:, :, 0] = thresh
                    self.arr[:, :, 1] = thresh
                    self.arr[:, :, 2] = thresh
                    self.arr[:, :, 3] = 255

                # 4. FPS Monitoring
                t0 = time.time()
                instant_fps = 1.0 / max(t0 - self.last_fps_time, 1e-6)
                self.last_fps_time = t0
                smoothed_fps = 0.90 * smoothed_fps + 0.10 * instant_fps

                if self.handler:
                    fps_str = f"FPS: {smoothed_fps:.1f}"
                    self.handler.post(lambda: self.txt_fps.setText(fps_str))

                # 5. Flush Direct Buffer into Output Display Bitmap
                with self.bmp_lock:
                    if hasattr(self, "bb_wrapper") and hasattr(self, "out_bmp"):
                        self.bb_wrapper.rewind()
                        self.out_bmp.copyPixelsFromBuffer(self.bb_wrapper)

            except Exception as err:
                print(f"[WORKER ERROR] {err}")
                self._clear_buffers()
            finally:
                self.worker_ready = True

    def shutdown(self):
        self.running = False
        self.frame_event.set()
        self._shutdown_camera()


# ── Stratum Application Lifecycle Hooks ──────────────────────────────────────
app = None


def onCreate():
    global app
    activity = stratum.getActivity()
    try:
        app = AestheticCameraApp(activity)
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