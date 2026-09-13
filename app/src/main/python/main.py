"""
Stratum Showcase Application: Aesthetic Real-time Computer Vision Camera
========================================================================
Clean architecture:
- Visual styles, dimensions, and shape drawables are defined in Android XML.
- Python drives the Camera2 hardware pipeline, native OpenCV processing,
  asynchronous image persistence, and touch interaction via Stratum.
"""

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
#hi
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

# Disable verbose C++ bridge logging for maximum frame throughput
stratum.set_log_enabled(True)


def _cast(obj, cls):
    """
    Safely casts a native JNI global reference (_ptr) into a typed Stratum wrapper.
    Stratum verifies inheritance via JNI env->IsInstanceOf().
    """
    if obj is None:
        return None
    return cls.from_ptr(obj)


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

    def get_fps(self, now: float | None = None) -> float | None:
        if now is None:
            now = time.perf_counter()
        with self._lock:
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

        # Camera Geometry & Configuration
        self.camera_ids = []
        self.current_cam_idx = 0
        self.sensor_orient = 90
        self.facing_front = False

        self.zoom_level = 1.0
        self.max_zoom = 1.0
        self.active_array_size = None

        self.preview_width = 1280
        self.preview_height = 720

        # Computer Vision Modes
        self.cv_mode = 0
        self.mode_labels = [
            "Normal (Raw)",
            "ORB Features",
            "White Edges",
            "Doc Scanner",
            "B&W Threshold"
        ]
        self.mode_icons = [
            "⚪ Normal",
            "⚡ ORB",
            "◈ Edges",
            "⌗ Scan",
            "◐ B&W"
        ]

        # Pipeline Synchronization
        self.running = True
        self.is_switching = False
        self.worker_busy = False
        self.first_frame_rendered = False
        self.frame_event = threading.Event()

        # Robust Sliding Window FPS Trackers
        self.hw_fps_tracker = SlidingFpsTracker(window_size=25)
        self.cv_fps_tracker = SlidingFpsTracker(window_size=15)
        self.last_hud_update = 0.0

        # Orientation Sensor Throttling
        self.last_sensor_time = 0.0
        self.current_ui_rotation = 0.0

        # Background MediaStore Save Queue & Buffer Locks
        self.save_queue = queue.Queue(maxsize=8)
        self.bmp_lock = threading.Lock()

        # ── 1. Inflate Layout & Bind UI Elements ─────────────────────────────
        self._inflate_and_bind_ui()

        # ── 2. Attach Orientation Accelerometer ──────────────────────────────
        self._init_orientation_sensor()

        # ── 3. Start Processing and Disk Persistence Workers ─────────────────
        self.processing_thread = threading.Thread(target=self._cv_worker_loop, daemon=True)
        self.processing_thread.start()

        self.saver_thread = threading.Thread(target=self._save_worker_loop, daemon=True)
        self.saver_thread.start()

    # ── UI INFLATION & VIEW BINDING ───────────────────────────────────────────
    def _res_id(self, name: str, res_type: str = "id") -> int:
        """
        Dynamically queries Android's resource table by string identifier.
        Equivalent to R.id.<name> or R.layout.<name> in Java.
        """
        res_id = self.res.getIdentifier(name, res_type, self.pkg)
        if res_id == 0:
            raise KeyError(f"Resource '{name}' of type '{res_type}' not found in package '{self.pkg}'")
        return res_id

    def _bind(self, name: str, cls):
        """Finds a view by its XML ID and casts it to its typed Stratum wrapper."""
        return _cast(self.root.findViewById(self._res_id(name, "id")), cls)

    def _inflate_and_bind_ui(self):
        """
        Inflates activity_camera.xml. All view properties, paddings, rounded
        drawables, and colors are declared in XML. Python only binds references.
        """
        inflater = self.activity.getLayoutInflater()
        layout_id = self._res_id("activity_camera", "layout")
        self.root = _cast(inflater.inflate(layout_id, None, False), FrameLayout)
        stratum.setContentView(self.activity, self.root)

        # 1. Viewfinder Layers
        self.texture_view = self._bind("texture_view", TextureView)
        self.image_view = self._bind("image_view", ImageView)
        self.boot_cover = self._bind("boot_cover", FrameLayout)
        self.flash_overlay = self._bind("flash_overlay", View)
        self.focus_ring = self._bind("focus_ring", View)

        # Intercept touch on viewfinder for camera AF/AE targeting
        self.texture_view.setOnTouchListener(self._on_viewfinder_touch)

        # Attach hardware surface listener
        self.texture_view.setSurfaceTextureListener({
            "onSurfaceTextureAvailable":   self._on_surface_available,
            "onSurfaceTextureSizeChanged": self._on_surface_size_changed,
            "onSurfaceTextureDestroyed":   self._on_surface_destroyed,
            "onSurfaceTextureUpdated":     self._on_surface_updated,
        })

        # 2. Top HUD Elements
        self.hud_badge = self._bind("hud_badge", LinearLayout)
        self.txt_fps = self._bind("txt_fps", TextView)
        self.txt_mode = self._bind("txt_mode", TextView)
        self.save_toast = self._bind("save_toast", TextView)

        # 3. Bottom Control Dock
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

        # Views that rotate gracefully with device orientation changes
        self.rotating_views = [
            self.btn_zoom_out,
            self.btn_mode,
            self.btn_capture,
            self.btn_switch,
            self.btn_zoom_in,
            self.hud_badge,
            self.save_toast,
        ]

    # ── TOUCH TO AUTOFOCUS & EXPOSURE ─────────────────────────────────────────
    def _on_viewfinder_touch(self, view, motion_event) -> bool:
        raw_event = _cast(motion_event, MotionEvent)
        if raw_event.getAction() != 0:  # MotionEvent.ACTION_DOWN
            return False

        touch_x = raw_event.getX()
        touch_y = raw_event.getY()

        self._animate_focus_ring(touch_x, touch_y)
        self._trigger_touch_focus(touch_x, touch_y)
        return True

    def _animate_focus_ring(self, x: float, y: float):
        """Displays and animates the reticle indicator centered at the tap position."""
        if not self.handler:
            return
        ring_half = 36.0  # 72dp / 2
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
            self.handler.postDelayed(lambda: pulse(step + 1), 20)

        self.handler.post(pulse)

    def _trigger_touch_focus(self, touch_x: float, touch_y: float):
            """Translates view coordinates to Camera2 sensor active array coordinates."""
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

                # Account for portrait sensor rotation
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

                # ── 1. Create real Java native MeteringRectangle[] array ───────────
                if not hasattr(self, "_metering_rect_class"):
                    self._metering_rect_class = Class.forName("android.hardware.camera2.params.MeteringRectangle")

                metering_array = Array.newInstance(self._metering_rect_class, 1)
                Array.set(metering_array, 0, metering_rect)

                # ── 2. Cancel previous AF cycle ────────────────────────────────────
                self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 2)  # CONTROL_AF_TRIGGER_CANCEL
                self.capture_session.capture(self.builder.build(), None, self.handler)

                # ── 3. Apply new AF & AE metering regions ──────────────────────────
                self.builder.set(CaptureRequest.sf_get_CONTROL_AF_REGIONS(), metering_array)
                self.builder.set(CaptureRequest.sf_get_CONTROL_AE_REGIONS(), metering_array)
                self.builder.set(CaptureRequest.sf_get_CONTROL_AF_MODE(), 1)     # CONTROL_AF_MODE_AUTO
                self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 0)  # CONTROL_AF_TRIGGER_IDLE

                # Resume repeating preview with the new region
                self.capture_session.setRepeatingRequest(self.builder.build(), None, self.handler)

                # ── 4. Trigger one-shot AF capture lock ────────────────────────────
                self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 1)  # CONTROL_AF_TRIGGER_START
                self.capture_session.capture(self.builder.build(), None, self.handler)
                self.builder.set(CaptureRequest.sf_get_CONTROL_AF_TRIGGER(), 0)  # Return to IDLE

            except Exception as err:
                print(f"[TOUCH FOCUS ERROR] {err}")

    # ── HARDWARE SENSORS & DYNAMIC ROTATION ───────────────────────────────────
    def _init_orientation_sensor(self):
        try:
            sensor_svc = self.activity.getSystemService("sensor")
            self.sensor_mgr = _cast(sensor_svc, SensorManager)
            if self.sensor_mgr:
                accel = self.sensor_mgr.getDefaultSensor(1)  # Sensor.TYPE_ACCELEROMETER
                if accel:
                    self.sensor_mgr.registerListener({
                        "onSensorChanged": self._on_sensor_changed,
                        "onAccuracyChanged": lambda s, a: None,
                    }, accel, 3)  # SensorManager.SENSOR_DELAY_NORMAL
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
                    self.handler.post(lambda: self._apply_ui_rotation(target_deg))
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
                self.boot_cover.setVisibility(8)  # View.GONE
                return
            self.boot_cover.setAlpha(max(0.0, 1.0 - (step * 0.22)))
            self.handler.postDelayed(lambda: fade(step + 1), 25)

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

        snapshot = None
        with self.bmp_lock:
            if self.cv_mode == 0:
                # Capture directly from the hardware TextureView
                snapshot = self.texture_view.getBitmap(self.preview_width, self.preview_height)
            else:
                # Capture processed CV image
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
        """Asynchronously encodes snapshots to JPEG and writes to MediaStore."""
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
        """Centers and scales the landscape camera stream onto a portrait display."""
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

        # Wait briefly for CV worker to cycle out
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
            """
            Triggered by Android's RenderThread each time a new hardware camera
            frame is available in the TextureView.
            """
            if self.is_switching:
                return

            # 1. Update hardware preview FPS via sliding window
            current_hw_fps = self.hw_fps_tracker.tick()

            # 2. Mode 0: Raw GPU rendering (Direct viewfinder)
            if self.cv_mode == 0:
                if self.image_view.getVisibility() != 8:
                    self.image_view.setVisibility(8)  # View.GONE
                if not self.first_frame_rendered:
                    self._dismiss_boot_cover()


                # Throttle HUD text updates to 4 times per second to prevent UI layout churn
                now = time.perf_counter()
                if current_hw_fps is not None and (now - self.last_hud_update >= 0.25):
                    self.last_hud_update = now
                    self.txt_fps.setText(f"FPS: {round(current_hw_fps)}")
                return

            # 3. Mode 1-4: Computer Vision Pipeline Active
            if self.image_view.getVisibility() != 0:
                self.image_view.setVisibility(0)  # View.VISIBLE

            # Mark busy immediately BEFORE signalling to eliminate race condition
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
            self.builder.set(CaptureRequest.sf_get_CONTROL_AF_MODE(), 4)  # CONTINUOUS_PICTURE

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

        # Reset trackers on switch
        self.hw_fps_tracker.reset()
        self.cv_fps_tracker.reset()
        self.txt_fps.setText("FPS: --")

    # ── COMPUTER VISION WORKER LOOP ──────────────────────────────────────────
    def _cv_worker_loop(self):
        """
        Background processing loop: fetches frame buffer via zero-copy native
        ByteBuffer, runs OpenCV kernels, and posts bitmap directly to UI.
        """
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
                # 1. Allocate / reuse direct frame buffers
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

                # 2. Zero-Copy Sync: Copy texture pixels to native memory
                self.bb_wrapper.rewind()
                self.in_bmp.copyPixelsToBuffer(self.bb_wrapper)

                if self.facing_front:
                    cv2.flip(self.arr, 1, dst=self.arr)

                # 3. Execute selected CV filter
                cv2.cvtColor(self.arr, cv2.COLOR_RGBA2GRAY, dst=self.gray_buf)

                if self.cv_mode == 1:
                    # ORB Keypoint tracking
                    kps = orb.detect(self.gray_buf, None)
                    cv2.drawKeypoints(
                        self.arr, kps, self.arr,
                        color=(0, 229, 255, 255),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )
                elif self.cv_mode == 2:
                    # Canny edge detection
                    edges = cv2.Canny(self.gray_buf, 60, 150)
                    self.arr[:, :, 0] = edges
                    self.arr[:, :, 1] = edges
                    self.arr[:, :, 2] = edges
                    self.arr[:, :, 3] = 255

                elif self.cv_mode == 3:
                    # Document scanner contour quad
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
                    # B&W threshold
                    _, thresh = cv2.threshold(self.gray_buf, 115, 255, cv2.THRESH_BINARY)
                    self.arr[:, :, 0] = thresh
                    self.arr[:, :, 1] = thresh
                    self.arr[:, :, 2] = thresh
                    self.arr[:, :, 3] = 255

# 4. Flush processed direct buffer back to Android Bitmap
                with self.bmp_lock:
                    self.bb_wrapper.rewind()
                    self.out_bmp.copyPixelsFromBuffer(self.bb_wrapper)

                # 5. Measure actual CV completion throughput
                cv_fps = self.cv_fps_tracker.tick()

                # 6. Post bitmap and HUD text directly to the UI thread
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