"""
Relays or republishes a camera feed as a plain MJPEG network stream, with
zero image processing on the frames — just pushes bytes through for maximum
FPS. Two modes depending on CAMERA_SOURCE:

  - Network URL (e.g. the Android "IP Webcam" app): pure byte relay, no
    decode/re-encode at all. Used for testing before the real USB camera is
    on hand.
  - USB camera index: captures raw frames and JPEG-encodes them, the one
    unavoidable step needed to put them on the network as MJPEG. Still no
    detection/measurement of any kind.

Main server pulls from this Pi exactly the same way either way:
  cv2.VideoCapture("http://<pi-ip>:8081/video")
  http://<pi-ip>:8081/shot.jpg

One-time Pi setup:
  sudo apt update
  sudo apt install -y python3-opencv v4l-utils
"""

import base64
import hmac
import multiprocessing as mp
import os
import platform
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn, TCPServer

IS_WINDOWS = platform.system() == "Windows"

DESKTOP_PROCESS_NAMES = [
    "gnome-shell", "plasmashell", "xfce4-session", "lxsession",
    "labwc", "sway", "mutter", "kwin_x11", "kwin_wayland",
]


def _ensure_display_env():
    # Linux-only: when launched from the desktop (double-clicked icon, or a
    # local terminal), DISPLAY/WAYLAND_DISPLAY are already set correctly and
    # nothing needs to change. This only kicks in when launched over SSH,
    # which starts a bare shell with no GUI context at all — it reads the
    # real values straight out of an already-running desktop process's own
    # environment, which works regardless of X11 vs Wayland or how that
    # session's display number was assigned. Not applicable on Windows,
    # which has no such concept.
    if IS_WINDOWS:
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    try:
        pid = subprocess.check_output(
            ["pgrep", "-u", str(os.getuid()), "-x", "|".join(DESKTOP_PROCESS_NAMES)],
            text=True,
        ).split()[0]
        with open(f"/proc/{pid}/environ", "rb") as f:
            env_pairs = f.read().split(b"\0")
        found = {}
        for pair in env_pairs:
            if b"=" not in pair:
                continue
            key, _, value = pair.partition(b"=")
            key = key.decode()
            if key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR"):
                found[key] = value.decode()
        os.environ.update(found)
        if found.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"
    except Exception:
        pass


_ensure_display_env()

import cv2
import numpy as np
import requests

# Settings come from a plain text config file next to this script, so they
# can be changed (new phone IP, new network, switch to USB camera index,
# new stream login) without editing any code. Create/edit
# qc_camera_config.txt with lines like:
#   CAMERA_SOURCE=http://192.168.1.6:8080/video
#   STREAM_USER=qcadmin
#   STREAM_PASS=change-me
# or, for the real USB camera:
#   CAMERA_SOURCE=0
CONFIG_FILE = os.path.join(os.path.expanduser("~"), "qc_camera_config.txt")
DEFAULTS = {
    "CAMERA_SOURCE": "http://192.168.1.6:8080/video",
    "STREAM_USER": "qcadmin",
    "STREAM_PASS": "change-me",
}


def load_config():
    config = dict(DEFAULTS)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    return config


def save_config_value(key, value):
    # Rewrites qc_camera_config.txt with one key updated (or appended if
    # new), preserving every other line as-is. Always written as plain
    # UTF-8 with no BOM — PowerShell's Out-File -Encoding utf8 adds one,
    # which previously corrupted whichever key landed on the first line.
    lines = []
    found = False
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8-sig") as f:
            for line in f:
                stripped = line.strip()
                if "=" in stripped and not stripped.startswith("#"):
                    existing_key, _, _ = stripped.partition("=")
                    if existing_key.strip() == key:
                        lines.append(f"{key}={value}\n")
                        found = True
                        continue
                lines.append(line if line.endswith("\n") else line + "\n")
    if not found:
        lines.append(f"{key}={value}\n")
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)


_config = load_config()
CAMERA_SOURCE = _config["CAMERA_SOURCE"]
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)
STREAM_USER = _config["STREAM_USER"]
STREAM_PASS = _config["STREAM_PASS"]
_focus_config = _config.get("CAMERA_FOCUS", "").strip()
CAMERA_FOCUS = int(_focus_config) if _focus_config.lstrip("-").isdigit() else None
_exposure_config = _config.get("CAMERA_EXPOSURE", "").strip()
CAMERA_EXPOSURE = int(_exposure_config) if _exposure_config.lstrip("-").isdigit() else None
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
JPEG_QUALITY = 85
STREAM_FPS = 15
PORT = 8081

frame_lock = threading.Lock()
latest_frame = None
latest_raw_frame = None
stop_event = threading.Event()
# Updated once the camera is open, from its own reported FPS, so the
# network stream isn't throttled below what the camera can deliver.
stream_delay = 1 / STREAM_FPS


def lock_camera_settings(cap):
    # Focus and exposure are deliberately left out here — they're set
    # explicitly by camera_setup() instead of whatever value they
    # happened to converge on during warmup.
    current_value = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, current_value)


def open_usb_camera():
    backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2
    cap = cv2.VideoCapture(CAMERA_SOURCE, backend)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_SOURCE}.")
        return None

    # Without forcing MJPG first, many UVC cameras silently ignore the
    # requested resolution and fall back to a low default (e.g. 800x600).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    print(f"Camera opened at {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")

    for _ in range(60):
        cap.read()
    lock_camera_settings(cap)
    return cap


def reopen_usb_camera():
    """Used after the camera drops mid-stream (see usb_capture_loop). Same
    as open_usb_camera() but also reapplies the already-calibrated focus
    and exposure, so a reconnect doesn't fall back to a blurry/flickering
    default — no interactive calibration screen here, it just restores
    the last confirmed-good values."""
    cap = open_usb_camera()
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if CAMERA_FOCUS is not None:
        cap.set(cv2.CAP_PROP_FOCUS, CAMERA_FOCUS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    if CAMERA_EXPOSURE is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    return cap


def camera_setup(cap):
    """One-time manual focus/exposure calibration shown before streaming
    starts. Displays the exact same full-resolution frames the live feed
    uses — no downscaling or re-encoding — so what's judged here is the
    real captured quality.

    If you never touch exposure here, behavior is unchanged from before:
    autoexposure gets locked at whatever value it converged to during
    warmup, same as always. Exposure tuning is here as an extra, optional
    lever against flicker under AC-powered lights — a fixed exposure that
    lands on a clean multiple of your mains cycle (10ms steps for 50Hz,
    ~8.3ms for 60Hz) averages the light's brightness oscillation out
    instead of catching it mid-flicker. Press 'p' (Windows only) to also
    open the camera driver's own native properties dialog, which on some
    webcams (including Logitech's) has an explicit Anti-Flicker/Power Line
    Frequency setting that's a more direct fix than tuning exposure blind.
    """
    global CAMERA_FOCUS, CAMERA_EXPOSURE
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    focus_val = float(CAMERA_FOCUS) if CAMERA_FOCUS is not None else cap.get(cv2.CAP_PROP_FOCUS)
    cap.set(cv2.CAP_PROP_FOCUS, focus_val)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    exposure_val = float(CAMERA_EXPOSURE) if CAMERA_EXPOSURE is not None else cap.get(cv2.CAP_PROP_EXPOSURE)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

    window = "Camera Setup"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    focus_step = 5
    exposure_step = 1

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        cv2.putText(display, f"Focus: {focus_val:.0f}   Exposure: {exposure_val:.0f}",
                    (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.putText(display, "+/- focus    4/5 exposure (helps with light flicker)",
                    (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        hint = "p: open camera driver settings (look for Anti-Flicker)    " if IS_WINDOWS else ""
        cv2.putText(display, f"{hint}ENTER confirm and start streaming",
                    (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):
            break
        elif key in (ord("+"), ord("=")):
            focus_val = min(255, focus_val + focus_step)
            cap.set(cv2.CAP_PROP_FOCUS, focus_val)
        elif key in (ord("-"), ord("_")):
            focus_val = max(0, focus_val - focus_step)
            cap.set(cv2.CAP_PROP_FOCUS, focus_val)
        elif key == ord("5"):
            exposure_val += exposure_step
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
        elif key == ord("4"):
            exposure_val -= exposure_step
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
        elif key == ord("p") and IS_WINDOWS:
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
        elif key == ord("q"):
            stop_event.set()
            break

    cv2.destroyWindow(window)
    if not stop_event.is_set():
        CAMERA_FOCUS = int(focus_val)
        CAMERA_EXPOSURE = int(exposure_val)
        save_config_value("CAMERA_FOCUS", CAMERA_FOCUS)
        save_config_value("CAMERA_EXPOSURE", CAMERA_EXPOSURE)
        print(f"Focus set to {focus_val:.0f}, exposure set to {exposure_val:.0f}, "
              f"saved to {CONFIG_FILE}")


def camera_worker_process(frame_queue, stop_evt):
    """Runs in a separate OS process, not a thread. When the camera drops
    (USB power event, driver reset, loose cable), cap.read() on DSHOW
    frequently just blocks forever instead of failing fast — and if that
    blocking native call doesn't release Python's GIL (a known rough edge
    here), NO other thread in that same process can run either, including
    a watchdog thread meant to notice and recover. A separate process has
    no such entanglement: the parent can kill it outright regardless of
    what it's stuck on, which a same-process thread fundamentally cannot
    guarantee. CAMERA_FOCUS/CAMERA_EXPOSURE are picked up fresh from the
    config file by reopen_usb_camera() since this module reloads on spawn.
    """
    cap = reopen_usb_camera()
    if cap is None:
        return

    # Some camera power/driver states (seen in practice: the activity LED
    # goes off, matching a sensor that's stopped actually streaming) keep
    # returning ret=True from read() with a stale or blank buffer forever
    # — no failure, no hang, so neither of the other recovery paths ever
    # fires. Real video, even of a perfectly still scene, always has tiny
    # frame-to-frame sensor noise, so identical encoded bytes repeating
    # many times in a row is a reliable sign the feed has actually frozen.
    # Exiting here lets the parent's existing "no new frame" watchdog
    # restart this process the same way it does for a hang.
    last_jpeg = None
    identical_count = 0
    FROZEN_FRAME_LIMIT = 15

    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
        jpeg_bytes = jpeg.tobytes()

        if jpeg_bytes == last_jpeg:
            identical_count += 1
            if identical_count >= FROZEN_FRAME_LIMIT:
                print("Camera feed appears frozen (identical frames repeating), "
                      "exiting worker so it gets restarted...")
                break
        else:
            identical_count = 0
        last_jpeg = jpeg_bytes

        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
        except Exception:
            pass
        try:
            frame_queue.put_nowait(jpeg_bytes)
        except Exception:
            pass
    cap.release()


def usb_capture_loop():
    global latest_frame, latest_raw_frame, stream_delay

    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=1)
    stop_evt = ctx.Event()

    def start_worker():
        p = ctx.Process(target=camera_worker_process, args=(frame_queue, stop_evt), daemon=True)
        p.start()
        return p

    proc = start_worker()

    # Measure the achieved rate from real frame arrivals instead of
    # trusting the unreliable driver-reported FPS (DSHOW often reports 0).
    warmup_deadline = time.perf_counter() + 3.0
    received = 0
    t0 = time.perf_counter()
    while time.perf_counter() < warmup_deadline and not stop_event.is_set():
        try:
            frame_queue.get(timeout=0.5)
            received += 1
        except queue.Empty:
            continue
    elapsed = time.perf_counter() - t0
    if received > 1 and elapsed > 0:
        measured_fps = received / elapsed
        stream_delay = 1 / measured_fps
        print(f"Measured camera rate: {measured_fps:.1f} FPS, streaming at that rate.")
    else:
        print(f"Could not measure camera rate, falling back to {STREAM_FPS} FPS.")

    # If no frame has arrived from the worker in this long, assume it's
    # stuck (hung read, crashed, camera dropped) and kill + restart the
    # whole process — a hard OS-level kill works no matter what it's
    # wedged on, unlike anything relying on cooperative shutdown.
    WATCHDOG_TIMEOUT = 5.0
    last_frame_time = time.monotonic()

    while not stop_event.is_set():
        try:
            jpeg_bytes = frame_queue.get(timeout=1.0)
            last_frame_time = time.monotonic()
            raw = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            with frame_lock:
                latest_frame = jpeg_bytes
                latest_raw_frame = raw
        except queue.Empty:
            pass

        if time.monotonic() - last_frame_time > WATCHDOG_TIMEOUT:
            print("Camera worker process appears stuck, killing and restarting it...")
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)
            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
            except Exception:
                pass
            proc = start_worker()
            last_frame_time = time.monotonic()
            print("Camera worker process restarted.")

    stop_evt.set()
    proc.terminate()
    proc.join(timeout=3)


def check_auth(handler):
    expected = "Basic " + base64.b64encode(
        f"{STREAM_USER}:{STREAM_PASS}".encode()).decode()
    provided = handler.headers.get("Authorization", "")
    if hmac.compare_digest(provided, expected):
        return True
    handler.send_response(401)
    handler.send_header("WWW-Authenticate", 'Basic realm="QC Camera"')
    handler.end_headers()
    return False


class UsbStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not check_auth(self):
            return

        if self.path == "/shot.jpg":
            with frame_lock:
                frame = latest_frame
            if frame is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            return

        if self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(stream_delay)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


class RelayHandler(BaseHTTPRequestHandler):
    """Pure passthrough: bytes come straight from the phone, untouched."""

    def do_GET(self):
        if not check_auth(self):
            return

        if self.path == "/video":
            upstream = requests.get(CAMERA_SOURCE, stream=True, timeout=5)
            self.send_response(upstream.status_code)
            self.send_header("Content-Type", upstream.headers.get(
                "Content-Type", "multipart/x-mixed-replace"))
            self.end_headers()
            try:
                for chunk in upstream.iter_content(chunk_size=4096):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                upstream.close()
            return

        if self.path == "/shot.jpg":
            # This source has no snapshot endpoint, so grab a single frame
            # from the video stream on demand instead (not used for the
            # continuous feed, so it doesn't affect streaming FPS).
            grab_cap = cv2.VideoCapture(CAMERA_SOURCE)
            ret, frame = grab_cap.read()
            grab_cap.release()
            if not ret:
                self.send_response(503)
                self.end_headers()
                return
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                self.send_response(500)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg.tobytes())))
            self.end_headers()
            self.wfile.write(jpeg.tobytes())
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


class ThreadingHTTPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def display_loop(is_network_source):
    # Only for showing the feed on the Pi's own screen — a separate concern
    # from what gets served to the main server, which stays untouched.
    disp_cap = cv2.VideoCapture(CAMERA_SOURCE) if is_network_source else None

    window = "QC Camera Feed"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Some window managers (GNOME/Mutter on X11) ignore a fullscreen request
    # made before the window is actually mapped on screen — applying it
    # right after the first real imshow() is more reliable than doing it
    # up front (works fine on the Pi's labwc either way).
    state = {"fullscreen": True, "fullscreen_applied": False,
             "button_rect": (0, 0, 0, 0), "close_rect": (0, 0, 0, 0)}

    def toggle_fullscreen():
        state["fullscreen"] = not state["fullscreen"]
        cv2.setWindowProperty(
            window, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if state["fullscreen"] else cv2.WINDOW_NORMAL,
        )

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bx1, by1, bx2, by2 = state["button_rect"]
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            toggle_fullscreen()
            return
        cx1, cy1, cx2, cy2 = state["close_rect"]
        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            stop_event.set()

    cv2.setMouseCallback(window, on_mouse)

    fail_count = 0
    RECONNECT_AFTER = 30  # ~3s of failed reads before retrying the connection

    while not stop_event.is_set():
        frame = None
        if is_network_source:
            ret, frame = disp_cap.read()
            if not ret:
                frame = None
                fail_count += 1
                if fail_count >= RECONNECT_AFTER:
                    disp_cap.release()
                    disp_cap = cv2.VideoCapture(CAMERA_SOURCE)
                    fail_count = 0
            else:
                fail_count = 0
        else:
            with frame_lock:
                frame = None if latest_raw_frame is None else latest_raw_frame.copy()

        # Always show *something* and keep pumping cv2.waitKey below — even
        # with no camera signal, the window must stay responsive to clicks
        # and 'q'/close, not freeze or sit invisible.
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for camera...", (40, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            cv2.putText(frame, f"Source: {CAMERA_SOURCE}", (40, 375),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "LIVE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "q: quit   f: toggle full screen", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        label = "Exit full screen" if state["fullscreen"] else "Full screen"
        bw, bh = 230, 40
        bx2 = frame.shape[1] - 20
        bx1 = bx2 - bw
        by1 = 20
        by2 = by1 + bh
        state["button_rect"] = (bx1, by1, bx2, by2)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        cv2.putText(frame, label, (bx1 + 12, by2 - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cw, ch = 50, 40
        cx1 = bx1 - 15 - cw
        cx2 = cx1 + cw
        cy1, cy2 = by1, by2
        state["close_rect"] = (cx1, cy1, cx2, cy2)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 0, 180), -1)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 1)
        cv2.putText(frame, "X", (cx1 + 16, cy2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF

        if not state["fullscreen_applied"]:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            state["fullscreen_applied"] = True

        if key == ord("q"):
            stop_event.set()
            break
        elif key == ord("f"):
            toggle_fullscreen()

    if state["fullscreen"]:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    if disp_cap is not None:
        disp_cap.release()
    cv2.destroyAllWindows()


def main():
    is_network_source = isinstance(CAMERA_SOURCE, str)
    print(f"Camera source: {CAMERA_SOURCE}  (from {CONFIG_FILE})")

    if is_network_source:
        print(f"Pure relay mode: forwarding bytes from {CAMERA_SOURCE}, "
              f"no decode/encode, no other processing.")
        server = ThreadingHTTPServer(("0.0.0.0", PORT), RelayHandler)
    else:
        cap = open_usb_camera()
        if cap is None:
            return
        camera_setup(cap)
        if stop_event.is_set():
            cap.release()
            cv2.destroyAllWindows()
            return
        # Calibration is done with this handle — the operational capture
        # loop opens its own fresh handle in a separate process, so this
        # one is released rather than left holding the device.
        cap.release()
        threading.Thread(target=usb_capture_loop, daemon=True).start()
        server = ThreadingHTTPServer(("0.0.0.0", PORT), UsbStreamHandler)

    print(f"Serving at http://0.0.0.0:{PORT}/video  (snapshot: /shot.jpg)")
    print(f"Authentication required — user: {STREAM_USER}  "
          f"(set in {CONFIG_FILE})")
    threading.Thread(target=server.serve_forever, daemon=True).start()

    display_loop(is_network_source)
    server.shutdown()


if __name__ == "__main__":
    mp.freeze_support()
    main()
