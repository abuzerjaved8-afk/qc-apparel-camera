"""
Garment size measurement on a Raspberry Pi, using a USB webcam (Logitech MX
Brio) at a fixed 3 ft (91.4 cm) lens-to-table distance. The Pi grabs frames
locally, measures the garment, and sends each measurement to the central
server over the company network (no separate laptop needed).

Controls (while the preview window is focused):
  c - calibrate using a reference object (e.g. A4 sheet) currently in frame
  s - print the current garment measurement and send it to the server
  l - re-lock camera settings (autofocus/exposure/white balance)
  q - quit

Calibration is saved to calibration.json and reused on the next run, as long
as the camera's focus/zoom/FOV mode hasn't changed.

One-time Pi setup:
  sudo apt update
  sudo apt install -y python3-opencv v4l-utils
  pip3 install requests
"""

import json
import os
import sys

# Required so cv2's GUI windows reach the Pi's Wayland desktop (labwc) when
# this script is launched over SSH, where these vars aren't inherited.
# Harmless no-ops when run from a local terminal on the Pi, where they're
# already set correctly.
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
os.environ.setdefault("QT_QPA_PLATFORM", "wayland")

import cv2
import numpy as np
import requests

CALIBRATION_FILE = "calibration.json"

# Set to an int (e.g. 0) for a local USB camera, or a URL string to pull an
# MJPEG stream instead (e.g. the Android "IP Webcam" app's /video endpoint).
# This is being used right now to test the pipeline with a phone standing in
# for the Logitech webcam, which isn't on hand yet.
CAMERA_SOURCE = "http://192.168.1.6:8080/video"
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
SERVER_URL = "http://SERVER_IP:PORT/garment-measurement"  # TODO: set to your backend's actual endpoint


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return None


def save_calibration(mm_per_px_x, mm_per_px_y):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"mm_per_px_x": mm_per_px_x, "mm_per_px_y": mm_per_px_y}, f)


def largest_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def rotated_box_dims(contour):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return w, h, box


def run_calibration(frame):
    contour = largest_contour(frame)
    if contour is None:
        print("No reference object detected — check lighting/background contrast.")
        return None

    w_px, h_px, box = rotated_box_dims(contour)
    preview = frame.copy()
    cv2.drawContours(preview, [box], 0, (0, 255, 0), 2)
    cv2.imshow("Calibration reference", preview)
    cv2.waitKey(1)

    try:
        known_w_mm = float(input(f"Detected reference box: {w_px:.1f}px x {h_px:.1f}px. "
                                  f"Enter its real WIDTH in mm: "))
        known_h_mm = float(input("Enter its real HEIGHT in mm: "))
    except ValueError:
        print("Invalid input, calibration cancelled.")
        cv2.destroyWindow("Calibration reference")
        return None

    cv2.destroyWindow("Calibration reference")
    mm_per_px_x = known_w_mm / w_px
    mm_per_px_y = known_h_mm / h_px
    save_calibration(mm_per_px_x, mm_per_px_y)
    print(f"Calibration saved: {mm_per_px_x:.4f} mm/px (x), {mm_per_px_y:.4f} mm/px (y)")
    return {"mm_per_px_x": mm_per_px_x, "mm_per_px_y": mm_per_px_y}


def lock_camera_settings(cap):
    locked = {}
    for name, auto_prop, auto_off_value, value_prop in [
        ("autofocus", cv2.CAP_PROP_AUTOFOCUS, 0, cv2.CAP_PROP_FOCUS),
        ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 0, cv2.CAP_PROP_EXPOSURE),
        ("auto_wb", cv2.CAP_PROP_AUTO_WB, 0, cv2.CAP_PROP_WB_TEMPERATURE),
    ]:
        current_value = cap.get(value_prop)
        auto_ok = cap.set(auto_prop, auto_off_value)
        value_ok = cap.set(value_prop, current_value)
        locked[name] = {"auto_disabled": bool(auto_ok), "value": current_value, "held": bool(value_ok)}
    return locked


def send_measurement(width_mm, height_mm):
    payload = {"width_cm": round(width_mm / 10, 2), "height_cm": round(height_mm / 10, 2)}
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=3)
        print(f"Sent to server: {payload} -> HTTP {resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to reach server ({SERVER_URL}): {e}")


def main():
    is_network_source = isinstance(CAMERA_SOURCE, str)

    if is_network_source:
        print(f"Opening network stream: {CAMERA_SOURCE}")
        cap = cv2.VideoCapture(CAMERA_SOURCE)
    else:
        # CAP_V4L2 gives more reliable property support on Linux/Raspberry
        # Pi than letting OpenCV auto-pick a backend.
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Could not open camera source {CAMERA_SOURCE}.")
        sys.exit(1)

    if is_network_source:
        print("Network stream source — skipping resolution/autofocus/exposure "
              "locking (those only apply to a local USB camera).")
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Stream resolution: {actual_w:.0f}x{actual_h:.0f}")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Capture resolution: {actual_w:.0f}x{actual_h:.0f}")

        print("Letting autofocus/auto-exposure settle for 2 seconds before locking...")
        for _ in range(60):
            cap.read()
        locked = lock_camera_settings(cap)
        for name, info in locked.items():
            status = "locked" if info["auto_disabled"] else "NOT supported by this camera/driver"
            print(f"  {name}: {status} (value held at {info['value']:.2f})")

    calibration = load_calibration()
    if calibration:
        print(f"Loaded calibration: {calibration['mm_per_px_x']:.4f} mm/px (x), "
              f"{calibration['mm_per_px_y']:.4f} mm/px (y)")
    else:
        print("No calibration found. Place a reference object (e.g. A4 sheet) "
              "on the table and press 'c' to calibrate.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break

        display = frame.copy()
        contour = largest_contour(frame)

        width_mm = height_mm = None
        if contour is not None and cv2.contourArea(contour) > 2000:
            w_px, h_px, box = rotated_box_dims(contour)
            cv2.drawContours(display, [box], 0, (0, 255, 0), 2)
            if calibration:
                width_mm = w_px * calibration["mm_per_px_x"]
                height_mm = h_px * calibration["mm_per_px_y"]
                label = f"{width_mm/10:.1f} x {height_mm/10:.1f} cm"
                cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

        cv2.putText(display, "c: calibrate  s: measure  l: relock camera  q: quit",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Garment measurement", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            new_calibration = run_calibration(frame)
            if new_calibration:
                calibration = new_calibration
        elif key == ord("s"):
            if width_mm is not None:
                print(f"Garment size: {width_mm/10:.1f} cm x {height_mm/10:.1f} cm")
                send_measurement(width_mm, height_mm)
            else:
                print("No measurement available — calibrate first and check contrast.")
        elif key == ord("l"):
            locked = lock_camera_settings(cap)
            for name, info in locked.items():
                status = "locked" if info["auto_disabled"] else "NOT supported"
                print(f"  {name}: {status} (value held at {info['value']:.2f})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
