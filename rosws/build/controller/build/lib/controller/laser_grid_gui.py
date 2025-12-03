"""
laser_grid_gui.py — now with a Return to Zero button

Features:
✔ Start Scan
✔ Stop Scan
✔ Return to Zero (A=0, B=0)
✔ Sequential axis movement (A then B)
✔ Automatic homing after grid scan
"""

import math
import time
import threading
import serial
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ------------------- Geometry & limits -------------------

H = 0.25
X_MAX_USER = 0.50
MAX_ANGLE_DEG = 45

MAX_ANGLE = math.radians(MAX_ANGLE_DEG)
X_GEOM_MAX = H * math.tan(MAX_ANGLE)

# ------------------- Y calibration -------------------

Y_MIN_M = -0.35
Y_MAX_M =  0.35

# ------------------- Serial settings ----------------------

SERIAL_PORT = "COM6"     # <<< CHANGE THIS
BAUD_RATE   = 9600
TIMEOUT_S   = 2.0

MOVE_DELAY_S = 1.5

# ------------------- Geometry -------------------

def angle_from_vertical_for_x(x, h=H):
    if x <= 0:
        return 0.0, False
    alpha_ideal = math.atan(x / h)
    exceeds = alpha_ideal > MAX_ANGLEa
    alpha_cmd = min(alpha_ideal, MAX_ANGLE)
    return alpha_cmd, exceeds

def plan_command(x_target, y_target, h=H):
    alpha_cmd, exceeded = angle_from_vertical_for_x(x_target, h)
    y_cmd = y_target
    return y_cmd, alpha_cmd, exceeded

# ------------------- Mapping -------------------

def alpha_to_A(alpha_cmd):
    if alpha_cmd <= 0:
        return 0
    frac = alpha_cmd / MAX_ANGLE
    return int(round(max(0, min(100, 100 * frac))))

def y_to_B(y_cmd):
    if y_cmd <= Y_MIN_M:
        return 0
    if y_cmd >= Y_MAX_M:
        return 100
    frac = (y_cmd - Y_MIN_M) / (Y_MAX_M - Y_MIN_M)
    return int(round(max(0, min(100, 100 * frac))))

# ------------------- Grid ----------------------

def generate_grid(x_min, x_max, y_min, y_max, nx, ny, serpentine=True):
    xs = [x_min + i * (x_max - x_min) / (nx - 1) for i in range(nx)]
    ys = [y_min + j * (y_max - y_min) / (ny - 1) for j in range(ny)]

    pts = []
    for i, x in enumerate(xs):
        row = ys[::-1] if (serpentine and i % 2 == 1) else ys
        for y in row:
            pts.append((x, y))
    return pts

# ------------------- Serial ----------------------

def send_AB_command(ser, A_val, B_val, log):
    cmd = f"A{A_val} B{B_val}\n"
    log(f"--> {cmd.strip()}")
    ser.write(cmd.encode("ascii"))
    ser.flush()

# ------------------- Return to Zero (worker) ----------------------

def return_to_zero(stop_event, log):
    """Return to A=0, B=0 sequentially."""
    try:
        log("Opening serial port for Return-to-Zero...")
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_S) as ser:
            time.sleep(2)

            log("Moving A → 0")
            send_AB_command(ser, 0, 0, log)

            t = time.time()
            while time.time() - t < MOVE_DELAY_S:
                if stop_event.is_set():
                    log("Return-to-zero canceled.")
                    return
                time.sleep(0.05)

            log("Moving B → 0")
            send_AB_command(ser, 0, 0, log)

            t = time.time()
            while time.time() - t < MOVE_DELAY_S:
                if stop_event.is_set():
                    log("Return-to-zero canceled.")
                    return
                time.sleep(0.05)

            log("Return-to-zero complete.")

    except Exception as e:
        log(f"Error during return-to-zero: {e}")

# ------------------- Grid scan (worker) ----------------------

def run_grid_scan(stop_event, log):
    try:
        log(f"h={H:.3f} m | x_max geom={X_GEOM_MAX:.3f} m")
        if X_MAX_USER > X_GEOM_MAX:
            log("Warning: requested x_max > geometric achievable max.")

        x_min = 0.05
        x_max = min(X_MAX_USER, X_GEOM_MAX)
        nx = 4
        ny = 4

        targets = generate_grid(x_min, x_max, Y_MIN_M, Y_MAX_M, nx, ny, serpentine=True)
        log(f"Generated {len(targets)} grid points.")

        log(f"Opening serial port {SERIAL_PORT}...")
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_S) as ser:
            time.sleep(2)

            last_A = 0
            last_B = 0

            # -------- Main scanning loop --------
            for idx, (xt, yt) in enumerate(targets):
                if stop_event.is_set():
                    log("Scan stopped.")
                    return

                log(f"\nPoint {idx+1}/{len(targets)} → x={xt:.3f}, y={yt:.3f}")

                y_cmd, alpha_cmd, exceeded = plan_command(xt, yt, H)
                A_val = alpha_to_A(alpha_cmd)
                B_val = y_to_B(y_cmd)

                # -------- Step 1: Move A --------
                log(f"Move A → {A_val}")
                send_AB_command(ser, A_val, last_B, log)
                last_A = A_val

                t = time.time()
                while time.time() - t < MOVE_DELAY_S:
                    if stop_event.is_set():
                        log("Stopped during A move.")
                        return
                    time.sleep(0.05)

                # -------- Step 2: Move B --------
                log(f"Move B → {B_val}")
                send_AB_command(ser, last_A, B_val, log)
                last_B = B_val

                t = time.time()
                while time.time() - t < MOVE_DELAY_S:
                    if stop_event.is_set():
                        log("Stopped during B move.")
                        return
                    time.sleep(0.05)

            # -------- Return home --------
            log("\nReturning to home (A=0, B=0)...")

            log("A → 0")
            send_AB_command(ser, 0, last_B, log)
            t = time.time()
            while time.time() - t < MOVE_DELAY_S:
                if stop_event.is_set(): return
                time.sleep(0.05)

            log("B → 0")
            send_AB_command(ser, 0, 0, log)
            t = time.time()
            while time.time() - t < MOVE_DELAY_S:
                if stop_event.is_set(): return
                time.sleep(0.05)

            log("Scan complete.")

    except Exception as e:
        log(f"Scan error: {e}")

# ------------------- GUI ----------------------

class LaserGridGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Laser Grid Controller")

        self.stop_event = threading.Event()
        self.scan_thread = None

        # --- Buttons ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(btn_frame, text="Start Scan", command=self.start_scan)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_scan, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.home_btn = tk.Button(btn_frame, text="Return to Zero", command=self.start_return_to_zero)
        self.home_btn.pack(side="left", padx=5)

        # --- Log window ---
        self.log_box = scrolledtext.ScrolledText(root, width=80, height=24, state="disabled")
        self.log_box.pack(padx=10, pady=10)

    # GUI-safe logging
    def log(self, msg):
        def write():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(0, write)

    # --- Start scan ---
    def start_scan(self):
        if self.scan_thread and self.scan_thread.is_alive():
            return

        self.stop_event.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.scan_thread = threading.Thread(
            target=run_grid_scan,
            args=(self.stop_event, self.log),
            daemon=True
        )
        self.scan_thread.start()

        self.root.after(200, self.check_thread)

    # --- Stop ---
    def stop_scan(self):
        self.log("Stop requested.")
        self.stop_event.set()

    # --- Return to Zero ---
    def start_return_to_zero(self):
        self.log("Return to Zero requested.")

        self.stop_event.clear()
        threading.Thread(
            target=return_to_zero,
            args=(self.stop_event, self.log),
            daemon=True
        ).start()

    # --- Thread watcher ---
    def check_thread(self):
        if self.scan_thread and self.scan_thread.is_alive():
            self.root.after(200, self.check_thread)
        else:
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.log("Scan thread finished.")

# ------------------- MAIN ----------------------

if __name__ == "__main__":
    root = tk.Tk()
    LaserGridGUI(root)
    root.mainloop()
