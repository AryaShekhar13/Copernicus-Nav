#!/usr/bin/env python3
"""Sample true pose during a command. Usage: spin_watch.py LIN ANG SECONDS"""
import math, re, subprocess, sys, time

lin = sys.argv[1] if len(sys.argv) > 1 else "0.0"
ang = sys.argv[2] if len(sys.argv) > 2 else "0.5"
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout

def num(b, k):
    m = re.search(rf'\b{k}:\s*([-+\d.eE]+)', b)
    return float(m.group(1)) if m else 0.0

def truth():
    out = sh('gz topic -e -t /world/flat_test/dynamic_pose/info -n 1')
    m = re.search(r'name: "ugv".*?position \{(.*?)\}.*?orientation \{(.*?)\}', out, re.S)
    if not m:
        return None
    p, q = m.group(1), m.group(2)
    qx, qy, qz, qw = (num(q, k) for k in "xyzw")
    yaw = math.degrees(math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    roll = math.degrees(math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2*(qw*qy - qz*qx)))))
    return num(p, 'x'), num(p, 'y'), num(p, 'z'), roll, pitch, yaw

def cmd(l, a):
    sh(f'gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {{x: {l}}}, angular: {{z: {a}}}"')

def show(label, t0):
    s = truth()
    if s is None:
        print(f"{time.time()-t0:5.1f}s {label:6s} could not parse pose")
        return
    print(f"{time.time()-t0:5.1f}s {label:6s} x={s[0]:7.2f} y={s[1]:7.2f} z={s[2]:5.2f} roll={s[3]:6.1f} pitch={s[4]:6.1f} yaw={s[5]:7.1f}")

cmd(0, 0)
time.sleep(2)
t0 = time.time()
show("rest", t0)
cmd(lin, ang)
print(f"--- sent linear={lin} angular={ang} ---")
end = time.time() + dur
while time.time() < end:
    show("moving", t0)
cmd(0, 0)
print("--- sent stop ---")
for _ in range(3):
    show("after", t0)
