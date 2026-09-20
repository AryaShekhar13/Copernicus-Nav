#!/usr/bin/env python3
"""Spin-in-place slip test. Usage: spin_test.py [seconds] [rad_per_s]"""
import math, re, subprocess, sys, time

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout

def cmd(lin, ang):
    sh(f'gz topic -t /cmd_vel -m gz.msgs.Twist -p "linear: {{x: {lin}}}, angular: {{z: {ang}}}"')

def num(block, key):
    m = re.search(rf'\b{key}:\s*([-+\d.eE]+)', block)
    return float(m.group(1)) if m else 0.0

def yaw(orient):
    return math.degrees(2 * math.atan2(num(orient, 'z'), num(orient, 'w')))

def read():
    odom = sh('gz topic -e -t /odom -n 1')
    o = re.search(r'pose \{.*?orientation \{(.*?)\}', odom, re.S)
    truth = sh('gz topic -e -t /world/flat_test/dynamic_pose/info -n 1')
    t = re.search(r'name: "ugv".*?position \{(.*?)\}.*?orientation \{(.*?)\}', truth, re.S)
    if not o or not t:
        sys.exit("Could not parse gz output. Is the sim running?")
    return yaw(o.group(1)), yaw(t.group(2)), num(t.group(1), 'x'), num(t.group(1), 'y')

def wrap(d):
    return (d + 180) % 360 - 180

cmd(0, 0)
time.sleep(2)
o0, t0, x0, y0 = read()
cmd(0, W)
time.sleep(DUR)
cmd(0, 0)
time.sleep(3)
o1, t1, x1, y1 = read()

do, dt = wrap(o1 - o0), wrap(t1 - t0)
print(f"odom  delta yaw: {do:8.1f} deg")
print(f"truth delta yaw: {dt:8.1f} deg")
print(f"truth/odom ratio: {dt / do:.2f}" if abs(do) > 1 else "odom did not rotate")
print(f"truth position drift while spinning: {math.hypot(x1 - x0, y1 - y0):.2f} m")
