"""Snake robot in MuJoCo with a live control panel.

    python run_sim.py                               # viewer + control panel
    python run_sim.py --headless -T 500             # no window: run the firmware over the track, print its log
    python run_sim.py --start ramp --headless -T 300   # start straight in front of the ramp
    python run_sim.py --controller sine --headless  # the manual sine gait instead
    python run_sim.py --hill 12 --amp 40            # any panel value can be set as a start value
    python run_sim.py --off slopeSteer,edgeSteer    # firmware switches off (see FW_FLAGS)

Track (build_model.py, sizes estimated from a photo of the real one): a walled
corridor with a 30 deg zigzag and a sharp 90 deg turn, then an open ramp without
side walls: up, down into a valley, up, a flat section, down.

Two controllers, chosen with the tabs in the panel (or --controller):

ESP32 firmware (default): firmware.py, a port of snake_robot/snake_robot.ino, ticking
    at 50 Hz against simulated sensors.  The head and tail accelerometers (BMI270:
    +-2 g, 16 bit) are MuJoCo accelerometers; the two Sharp IR sensors are rays cast
    from the sensor seats on the nose, turned into the sensor's output voltage and
    fed through the firmware's own sharpToCm().  In the panel the robot starts
    STOPPED like the real one: press Run, or type the Serial Monitor / Bluetooth commands
    (r, s, p, i, c, j N DEG, t N DEG, a DEG, f HZ, d DEG, and the switches v h e b k w) in the
    serial box.
    The .ino and the port contain additions to the version first pasted, marked
    [SIM] in the .ino; each has a switch (FW_FLAGS, a checkbox in the panel):
    the wave on the pitch joints, slope and edge steering for the open ramp, the
    body bend that follows the terrain, the original slope bends (off), and turning
    away at a wall without backing up first.

Manual sine: every joint follows the same sine law, with one shared frequency f,

    horizontal (yaw joints):    theta_k = A_h sin(2 pi f t + k dphi_h)          + O_h
    vertical   (pitch joints):  theta_k = A_v sin(2 pi f t + k dphi_v + phi_vh) + O_v

    k counts the joints of that plane from tail to head, dphi is the phase lag
    between neighbouring joints of the same plane, and phi_vh is the phase
    difference between the vertical and the horizontal wave.

    The default is a vertical wave only (A_v 30 deg, dphi_v -120 deg, 0.5 Hz), running
    from tail to head: a "caterpillar" gait, the same one the firmware tab uses.

Friction: the robot has no wheels or scales, so the default is isotropic: MuJoCo's
own friction with coefficient mu, the same in every direction.  A wave on the yaw
joints alone gives no reliable forward motion then; the vertical wave does.  For comparison,
--aniso (or the checkbox) switches to anisotropic friction, as with snake scales
or passive wheels: contacts with the course are made frictionless and the
friction force is applied here instead, per contact, as Coulomb friction mu_axial
along the segment and mu_lateral across it (in the surface's tangent plane, so
it also works on the ramps), smoothed with tanh around v0.  The ramp boards have
their own coefficient (rubber sheet, "mu on ramp").
"""
import argparse
import dataclasses
import pathlib
import time
import tkinter as tk
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np

import firmware as fw
from build_model import (IR_ANGLE_DEG, NOSE_LENGTH, RAMP_ANGLE_DEG, RAMP_FRICTION, RAMP_SECTIONS,
                         RAMP_WIDTH, course_poses, ramp_frame, ramp_profile, unit, yaw_quat)

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "snake.xml"


@dataclasses.dataclass
class Param:
    name: str
    label: str
    lo: float
    hi: float
    default: float
    unit: str
    step: float
    group: str
    tab: str            # "fw", "sine" or "world" (shared by both controllers)
    attr: str = ""      # firmware variable this value is written to


PARAMS = [
    # ---- ESP32 firmware: the variables and constants of the .ino
    Param("amp",          "Amplitude (a)",   0.0,  45.0, 45.0, "deg", 1.0,   "Gait", "fw", "gaitAmpDeg"),
    Param("fw_freq",      "Frequency (f)",   0.05,  2.0,  0.5, "Hz",  0.01,  "Gait", "fw", "gaitFreqHz"),
    Param("fw_phase",     "Phase lag (d)", 0.0, 180.0, 120.0, "deg", 1.0,  "Gait", "fw", "gaitPhaseDeg"),
    Param("steer_gain",   "Steer gain",      0.0,   5.0,  2.5, "°/cm", 0.1,  "Steering and slopes", "fw", "STEER_GAIN"),
    Param("slope_gain",   "Slope steer gain", 0.0,  5.0,  0.5, "°/°", 0.1,   "Steering and slopes", "fw", "SLOPE_STEER_GAIN"),
    Param("kink_trigger", "Kink trigger",    1.0,  20.0,  6.0, "deg", 0.5,   "Steering and slopes", "fw", "KINK_TRIGGER_DEG"),
    Param("valley_bend",  "Valley bend",     0.0,  60.0, 30.0, "deg", 1.0,   "Steering and slopes", "fw", "VALLEY_BEND_DEG"),
    Param("crest_bend",   "Crest bend",      0.0,  60.0, 20.0, "deg", 1.0,   "Steering and slopes", "fw", "CREST_BEND_DEG"),
    Param("crawl_speed",  "Crawl speed",     0.0,   0.1, 0.03, "m/s", 0.001, "Steering and slopes", "fw", "CRAWL_SPEED_MPS"),
    # ---- manual sine gait
    Param("freq",       "Frequency f",          0.0,  3.0,  0.5, "Hz",  0.01, "Wave (both planes)", "sine"),
    # default: vertical wave only, running tail -> head (negative lag).  With isotropic
    # friction this is what drives the robot forwards; a yaw wave alone does not.
    Param("h_amp",      "Amplitude A_h",        0.0, 90.0,  0.0, "deg", 1.0,  "Horizontal (yaw joints)", "sine"),
    Param("h_phase",    "Phase lag Δφ_h",    -180, 180, 120.0, "deg", 1.0,  "Horizontal (yaw joints)", "sine"),
    Param("h_offset",   "Offset O_h",         -90.0, 90.0,  0.0, "deg", 1.0,  "Horizontal (yaw joints)", "sine"),
    Param("v_amp",      "Amplitude A_v",        0.0, 90.0, 30.0, "deg", 1.0,  "Vertical (pitch joints)", "sine"),
    Param("v_phase",    "Phase lag Δφ_v",    -180, 180, -120.0, "deg", 1.0, "Vertical (pitch joints)", "sine"),
    Param("v_offset",   "Offset O_v",         -90.0, 90.0,  0.0, "deg", 1.0,  "Vertical (pitch joints)", "sine"),
    Param("vh_phase",   "Phase vert. vs hor. φ_vh", -180, 180, 90.0, "deg", 1.0, "Vertical (pitch joints)", "sine"),
    # ---- world
    Param("mu",         "μ (isotropic)",        0.0,  3.0,  0.5, "",    0.01, "Friction", "world"),
    Param("mu_axial",   "μ along body (aniso)", 0.0,  3.0,  0.1, "",    0.01, "Friction", "world"),
    Param("mu_lateral", "μ across body (aniso)", 0.0, 3.0,  1.0, "",    0.01, "Friction", "world"),
    Param("mu_ramp",    "μ on ramp (rubber)",   0.0,  3.0, RAMP_FRICTION, "", 0.01, "Friction", "world"),
    Param("hill",       "Ramp slope",           0.0, 30.0, RAMP_ANGLE_DEG, "deg", 0.5, "Course and sensors", "world"),
    Param("ir_angle",   "IR beam angle",        0.0, 90.0, IR_ANGLE_DEG, "deg", 1.0, "Course and sensors", "world"),
]
SPECS = {p.name: p for p in PARAMS}

# the firmware's on/off switches: variable -> (panel group, label, default)
FW_FLAGS = {
    "gaitVertical": ("Gait", "Wave on pitch joints (vertical; off = yaw wave as in the original .ino)", True),
    "slopeSteer":   ("Steering and slopes", "Slope steering: hold the fall line where there are no walls", True),
    "edgeSteer":    ("Steering and slopes", "Edge steering: keep away from the edge of the board", True),
    "bendFollow":   ("Steering and slopes", "Body bend follows the terrain (crests and valleys)", True),
    "slopeBends":   ("Steering and slopes", "Original slope bends (valley / crest kinks)", False),
    "wallTurn":     ("Steering and slopes", "At a wall: keep turning away (off = back up, then turn)", True),
}


class SineGait:
    """Same sine law for both planes; the phase 2 pi f t is integrated so that
    changing f while running does not make the joints jump."""

    def __init__(self, model, data, ramp=1.0):
        self.ramp = ramp
        self.phase = 0.0
        # classify actuators by the direction of their joint axis in the straight pose
        # (actuators are ordered tail -> head, so list order = k).  Modules mounted
        # rolled by 180/270 deg have their axis reversed; the sign makes a positive
        # angle bend every joint of a plane the same way.
        mujoco.mj_forward(model, data)
        yaw, pitch, sign = [], [], np.ones(model.nu)
        for a in range(model.nu):
            ax = data.xaxis[model.actuator_trnid[a, 0]]
            if abs(ax[2]) > 0.9:
                yaw.append(a)
                sign[a] = np.sign(ax[2])
            else:
                pitch.append(a)
                sign[a] = np.sign(ax[1])
        self.yaw, self.pitch, self.sign = np.array(yaw), np.array(pitch), sign

    def reset(self):
        self.phase = 0.0

    def __call__(self, p, t, dt, ctrl):
        self.phase += 2 * np.pi * p["freq"] * dt
        s = min(1.0, t / self.ramp) if self.ramp > 0 else 1.0     # ease in after a reset
        rad = np.radians
        k = np.arange(len(self.yaw))
        ctrl[self.yaw] = s * (rad(p["h_amp"]) * np.sin(self.phase + k * rad(p["h_phase"]))
                              + rad(p["h_offset"]))
        k = np.arange(len(self.pitch))
        ctrl[self.pitch] = s * (rad(p["v_amp"]) * np.sin(self.phase + k * rad(p["v_phase"])
                                                         + rad(p["vh_phase"]))
                                + rad(p["v_offset"]))
        ctrl *= self.sign


class FirmwareController:
    """Runs firmware.Firmware against the simulated robot: samples the sensors,
    calls tick() at LOOP_HZ and holds the servo targets in between, like PWM."""

    TICK = 1.0 / fw.Firmware.LOOP_HZ
    G = 9.81
    ACC_RANGE_G = 2.0           # BMI270 in +-2 g, 16 bit
    ACC_LSB_PER_G = 16384.0
    IR_DRAW_M = 0.30            # length of the drawn ray when nothing is in range

    def __init__(self, model, data):
        self.site = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
                     for n in ("imu_head", "ir_left", "ir_right")}
        self.head_body = model.site_bodyid[self.site["imu_head"]]
        self.acc_adr = {n: model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)]
                        for n in ("acc_head", "acc_tail")}

        # firmware joint i (head -> tail) drives actuator nu-1-i (actuators run tail -> head).
        # sign turns the firmware's convention (yaw + = bends left, pitch + = the part in
        # front tilts up) into the joint's own direction: the part in front of a joint is
        # its child body, so "left" is a positive turn about +z and "up" one about -y.
        if model.nu != 7:
            raise SystemExit("the firmware drives exactly 7 joints (P1 Y2 P3 Y4 P5 Y6 P7)")
        mujoco.mj_forward(model, data)
        self.act = np.arange(model.nu)[::-1]
        self.sign = np.empty(model.nu)
        for i, a in enumerate(self.act):
            ax = data.xaxis[model.actuator_trnid[a, 0]]
            is_yaw = abs(ax[2]) > 0.9
            if is_yaw != (i in fw.YAW_JOINTS):
                raise SystemExit(f"joint {fw.JOINT_NAME[i]} has the wrong type in the model; "
                                 "the firmware expects P1 Y2 P3 Y4 P5 Y6 P7 from head to tail")
            self.sign[i] = np.sign(ax[2]) if is_yaw else -np.sign(ax[1])
        self.reset()

    def reset(self):
        """Power cycle: a fresh firmware, STOPPED."""
        self.log = []
        self.fw = fw.Firmware()
        self.fw.headImu.ok = self.fw.tailImu.ok = True
        self.next_tick = self.TICK
        self.last_ms = 0
        self.rays = []
        self.acc_sum = None
        self.acc_count = 0
        self.collect(0.0)

    def collect(self, t):
        self.log += [(t, line) for line in self.fw.out]
        self.fw.out.clear()

    def command(self, line, t):
        self.fw.command(line)
        self.collect(t)

    # ---- simulated sensors
    def read_acc(self, name):
        """Acceleration averaged over the last tick, then clipped and quantised like the
        BMI270.  The average stands in for the sensor's limited bandwidth (100 Hz output
        rate): one instantaneous sample would alias the simulation's 1 kHz contact chatter."""
        adr = self.acc_adr[name]
        a = self.acc_sum[adr:adr + 3] / self.acc_count / self.G
        a = np.clip(a, -self.ACC_RANGE_G, self.ACC_RANGE_G - 1 / self.ACC_LSB_PER_G)
        return np.round(a * self.ACC_LSB_PER_G) / self.ACC_LSB_PER_G

    @staticmethod
    def sharp_millivolts(dist_m):
        """Output of a Sharp GP2Y0A41SK0F: the inverse of the firmware's curve fit,
        folding back towards 0 V below 4 cm as the real sensor does."""
        if dist_m < 0:
            return 0.0                                    # nothing hit
        cm = dist_m * 100.0
        volts = (max(cm, 4.0) / 12.08) ** (-1 / 1.058)
        return 1000.0 * volts * min(1.0, cm / 4.0)

    def read_ir(self, model, data, angle_deg, walls):
        frame = data.site_xmat[self.site["imu_head"]].reshape(3, 3)      # columns: forward, left, up
        a = np.radians(angle_deg)
        groups = np.array([1, 1, int(walls), 0, 0, 0], dtype=np.uint8)
        geomid = np.zeros(1, dtype=np.int32)
        self.rays, mv = [], []
        for name, side in (("ir_left", +1), ("ir_right", -1)):
            origin = data.site_xpos[self.site[name]].copy()
            direction = np.cos(a) * frame[:, 0] + side * np.sin(a) * frame[:, 1]
            dist = mujoco.mj_ray(model, data, origin, direction, groups, 1, self.head_body, geomid)
            mv.append(self.sharp_millivolts(dist))
            in_range = 0 <= dist <= self.fw.IR_MAX_CM / 100
            self.rays.append((origin, origin + direction * (dist if in_range else self.IR_DRAW_M), in_range))
        return mv

    def __call__(self, model, data, ctrl, angle_deg, walls):
        self.acc_sum = data.sensordata.copy() if self.acc_sum is None else self.acc_sum + data.sensordata
        self.acc_count += 1
        if data.time + 1e-9 < self.next_tick:
            return
        self.next_tick += self.TICK
        now_ms = int(round(data.time * 1000))
        dt, self.last_ms = (now_ms - self.last_ms) / 1000.0, now_ms
        left_mv, right_mv = self.read_ir(model, data, angle_deg, walls)
        servo = self.fw.tick(dt, now_ms, left_mv, right_mv,
                             self.read_acc("acc_head"), self.read_acc("acc_tail"))
        self.acc_sum, self.acc_count = None, 0
        ctrl[self.act] = np.radians(servo) * self.sign
        self.collect(data.time)


class GroundFriction:
    def __init__(self, model, v0=0.005):
        self.static = model.geom_bodyid == 0        # floor, course slabs, walls, obstacle
        self.v0 = v0
        self.anisotropic = None

    def configure(self, model, anisotropic, mu):
        """Anisotropic: frictionless contacts (condim 1, contact condim is the max over
        the pair) plus the forces below.  Isotropic: MuJoCo friction coefficient mu."""
        self.anisotropic = anisotropic
        model.geom_condim[:] = 1 if anisotropic else 3
        model.geom_friction[:, 0] = mu

    def __call__(self, model, data, mu_axial, mu_lateral):
        data.xfrc_applied[:] = 0
        if not self.anisotropic or data.ncon == 0:
            return
        con = data.contact[:data.ncon]
        g1 = self.static[con.geom[:, 0]]
        g2 = self.static[con.geom[:, 1]]
        sel = (g1 | g2) & (con.dim == 1)
        if not sel.any():
            return
        geom = np.where(g1[sel], con.geom[sel, 1], con.geom[sel, 0])
        body = model.geom_bodyid[geom]
        pos = con.pos[sel]
        normal = con.frame[sel, :3]
        fn = np.abs(data.efc_force[con.efc_address[sel]])

        # velocity of the contact point on the body (cvel is about the subtree CoM)
        cvel = data.cvel[body]
        ref = data.subtree_com[model.body_rootid[body]]
        v = cvel[:, 3:] + np.cross(cvel[:, :3], pos - ref)

        # segment axis (module z) projected on the surface, and the sideways direction
        axis = data.xmat[body].reshape(-1, 3, 3)[:, :, 2].copy()
        axis -= np.einsum("ij,ij->i", axis, normal)[:, None] * normal
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
        side = np.cross(normal, axis)

        va = np.einsum("ij,ij->i", axis, v)
        vs = np.einsum("ij,ij->i", side, v)
        f = -fn[:, None] * (mu_axial * np.tanh(va / self.v0)[:, None] * axis
                            + mu_lateral * np.tanh(vs / self.v0)[:, None] * side)
        torque = np.cross(pos - data.xipos[body], f)
        np.add.at(data.xfrc_applied, (body, slice(0, 3)), f)
        np.add.at(data.xfrc_applied, (body, slice(3, 6)), torque)


class Simulation:
    STARTS = ("corridor", "ramp")
    RAMP_START_GAP = 0.10       # m between the nose tip and the ramp foot when starting at the ramp

    def __init__(self, params, anisotropic, controller="firmware", walls=True, fw_flags=None,
                 start="corridor"):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.params = dict(params)
        self.controller = controller
        self.fw_flags = {name: default for name, (_group, _label, default) in FW_FLAGS.items()}
        self.fw_flags.update(fw_flags or {})
        self.start = start
        self.start_offset = (0.0, 0.0)          # extra sideways shift [m] and heading [deg] at the start
        self.gait = SineGait(self.model, self.data)
        self.fwc = FirmwareController(self.model, self.data)
        self.friction = GroundFriction(self.model)
        self.anisotropic = anisotropic
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "m1_bottom")
        self.wall_geoms = np.flatnonzero(self.model.geom_group == 2)
        self.wall_rgba = self.model.geom_rgba[self.wall_geoms].copy()
        self.ramp_geoms = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ramp_{i + 1}")
                                    for i in range(len(RAMP_SECTIONS))])
        # the model is built lying along +x: remember where its nose tip is in that pose
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.tip0 = self.nose()[:2] + [NOSE_LENGTH / 1000.0, 0.0]
        self.set_walls(walls)
        self.reset()

    def set(self, name, value):
        self.params[name] = value
        if SPECS[name].attr:
            setattr(self.fwc.fw, SPECS[name].attr, value)
        elif name == "mu":
            self.friction.configure(self.model, self.anisotropic, value)
            self.model.geom_friction[self.ramp_geoms, 0] = self.params["mu_ramp"]
        elif name == "mu_ramp":
            self.model.geom_friction[self.ramp_geoms, 0] = value
        elif name == "hill":
            for geom, (pos, quat) in course_poses(value).items():
                g = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                self.model.geom_pos[g] = pos
                self.model.geom_quat[g] = quat

    def set_anisotropic(self, on):
        self.anisotropic = on
        self.set("mu", self.params["mu"])

    def set_walls(self, on):
        """Corridor walls: solid and visible to the IR rays, or gone."""
        self.walls = on
        self.model.geom_contype[self.wall_geoms] = int(on)
        self.model.geom_conaffinity[self.wall_geoms] = int(on)
        self.model.geom_rgba[self.wall_geoms] = self.wall_rgba * [1, 1, 1, float(on)]

    def set_flag(self, name, on):
        """One of the firmware's switches (FW_FLAGS)."""
        self.fw_flags[name] = on
        setattr(self.fwc.fw, name, on)

    def place(self):
        """Put the robot at its start: in the corridor as built, or straight in front
        of the ramp foot.  start_offset shifts it sideways and turns it about its nose tip."""
        if self.start == "ramp":
            origin, heading = ramp_frame()
            tip = origin - self.RAMP_START_GAP * unit(heading)
        else:
            tip, heading = self.tip0.copy(), 0.0
        tip = tip + self.start_offset[0] * unit(heading + 90.0)
        heading += self.start_offset[1]
        c, s = unit(heading)
        self.data.qpos[:2] = tip + np.array([[c, -s], [s, c]]) @ (self.model.qpos0[:2] - self.tip0)
        mujoco.mju_mulQuat(self.data.qpos[3:7], yaw_quat(heading), self.model.qpos0[3:7])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.place()
        self.gait.reset()
        self.fwc.reset()
        for name, on in self.fw_flags.items():        # switches and tunables go into the fresh firmware too
            setattr(self.fwc.fw, name, on)
        for name, value in self.params.items():
            self.set(name, value)
        mujoco.mj_forward(self.model, self.data)
        self.com0 = self.data.subtree_com[self.root].copy()
        self.left_ramp_at = None

    def track(self):
        """Where the robot's centre of mass is relative to the ramp: distance along the
        ground from the foot, sideways offset from the centreline (+ = left), the board
        it is over (0 = before the foot, len + 1 = past the end) and whether it has left
        the ramp sideways."""
        origin, heading = ramp_frame()
        rel = self.data.subtree_com[self.root][:2] - origin
        along, side = rel @ unit(heading), rel @ unit(heading + 90.0)
        ends = ramp_profile(self.params["hill"])[:, 0]
        board = int(np.searchsorted(ends, along))
        on = 0 < board <= len(RAMP_SECTIONS)
        if on and abs(side) > RAMP_WIDTH / 2 and self.left_ramp_at is None:
            self.left_ramp_at = (self.data.time, along, side)
        return along, side, board, self.left_ramp_at

    def command(self, line):
        """A line typed into the firmware's serial port."""
        self.fwc.command(line, self.data.time)
        for spec in PARAMS:                            # 'a' and 'f' change gait values
            if spec.attr:
                self.params[spec.name] = getattr(self.fwc.fw, spec.attr)
        for name in self.fw_flags:                     # 'v', 'h', 'e', 'b', 'k', 'w' flip switches
            self.fw_flags[name] = getattr(self.fwc.fw, name)

    def drain_log(self):
        log, self.fwc.log = self.fwc.log, []
        return log

    def step(self):
        m, d, p = self.model, self.data, self.params
        if self.controller == "firmware":
            self.fwc(m, d, d.ctrl, p["ir_angle"], self.walls)
        else:
            self.gait(p, d.time, m.opt.timestep, d.ctrl)
        self.friction(m, d, p["mu_axial"], p["mu_lateral"])
        mujoco.mj_step(m, d)

    def displacement(self):
        return self.data.subtree_com[self.root] - self.com0

    def nose(self):
        return self.data.site_xpos[self.fwc.site["imu_head"]]


def track_text(sim):
    """One line saying where the robot is on the track."""
    along, side, board, left = sim.track()
    if left:
        return (f"LEFT THE RAMP at t = {left[0]:.0f} s, {left[1] * 100:.0f} cm from the foot "
                f"({'left' if left[2] > 0 else 'right'} edge)")
    if board == 0:
        return f"before the ramp ({-along * 100:.0f} cm to the foot)"
    if board > len(RAMP_SECTIONS):
        return f"past the ramp ({(along - ramp_profile(sim.params['hill'])[-1, 0]) * 100:.0f} cm beyond the end)"
    return (f"ramp board {board}/{len(RAMP_SECTIONS)}, {along * 100:.0f} cm from the foot, "
            f"{side * 100:+.1f} cm off centre (edge at ±{RAMP_WIDTH * 50:.0f})")


# ------------------------------------------------------------------ GUI
class ParamRow:
    """Slider + entry box for one parameter; either one updates the other."""

    def __init__(self, parent, row, spec, value, on_change):
        self.spec, self.on_change = spec, on_change
        self.value = value
        ttk.Label(parent, text=spec.label).grid(row=row, column=0, sticky="w", padx=(0, 8))
        self.var = tk.DoubleVar(value=value)
        ttk.Scale(parent, from_=spec.lo, to=spec.hi, variable=self.var, length=220,
                  command=self._slid).grid(row=row, column=1, sticky="ew")
        self.entry = ttk.Entry(parent, width=8, justify="right")
        self.entry.grid(row=row, column=2, padx=(8, 2))
        ttk.Label(parent, text=spec.unit, width=5).grid(row=row, column=3, sticky="w")
        for event in ("<Return>", "<KP_Enter>", "<FocusOut>"):
            self.entry.bind(event, self._typed)
        self._show(value)

    def _show(self, v):
        decimals = max(0, -int(np.floor(np.log10(self.spec.step))))
        self.entry.delete(0, tk.END)
        self.entry.insert(0, f"{v:.{decimals}f}")

    def _slid(self, _):
        v = round(self.var.get() / self.spec.step) * self.spec.step
        self._apply(v)

    def _typed(self, _):
        try:
            v = float(self.entry.get().replace(",", "."))
        except ValueError:
            self._show(self.value)
            return
        v = min(max(v, self.spec.lo), self.spec.hi)
        self.var.set(v)
        self._apply(v)

    def _apply(self, v):
        self._show(v)
        if v != self.value:
            self.value = v
            self.on_change(self.spec.name, v)

    def set(self, v):
        self.var.set(v)
        self._apply(v)


class ControlPanel:
    FRAME_DT = 1 / 60
    LOG_LINES = 300
    MONO = ("Consolas", 9)

    def __init__(self, sim, viewer):
        self.sim, self.viewer = sim, viewer
        self.paused = False
        self.root = tk.Tk()
        self.root.title("Snake controls")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # one tab per controller; the selected tab drives the robot
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="x")
        parents = {"fw": ttk.Frame(self.tabs, padding=6), "sine": ttk.Frame(self.tabs, padding=6),
                   "world": outer}
        self.tabs.add(parents["fw"], text="ESP32 firmware")
        self.tabs.add(parents["sine"], text="Manual sine")
        self.tabs.select(0 if sim.controller == "firmware" else 1)
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)

        run = ttk.Frame(parents["fw"])
        run.pack(fill="x", pady=(0, 4))
        ttk.Button(run, text="Run (r)", command=lambda: self._send("r")).pack(side="left")
        ttk.Button(run, text="Stop (s)", command=lambda: self._send("s")).pack(side="left", padx=6)
        ttk.Label(parents["sine"], justify="left", font=self.MONO, text=(
            "θ_k = A·sin(2πf·t + k·Δφ [+ φ_vh]) + O\n"
            "k = joint index in its plane, tail → head")).pack(anchor="w", pady=(0, 6))

        self.rows, frames = {}, {}
        for spec in PARAMS:
            key = (spec.tab, spec.group)
            if key not in frames:
                frames[key] = ttk.LabelFrame(parents[spec.tab], text=spec.group, padding=(8, 4))
                frames[key].pack(fill="x", pady=3)
                frames[key].columnconfigure(1, weight=1)
                if spec.group == "Friction":
                    self.aniso = tk.BooleanVar(value=sim.anisotropic)
                    ttk.Checkbutton(frames[key], text="Anisotropic (scales / wheels)",
                                    variable=self.aniso, command=self._toggle_aniso
                                    ).grid(row=0, column=0, columnspan=4, sticky="w")
            frame = frames[key]
            self.rows[spec.name] = ParamRow(frame, frame.grid_size()[1], spec,
                                            sim.params[spec.name], sim.set)
        self.flags = {}
        for name, (group, label, _default) in FW_FLAGS.items():
            frame = frames[("fw", group)]
            var = self.flags[name] = tk.BooleanVar(value=sim.fw_flags[name])
            ttk.Checkbutton(frame, text=label, variable=var,
                            command=lambda name=name, var=var: self.sim.set_flag(name, var.get())
                            ).grid(row=frame.grid_size()[1], column=0, columnspan=4, sticky="w")
        course = frames[("world", "Course and sensors")]
        self.walls = tk.BooleanVar(value=sim.walls)
        ttk.Checkbutton(course, text="Corridor walls", variable=self.walls,
                        command=lambda: self.sim.set_walls(self.walls.get())
                        ).grid(row=course.grid_size()[1], column=0, columnspan=4, sticky="w")

        # serial monitor of the firmware
        serial = ttk.Frame(parents["fw"])
        serial.pack(fill="x", pady=(6, 2))
        ttk.Label(serial, text="Serial >").pack(side="left")
        self.serial = ttk.Entry(serial, font=self.MONO)
        self.serial.pack(side="left", fill="x", expand=True, padx=4)
        self.serial.bind("<Return>", self._serial_entered)
        ttk.Button(serial, text="Send", command=self._serial_entered).pack(side="left")
        self.log = tk.Text(parents["fw"], height=7, width=58, font=("Consolas", 8), state="disabled",
                           wrap="word")
        self.log.pack(fill="x")
        self.telemetry = ttk.Label(parents["fw"], font=self.MONO, justify="left")
        self.telemetry.pack(anchor="w", pady=(4, 0))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(8, 4))
        self.pause_btn = ttk.Button(buttons, text="Pause", command=self._toggle_pause)
        self.pause_btn.pack(side="left")
        ttk.Button(buttons, text="Reset robot", command=self.sim.reset).pack(side="left", padx=6)
        ttk.Button(buttons, text="Default values", command=self._defaults).pack(side="left")
        ttk.Label(buttons, text="Start at").pack(side="left", padx=(12, 4))
        self.start = ttk.Combobox(buttons, values=Simulation.STARTS, state="readonly", width=9)
        self.start.set(sim.start)
        self.start.pack(side="left")
        self.start.bind("<<ComboboxSelected>>", self._start_changed)

        self.status = ttk.Label(outer, font=self.MONO, justify="left")
        self.status.pack(anchor="w", pady=(4, 0))

        self.last = time.perf_counter()
        self.root.after(0, self._tick)

    def _tab_changed(self, _):
        self.sim.controller = "firmware" if self.tabs.index("current") == 0 else "sine"

    def _start_changed(self, _):
        self.sim.start = self.start.get()
        self.sim.reset()

    def _send(self, line):
        self.sim.command(line)
        for spec in PARAMS:                    # show values changed by 'a' / 'f'
            if spec.attr:
                self.rows[spec.name].set(self.sim.params[spec.name])
        for name, var in self.flags.items():   # and switches flipped by 'v', 'h', 'e', 'b', 'k', 'w'
            var.set(self.sim.fw_flags[name])

    def _serial_entered(self, _=None):
        line = self.serial.get().strip()
        self.serial.delete(0, tk.END)
        if line:
            self._log(f"> {line}")
            self._send(line)

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert(tk.END, text + "\n")
        extra = int(self.log.index("end-1c").split(".")[0]) - self.LOG_LINES
        if extra > 0:
            self.log.delete("1.0", f"{extra + 1}.0")
        self.log.see(tk.END)
        self.log.config(state="disabled")

    def _toggle_aniso(self):
        self.sim.set_anisotropic(self.aniso.get())

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    def _defaults(self):
        for spec in PARAMS:
            self.rows[spec.name].set(spec.default)
        self.aniso.set(False)
        self._toggle_aniso()
        self.walls.set(True)
        self.sim.set_walls(True)
        for name, (_group, _label, default) in FW_FLAGS.items():
            self.flags[name].set(default)
            self.sim.set_flag(name, default)

    def _draw_rays(self):
        """The two IR beams in the viewer: red while something is in range."""
        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            if self.sim.controller != "firmware":
                return
            for origin, end, in_range in self.sim.fwc.rays:
                rgba = np.array([1.0, 0.1, 0.1, 1.0] if in_range else [0.5, 0.5, 0.5, 0.6], dtype=np.float32)
                geom = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                                    np.zeros(9), rgba)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3, origin, end)
                scn.ngeom += 1

    def _tick(self):
        if not self.viewer.is_running():
            self.root.destroy()
            return
        now = time.perf_counter()
        elapsed, self.last = min(now - self.last, 0.05), now    # cap: slow PCs run slower, not jerkier
        if not self.paused:
            target = self.sim.data.time + elapsed
            while self.sim.data.time < target:
                self.sim.step()
        self._draw_rays()
        self.viewer.sync()

        for t, line in self.sim.drain_log():
            self._log(f"[{t:6.1f}] {line}")
        f = self.sim.fwc.fw
        self.telemetry.config(text=(
            f"mode {fw.MODE_NAME[f.mode]:8s} amplitude {f.ampScale * 100:3.0f} %   dir {f.gaitDir:+d}\n"
            f"IR  L {f.irLeftCm:5.1f} cm   R {f.irRightCm:5.1f} cm   steer {f.steerDeg:+5.1f}°\n"
            f"pitch  head {f.headImu.pitchAvg:+5.1f}°  tail {f.tailImu.pitchAvg:+5.1f}°  "
            f"base {f.pitchBaseline:+5.1f}°\n"
            "bends  " + "  ".join(f"{fw.JOINT_NAME[j]} {f.pitchTarget[k]:+3.0f}"
                                  for k, j in enumerate(fw.PITCH_JOINTS)) + "\n"
            "joints " + " ".join(f"{a:+4.0f}" for a in f.jointDeg)))

        d = self.sim.displacement()
        t = self.sim.data.time
        self.status.config(text=(
            f"t = {t:6.1f} s{'  (paused)' if self.paused else ''}   "
            f"moved {np.linalg.norm(d[:2]) * 100:6.1f} cm   height {d[2] * 100:+5.1f} cm\n"
            + track_text(self.sim)))
        spent = time.perf_counter() - now
        self.root.after(max(1, int((self.FRAME_DT - spent) * 1000)), self._tick)

    def close(self):
        self.viewer.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ------------------------------------------------------------------ main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", choices=("firmware", "sine"), default="firmware")
    for spec in PARAMS:
        p.add_argument("--" + spec.name.replace("_", "-"), dest=spec.name, type=float,
                       default=spec.default, help=f"{spec.label} [{spec.unit or '-'}] ({spec.tab})")
    names = ", ".join(FW_FLAGS)
    p.add_argument("--yaw-wave", action="store_true",
                   help="firmware: wave on the yaw joints as in the original .ino (same as --off gaitVertical)")
    p.add_argument("--off", default="", metavar="FLAGS",
                   help=f"firmware switches to start switched off, comma separated ({names})")
    p.add_argument("--on", default="", metavar="FLAGS", help="firmware switches to start switched on")
    p.add_argument("--aniso", action="store_true",
                   help="start with anisotropic friction (scales / wheels) instead of isotropic")
    p.add_argument("--start", choices=Simulation.STARTS, default="corridor",
                   help="start in the corridor, or straight in front of the ramp")
    p.add_argument("--no-walls", action="store_true", help="start without the corridor walls")
    p.add_argument("--headless", action="store_true", help="run without the viewer")
    p.add_argument("-T", "--duration", type=float, default=10.0, help="headless duration [s]")
    return p.parse_args()


def run_headless(sim, duration):
    firmware = sim.controller == "firmware"
    if firmware:
        sim.command("r")                      # the real robot waits for "r" (USB or Bluetooth)
    mode = sim.fwc.fw.mode
    board, left = sim.track()[2:]
    while sim.data.time < duration:
        sim.step()
        where = f"[{sim.data.time:6.1f} s]"
        new_board, new_left = sim.track()[2:]
        if new_board != board or new_left != left:        # (sim) lines are not firmware output
            print(where, "(sim)", track_text(sim))
            board, left = new_board, new_left
        if not firmware:
            continue
        f = sim.fwc.fw
        for _, line in sim.drain_log():
            print(where, line)
        if f.mode != mode:
            print(where, f"(sim) mode {fw.MODE_NAME[mode]} -> {fw.MODE_NAME[f.mode]}, "
                         f"IR L {f.irLeftCm:.1f} R {f.irRightCm:.1f} cm")
            mode = f.mode
    if not np.all(np.isfinite(sim.data.qpos)):
        raise SystemExit("simulation diverged")
    d = sim.displacement()
    print(f"{duration:.1f} s: moved {np.linalg.norm(d[:2]) * 100:.1f} cm from the start "
          f"(x {d[0] * 100:+.1f}, y {d[1] * 100:+.1f}, z {d[2] * 100:+.1f} cm); {track_text(sim)}")
    warnings = {name: sim.data.warning[w].number for name, w in mujoco.mjtWarning.__members__.items()
                if int(w) < len(sim.data.warning) and sim.data.warning[w].number}
    if warnings:
        print("MuJoCo warnings:", warnings)


def main():
    args = parse_args()
    flags = {}
    for names, on in ((args.off, False), (args.on, True), ("gaitVertical" if args.yaw_wave else "", False)):
        for name in filter(None, (n.strip() for n in names.split(","))):
            if name not in FW_FLAGS:
                raise SystemExit(f"unknown firmware switch {name!r}; choose from {', '.join(FW_FLAGS)}")
            flags[name] = on
    sim = Simulation({spec.name: getattr(args, spec.name) for spec in PARAMS},
                     anisotropic=args.aniso, controller=args.controller, fw_flags=flags,
                     walls=not args.no_walls, start=args.start)
    if args.headless:
        run_headless(sim, args.duration)
        return

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "m4_bottom")
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, -60.0, -25.0
        ControlPanel(sim, viewer).run()


if __name__ == "__main__":
    main()
