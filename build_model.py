"""Build a MuJoCo model of our snake robot ("Slange robot") from the Inventor STL exports.

One module = Modul_1 (motor slot) --hinge-- Modul_2 (motor clamp, holds the servo).
The hinge runs through the servo shaft on one side and the bearing pin on the
other.  Modul_2's square end plate bolts onto the next module's Modul_1 plate
(4 x M4 on a 55 x 55 mm square, so any multiple of 90 deg roll fits).

Pipeline:
  1. Register every part STL into the frame of Motor_Asembely_modul_1.stl.  The
     assembly file is the parts concatenated, so each part is a contiguous block
     of triangles and its rigid transform follows from Kabsch.
  2. Hinge axis = axis of the 14x5x5 bearing.  Snake axis = direction from the
     Modul_1 end plate to the Modul_2 end plate.  The assembly is saved straight.
  3. Express the parts in body frames (Modul_1: centre of its end plate,
     Modul_2: hinge pivot), write them to meshes/, and emit snake.xml.

Units: STLs are in mm, the MJCF is in m (mesh scale 0.001).
The previous GZ-I/S3003 model is still available: python build_model_gzi.py
"""
import pathlib
import struct

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
STL_DIR = ROOT / "Slange robot" / "Slange robot stl"
MESH_DIR = ROOT / "meshes"
XML_PATH = ROOT / "snake.xml"
ASSEMBLY_STL = "Motor_Asembely_modul_1.stl"

# ---------------------------------------------------------------- parameters
PATTERN = "pypypyp"          # one char per module: p = pitch joint, y = yaw joint
# printed parts: settings from "Slange robot gcode/Modul_2_slange_robot_motor_klammer.3mf"
# (Bambu PLA Basic, 0.2 mm layers, 2 wall loops, 5 top / 3 bottom layers, 15 % grid infill).
# Mass = solid shell under the whole surface + sparse infill inside, see printed_mass().
PLA_DENSITY = 1260.0         # kg/m^3
PRINT_SHELL_MM = 0.82        # mean of walls (0.87), top (1.0) and bottom (0.6); print orientation unknown
PRINT_INFILL = 0.15
SERVO_MASS = 0.044           # kg, Parallax 900-00005 datasheet
BEARING_MASS = 0.003         # kg, 14x5x5 ball bearing (the model is a solid ring)
SERVO_TORQUE = 0.27          # N*m, Parallax 900-00005 stall torque at 6 V (38 oz-in)
SERVO_SPEED = 5.5            # rad/s, Parallax 900-00005 no-load speed at 6 V (0.19 s / 60 deg)
SERVO_KP = 5.0               # N*m/rad, position-loop gain
SERVO_ARMATURE = 1e-3        # kg*m^2, reflected rotor inertia through the gearbox
JOINT_RANGE_DEG = 90.0
FLOOR_FRICTION = 0.5         # isotropic friction coefficient (the "mu" slider in run_sim.py overrides it)
# servo output shaft in the servo mesh's own frame (same S3003 mesh as the GZ-I model):
# the line through (x, y) along local z.  Only used to check the hinge axis.
SERVO_SHAFT_LOCAL = (29.37, 9.19)

# nose module, bolted onto the free end plate at the head.  Positions are in the nose
# STL's own frame (mm), measured from Nase_modul_IRsensorer.stl: y runs from the bolt
# plate (y = 0) to the tip, and the two IR seats are the midpoints of the hole pairs
# (37 mm apart, the Sharp GP2Y0A41 hole spacing) on the outer face of each side wall.
NOSE_STL = "Nase_modul_IRsensorer.stl"
NOSE_PLATE_CENTRE = (-42.5, 0.0, -42.5)
NOSE_LENGTH = 65.0
NOSE_IR_SEATS = {
    "ir_left":  ((-42.16, 24.49, -13.14), (0.0, 0.559, 0.829)),
    "ir_right": ((-42.16, 24.51, -71.89), (0.0, 0.559, -0.829)),
}
IR_SENSOR_DEPTH = 13.5       # mm, Sharp GP2Y0A41 lens above its mounting face
# the seat normals point 56 deg off straight ahead (the firmware comment says "about 30")
IR_ANGLE_DEG = 56.0          # default beam angle from straight ahead, adjustable in run_sim.py

# test track, after the photo of the real one: a walled corridor with turns, then an
# open ramp track (no side walls) in line with the corridor's exit.
# ALL SIZES BELOW ARE ESTIMATES FROM THE PHOTO - replace them with measurements.
#
# corridor: the centreline starts at CORRIDOR_START heading +x (the robot starts in the
# first segment, nose tip at x = 0.30).  Each entry is (segment length in m, turn at its
# end in deg, + = left).  The last section is 1.0 m of straight corridor aimed at the
# ramp: with 0.6 m (shorter than the robot) it came out of the sharp turn still skewed
# and missed the ramp.
CORRIDOR_START = (-0.5, 0.0)
CORRIDOR_PATH = [(0.9, +30.0), (0.7, -30.0), (0.7, +90.0), (1.0, 0.0)]
CORRIDOR_WIDTH = 0.35        # m, between the walls
WALL_HEIGHT = 0.18           # m
WALL_THICKNESS = 0.04        # m
# ramp: boards laid end to end along the corridor's exit direction.  Each entry is
# (length along the board in m, slope sign); the slope angle itself is one value,
# adjustable in run_sim.py.  Equal lengths bring the valley and the end back to the floor.
# RAMP_GAP: open floor between the corridor exit and the ramp foot.  Nothing guides the
# robot there.  With 0.20 m only 2 of 6 simulated runs got over the whole ramp; with the
# foot directly at the exit (0) 5 of 5 did, so the walls should lead right up to the ramp.
RAMP_GAP = 0.00              # m
RAMP_SECTIONS = [(0.75, +1), (0.75, -1), (0.75, +1), (0.60, 0), (0.75, -1)]
RAMP_WIDTH = 0.40            # m
RAMP_ANGLE_DEG = 15.0        # initial slope of the sloped boards
RAMP_FRICTION = 0.8          # rubber sheet on the boards (the "mu on ramp" slider overrides it)
SLAB_THICKNESS = 0.02        # m
SLAB_SINK = 0.001            # m

# part file -> (short name, body, mass source, rgba, collides with floor/other modules)
# The servo body is clamped in Modul_2 and its output shaft drives Modul_1.
# The pin is pressed into Modul_1's lower arm and runs in the bearing seated in Modul_2.
PARTS = {
    "Modul_1_slange_robot_motor_slott":   ("modul1",  "bottom", "pla",     "0.20 0.45 0.80 1", True),
    "lagerpinn_mellom_moduler":           ("pin",     "bottom", "pla",     "0.90 0.90 0.90 1", False),
    "Modul_2_slange_robot_motor_klammer": ("modul2",  "top",    "pla",     "0.95 0.55 0.15 1", True),
    "Servo_motor_forenkla_1":             ("servo",   "top",    "servo",   "0.15 0.15 0.15 1", False),
    "forenklalager_14_5_5":               ("bearing", "top",    "bearing", "0.70 0.70 0.72 1", False),
}


# ------------------------------------------------------------------ STL I/O
def load_stl(path):
    """Binary STL -> (n, 3, 3) float array of triangle vertices."""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n],
                        dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(n, 3, 3).astype(float)


def save_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float32)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    rec["n"] = normals
    rec["v"] = tris.reshape(-1, 9)
    path.write_bytes(b"snake-sim generated".ljust(80, b"\0")
                     + struct.pack("<I", len(tris)) + rec.tobytes())


def mesh_volume(tris):
    """Signed volume (mm^3) of a closed triangle mesh."""
    return np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0


# ------------------------------------------------------------- registration
def edge_signature(tris, decimals=2):
    e = np.stack([np.linalg.norm(tris[:, 0] - tris[:, 1], axis=1),
                  np.linalg.norm(tris[:, 1] - tris[:, 2], axis=1),
                  np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1)], axis=1)
    return np.round(e, decimals)


def kabsch(src, dst):
    """Rotation R and translation t minimising |R @ src + t - dst|."""
    sc, dc = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - sc).T @ (dst - dc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, dc - r @ sc


def register_parts(assembly, parts):
    """Find each part's triangle block in the assembly and its rigid transform.
    Returns {name: (tris in assembly frame, R, t)}."""
    sig_asm = edge_signature(assembly)
    placed = {}
    for name, tris in parts.items():
        sig = edge_signature(tris)
        n = len(tris)
        scores = [np.mean(np.all(sig_asm[s:s + n] == sig, axis=1))
                  for s in range(len(assembly) - n + 1)]
        start = int(np.argmax(scores))
        r, t = kabsch(tris.reshape(-1, 3), assembly[start:start + n].reshape(-1, 3))
        moved = tris @ r.T + t
        err = np.abs(moved - assembly[start:start + n]).max()
        print(f"  {name:36s} triangles {start:5d}-{start + n - 1:5d}  "
              f"match {scores[start]:6.1%}  max error {err:.4f} mm")
        if err > 0.01:
            raise RuntimeError(f"registration of {name} failed (error {err:.3f} mm)")
        placed[name] = (moved, r, t)
    return placed


def line_distance(p, a, point_on_line):
    d = p - point_on_line
    return np.linalg.norm(d - d.dot(a) * a)


def plate_normal(tris, hinge_axis, rough_direction):
    """Normal of the largest flat face perpendicular to the hinge (the end plate),
    signed along rough_direction."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area, 1e-12)[:, None]
    n *= np.where(n @ rough_direction < 0, -1.0, 1.0)[:, None]      # fold +/- together
    ok = np.abs(n @ hinge_axis) < 1e-3
    keys, inverse = np.unique(np.round(n[ok], 3), axis=0, return_inverse=True)
    best = keys[np.argmax(np.bincount(inverse.ravel(), weights=area[ok]))]
    return best / np.linalg.norm(best)


def plate_centre(verts, direction):
    """Centre of the flat face of a part that lies furthest along `direction`."""
    h = verts @ direction
    face = verts[np.abs(h - h.max()) < 0.01]
    return (face.min(0) + face.max(0)) / 2


# ---------------------------------------------------------------- geometry
def module_geometry():
    print("Registering parts into the assembly frame:")
    assembly = load_stl(STL_DIR / ASSEMBLY_STL)
    parts = {name: load_stl(STL_DIR / f"{name}.stl") for name in PARTS}
    placed = register_parts(assembly, parts)
    short = {PARTS[k][0]: v for k, v in placed.items()}

    # hinge axis: the bearing's thinnest local direction, through its centre
    local = parts["forenklalager_14_5_5"].reshape(-1, 3)
    _, r, t = short["bearing"]
    axis_local = np.eye(3)[np.argmin(local.max(0) - local.min(0))]
    hinge_axis = r @ axis_local
    hinge_point = r @ ((local.min(0) + local.max(0)) / 2) + t
    _, r, t = short["servo"]
    shaft = r @ np.array([*SERVO_SHAFT_LOCAL, 0.0]) + t
    shaft_dir = r @ np.array([0.0, 0.0, 1.0])
    print(f"  hinge axis {np.round(hinge_axis, 3)} through {np.round(hinge_point, 2)} mm; "
          f"servo shaft {line_distance(shaft, hinge_axis, hinge_point):.1f} mm off it, "
          f"{np.degrees(np.arccos(min(1.0, abs(shaft_dir @ hinge_axis)))):.1f} deg skew")

    # snake axis: normal of Modul_1's end plate, pointing from Modul_1 towards Modul_2
    m1 = short["modul1"][0].reshape(-1, 3)
    m2 = short["modul2"][0].reshape(-1, 3)
    snake_axis = plate_normal(short["modul1"][0], hinge_axis, m2.mean(0) - m1.mean(0))

    # module frame: origin at the centre of Modul_1's end plate,
    # x = hinge axis, z = snake axis (towards the next module), y = z cross x
    origin = plate_centre(m1, -snake_axis)
    rot = np.column_stack([hinge_axis, np.cross(snake_axis, hinge_axis), snake_axis])
    to_module = lambda p: (p - origin) @ rot

    pivot = to_module(hinge_point)
    pivot -= pivot[0] * np.array([1.0, 0.0, 0.0])      # any point on the axis; take x = 0
    next_module = to_module(plate_centre(m2, snake_axis))
    geo = {
        "pivot": pivot / 1000.0,                 # m, in module frame
        "next": next_module / 1000.0,            # m, next module's origin in module frame
        "meshes": {},
    }
    for name, (tris, _r, _t) in short.items():
        body = next(v[1] for v in PARTS.values() if v[0] == name)
        geo["meshes"][name] = to_module(tris) - (pivot if body == "top" else 0.0)
    ext = np.ptp(np.concatenate([to_module(m1), to_module(m2)]), axis=0)
    print(f"  module pitch {np.linalg.norm(next_module):.1f} mm, cross-section "
          f"{ext[0]:.1f} x {ext[1]:.1f} mm, hinge {pivot[2]:.1f} mm from the Modul_1 plate, "
          f"next module offset along hinge {next_module[0]:+.1f} mm")
    return geo


def printed_mass(tris):
    """Mass (kg) of a part printed with the settings above: a solid shell of
    PRINT_SHELL_MM under its surface, and PRINT_INFILL of the rest."""
    volume = abs(mesh_volume(tris))                                                       # mm^3
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum() / 2
    shell = min(volume, area * PRINT_SHELL_MM)
    return (shell + PRINT_INFILL * (volume - shell)) * 1e-9 * PLA_DENSITY


def part_mass(name, tris):
    source = next(v[2] for v in PARTS.values() if v[0] == name)
    if source == "servo":
        return SERVO_MASS
    if source == "bearing":
        return BEARING_MASS
    return printed_mass(tris)


# --------------------------------------------------------------------- MJCF
def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def quat_from_mat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).flatten())
    return q


def unit(heading_deg):
    h = np.radians(heading_deg)
    return np.array([np.cos(h), np.sin(h)])


def yaw_quat(heading_deg):
    h = np.radians(heading_deg)
    return np.array([np.cos(h / 2), 0.0, 0.0, np.sin(h / 2)])


def corridor_centreline():
    """Corner points (n+1, 2) and segment headings (n) of the corridor centreline."""
    points, headings = [np.array(CORRIDOR_START, dtype=float)], []
    heading = 0.0
    for length, turn in CORRIDOR_PATH:
        headings.append(heading)
        points.append(points[-1] + length * unit(heading))
        heading += turn
    return np.array(points), headings


def corridor_walls():
    """[(name, centre xy, half length, heading)] of the wall boxes on both sides.
    The inner faces of neighbouring walls meet at the corners; on the outside of a
    turn the boxes are lengthened so that they overlap instead of leaving a wedge."""
    points, headings = corridor_centreline()
    n = len(headings)
    turns = [headings[i + 1] - headings[i] for i in range(n - 1)]
    walls = []
    for side, label in ((+1, "left"), (-1, "right")):
        normals = [side * np.array([-np.sin(np.radians(h)), np.cos(np.radians(h))]) for h in headings]
        # inner-face corner points: end points, and mitre points at the turns
        corners = [points[0] + CORRIDOR_WIDTH / 2 * normals[0]]
        for i in range(n - 1):
            mitre = (normals[i] + normals[i + 1]) / (1 + normals[i] @ normals[i + 1])
            corners.append(points[i + 1] + CORRIDOR_WIDTH / 2 * mitre)
        corners.append(points[-1] + CORRIDOR_WIDTH / 2 * normals[-1])
        for i in range(n):
            a, b = corners[i], corners[i + 1]
            along = unit(headings[i])
            # this wall is on the outside of a turn if the turn goes away from its side
            if i > 0 and turns[i - 1] * side < 0:
                a = a - WALL_THICKNESS * np.tan(np.radians(abs(turns[i - 1])) / 2) * along
            if i < n - 1 and turns[i] * side < 0:
                b = b + WALL_THICKNESS * np.tan(np.radians(abs(turns[i])) / 2) * along
            centre = (a + b) / 2 + WALL_THICKNESS / 2 * normals[i]
            walls.append((f"wall_{label}_{i + 1}", centre, np.linalg.norm(b - a) / 2, headings[i]))
    return walls


def ramp_frame():
    """Foot of the ramp (xy) and its heading: in line with the corridor exit."""
    points, headings = corridor_centreline()
    return points[-1] + RAMP_GAP * unit(headings[-1]), headings[-1]


def ramp_profile(angle_deg):
    """Points (n+1, 2) of the ramp's top surface as (distance along the ground, height)."""
    profile = [np.zeros(2)]
    for length, sign in RAMP_SECTIONS:
        a = np.radians(sign * angle_deg)
        profile.append(profile[-1] + length * np.array([np.cos(a), np.sin(a)]))
    return np.array(profile)


def course_poses(angle_deg):
    """{geom name: (pos, quat)} of the ramp boards for a given slope angle.
    The boards sit SLAB_SINK below their nominal height so that the end faces at the
    foot stay under the floor: flush with it, a body resting a fraction of a mm into
    the (soft) floor contact would catch on them like on a step."""
    origin, heading = ramp_frame()
    forward = np.array([*unit(heading), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    profile = ramp_profile(angle_deg)
    poses = {}
    for i, (length, sign) in enumerate(RAMP_SECTIONS):
        a = np.radians(sign * angle_deg)
        along = np.cos(a) * forward + np.sin(a) * up
        normal = -np.sin(a) * forward + np.cos(a) * up
        start = np.array([*origin, -SLAB_SINK]) + profile[i, 0] * forward + profile[i, 1] * up
        pos = start + along * length / 2 - normal * SLAB_THICKNESS / 2
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, yaw_quat(heading), np.array([np.cos(-a / 2), 0.0, np.sin(-a / 2), 0.0]))
        poses[f"ramp_{i + 1}"] = (pos, quat)
    return poses


def head_frame(roll_deg):
    """Body-frame directions (forward, left, up) of a module rolled by roll_deg.
    Module frame: x = hinge, z = snake axis; at roll 0 the hinge points left and y up."""
    r = np.radians(roll_deg)
    return (np.array([0.0, 0.0, 1.0]),
            np.array([np.cos(r), -np.sin(r), 0.0]),
            np.array([np.sin(r), np.cos(r), 0.0]))


def nose_geometry(geo, roll_deg):
    """Nose mesh and sensor sites in the frame of the head's top body.  The nose's
    bolt plate sits centred on the free Modul_2 plate, tip forward, IR seats left/right."""
    fwd, left, up = head_frame(roll_deg)
    attach = (geo["next"] - geo["pivot"]) * 1000.0            # mm, centre of the free plate
    rot = np.column_stack([up, fwd, left])                    # nose STL x, y, z in the body frame
    place = lambda p: attach + (np.asarray(p, dtype=float) - NOSE_PLATE_CENTRE) @ rot.T
    tris = load_stl(STL_DIR / NOSE_STL)
    frame = quat_from_mat(np.column_stack([fwd, left, up]))   # site axes: x forward, y left, z up
    sites = {"imu_head": (attach / 1000.0, frame)}
    for name, (seat, normal) in NOSE_IR_SEATS.items():
        lens = np.asarray(seat) + IR_SENSOR_DEPTH * np.asarray(normal)
        sites[name] = (place(lens) / 1000.0, frame)
    return place(tris), sites


def geoms_xml(body, geo, indent):
    lines = []
    for short, b, _source, rgba, collides in PARTS.values():
        if b != body:
            continue
        mass = part_mass(short, geo["meshes"][short])
        cls = "collision" if collides else "visual"
        lines.append(f'{indent}<geom class="{cls}" mesh="{short}" rgba="{rgba}" mass="{mass:.5f}"/>')
    return lines


def module_rolls(pattern):
    """Roll of each module about the snake axis.  Pitch modules sit at 0/180 deg,
    yaw modules at 90/270 deg.  Each step turns by +90 deg (type change) or
    180 deg (same type), so the 5 mm hinge-axis step of every module alternates
    direction and the snake stays straight instead of drifting sideways/upwards."""
    rolls = [0.0 if pattern[0] == "p" else 90.0]
    for prev, cur in zip(pattern, pattern[1:]):
        rolls.append((rolls[-1] + (90.0 if prev != cur else 180.0)) % 360.0)
    return rolls


def site_xml(name, pos, quat, indent):
    return f'{indent}<site name="{name}" pos="{fmt(pos)}" quat="{fmt(quat)}" size="0.004" rgba="1 0 0 1"/>'


def course_xml():
    """Static course geoms: the ramp boards (posed again at runtime when the slope
    slider moves) and the corridor walls (geom group 2)."""
    lines = []
    for (name, (pos, quat)), (length, _sign) in zip(course_poses(RAMP_ANGLE_DEG).items(), RAMP_SECTIONS):
        size = np.array([length, RAMP_WIDTH, SLAB_THICKNESS]) / 2
        lines.append(f'    <geom name="{name}" class="ramp" size="{fmt(size)}" pos="{fmt(pos)}" '
                     f'quat="{fmt(quat)}"/>')
    for name, centre, half_length, heading in corridor_walls():
        size = [half_length, WALL_THICKNESS / 2, WALL_HEIGHT / 2]
        lines.append(f'    <geom name="{name}" class="wall" size="{fmt(size)}" '
                     f'pos="{fmt([*centre, WALL_HEIGHT / 2])}" quat="{fmt(yaw_quat(heading))}"/>')
    return "\n".join(lines)


def build_xml(geo, pattern, root_z, nose_sites, nose_mass):
    pivot = geo["pivot"]
    # module frame -> world: hinge (x) along world y, module y up, snake axis (z) along world x
    # the nose tip starts where the head plate would be without a nose
    root_pos = np.array([-len(pattern) * geo["next"][2] / 2 - NOSE_LENGTH / 1000.0, 0.0, root_z])
    rolls = np.radians(module_rolls(pattern))
    tail_fwd, tail_left, tail_up = head_frame(np.degrees(rolls[0]))
    tail_frame = quat_from_mat(np.column_stack([tail_fwd, tail_left, tail_up]))

    lines = []
    indent = "    "
    for i, c in enumerate(pattern, start=1):
        if i == 1:
            q = quat_from_mat(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
                              @ np.array([[np.cos(rolls[0]), -np.sin(rolls[0]), 0],
                                          [np.sin(rolls[0]), np.cos(rolls[0]), 0], [0, 0, 1]]))
            lines.append(f'{indent}<body name="m1_bottom" pos="{fmt(root_pos)}" quat="{fmt(q)}">')
            lines.append(f'{indent}  <freejoint name="root"/>')
        else:
            rel = rolls[i - 1] - rolls[i - 2]
            q = np.array([np.cos(rel / 2), 0, 0, np.sin(rel / 2)])
            lines.append(f'{indent}<body name="m{i}_bottom" pos="{fmt(geo["next"] - pivot)}" '
                         f'quat="{fmt(q)}">')
        lines += geoms_xml("bottom", geo, indent + "  ")
        if i == 1:
            lines.append(site_xml("imu_tail", np.zeros(3), tail_frame, indent + "  "))
        lines.append(f'{indent}  <body name="m{i}_top" pos="{fmt(pivot)}">')
        lines.append(f'{indent}    <joint name="j{i}" class="servo"/>')
        lines += geoms_xml("top", geo, indent + "    ")
        if i == len(pattern):
            lines.append(f'{indent}    <geom class="collision" mesh="nose" rgba="0.30 0.70 0.35 1" '
                         f'mass="{nose_mass:.5f}"/>')
            for name, (pos, quat) in nose_sites.items():
                lines.append(site_xml(name, pos, quat, indent + "    "))
        indent += "    "
    for _ in pattern:
        indent = indent[:-4]
        lines.append(f"{indent}  </body>")
        lines.append(f"{indent}</body>")
    bodies = "\n".join(lines)

    excludes = []
    for i in range(1, len(pattern)):
        for a in (f"m{i}_bottom", f"m{i}_top"):
            for b in (f"m{i + 1}_bottom", f"m{i + 1}_top"):
                excludes.append(f'    <exclude body1="{a}" body2="{b}"/>')
    excludes = "\n".join(excludes)

    meshes = "\n".join(f'    <mesh name="{short}" file="{short}.stl" scale="0.001 0.001 0.001"/>'
                       for short in [v[0] for v in PARTS.values()] + ["nose"])
    actuators = "\n".join(f'    <position name="servo{i}" joint="j{i}" class="servo"/>'
                          for i in range(1, len(pattern) + 1))
    rng = JOINT_RANGE_DEG
    return f"""<!-- Generated by build_model.py, do not edit by hand. -->
<mujoco model="slange_robot_{pattern}">
  <compiler angle="degree" meshdir="meshes" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>

  <default>
    <geom condim="3" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
    <default class="collision">
      <geom type="mesh" contype="1" conaffinity="1" group="0"/>
    </default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="0"/>
    </default>
    <default class="wall">
      <geom type="box" group="2" rgba="0.86 0.66 0.42 1"/>
    </default>
    <!-- priority 1: contacts on the boards use the board's friction, not the larger of the two -->
    <default class="ramp">
      <geom type="box" rgba="0.12 0.12 0.13 1" friction="{RAMP_FRICTION} 0.005 0.0001" priority="1"/>
    </default>
    <!-- servo: damping = stall torque / no-load speed gives the torque-speed line -->
    <default class="servo">
      <joint type="hinge" axis="1 0 0" range="-{rng} {rng}"
             damping="{SERVO_TORQUE / SERVO_SPEED:.4f}" armature="{SERVO_ARMATURE}"/>
      <position kp="{SERVO_KP}" ctrlrange="-{rng} {rng}" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>
    </default>
  </default>

  <asset>
{meshes}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.86" rgb2="0.72 0.74 0.77"
             width="512" height="512" mark="edge" markrgb="0.6 0.6 0.6"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.95 0.95 0.97"
             width="256" height="256"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
{course_xml()}
{bodies}
  </worldbody>

  <contact>
{excludes}
  </contact>

  <actuator>
{actuators}
  </actuator>

  <sensor>
    <accelerometer name="acc_head" site="imu_head"/>
    <accelerometer name="acc_tail" site="imu_tail"/>
  </sensor>
</mujoco>
"""


def lowest_point(model):
    """Lowest world z of any collision mesh vertex in the initial pose."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low = np.inf
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[g] == 0:
            continue
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        low = min(low, (v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
    return low


def main():
    geo = module_geometry()
    MESH_DIR.mkdir(exist_ok=True)
    for name, tris in geo["meshes"].items():
        save_stl(MESH_DIR / f"{name}.stl", tris)
    rolls = module_rolls(PATTERN)
    nose_tris, nose_sites = nose_geometry(geo, rolls[-1])
    save_stl(MESH_DIR / "nose.stl", nose_tris)
    nose_mass = printed_mass(nose_tris)

    # build once, then lift the snake so its lowest point sits 2 mm above the floor
    XML_PATH.write_text(build_xml(geo, PATTERN, 0.0, nose_sites, nose_mass))
    low = lowest_point(mujoco.MjModel.from_xml_path(str(XML_PATH)))
    XML_PATH.write_text(build_xml(geo, PATTERN, 0.002 - low, nose_sites, nose_mass))

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    total = model.body_subtreemass[1]
    print(f"  module rolls {[int(r) for r in rolls]} deg")
    print(f"Wrote {XML_PATH.name}: {model.nbody - 1} bodies, {model.njnt - 1} servo joints, "
          f"{model.nu} actuators, total mass {total * 1000:.0f} g "
          f"({(total - nose_mass) / len(PATTERN) * 1000:.1f} g per module + {nose_mass * 1000:.0f} g nose)")


if __name__ == "__main__":
    main()
