"""Build a MuJoCo model of the modular snake robot from the SolidWorks STL exports.

Pipeline:
  1. Register every part STL (each in its own local frame) into the frame of
     assembly1-4.17_TOP.stl.  The assembly file is the six parts concatenated,
     so each part occupies a contiguous block of triangles and the rigid
     transform follows from Kabsch on corresponding vertices.
  2. The assembly was saved with the joint bent; rotate the top half about the
     hinge axis back to the straight pose.
  3. Express the parts in body frames (bottom half: bottom-face center,
     top half: hinge pivot), write them to meshes/, and emit snake.xml.

Units: STLs are in mm, the MJCF is in m (mesh scale 0.001).
"""
import pathlib
import struct

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
MESH_DIR = ROOT / "meshes"
XML_PATH = ROOT / "snake.xml"
ASSEMBLY_STL = "assembly1-4.17_TOP.stl"

# ---------------------------------------------------------------- parameters
PATTERN = "pypypyp"          # one char per module: p = pitch joint, y = yaw joint
PLA_DENSITY = 1240.0         # kg/m^3, printed plates / horn plate
STEEL_DENSITY = 7850.0       # kg/m^3, shaft and bushing
SERVO_MASS = 0.037           # kg, Futaba S3003 datasheet
SERVO_TORQUE = 0.31          # N*m, S3003 stall torque at 4.8 V (3.2 kg*cm)
SERVO_SPEED = 4.55           # rad/s, S3003 no-load speed at 4.8 V (0.23 s / 60 deg)
SERVO_KP = 5.0               # N*m/rad, position-loop gain
SERVO_ARMATURE = 1e-3        # kg*m^2, reflected rotor inertia through the gearbox
JOINT_RANGE_DEG = 90.0
FLOOR_FRICTION = 0.5         # used when anisotropic friction is switched off

# hill: a flat ramp ahead of the snake, hinged at its lower edge (angle adjustable in run_sim.py)
HILL_START = 0.40            # m, x of the lower edge (the snake's head starts at x = 0.27)
HILL_LENGTH = 3.0            # m, along the slope
HILL_WIDTH = 3.0             # m
HILL_THICKNESS = 0.05        # m
HILL_ANGLE_DEG = 10.0        # initial slope

# part -> (half, material, rgba, collides with the floor/other modules)
PARTS = {
    "dimokuai":   ("bottom", "pla",   "0.20 0.45 0.80 1", True),
    "S3003":      ("bottom", "servo", "0.15 0.15 0.15 1", False),
    "zhoutao":    ("bottom", "steel", "0.70 0.70 0.72 1", False),
    "dingmokuai": ("top",    "pla",   "0.95 0.55 0.15 1", True),
    "servoplate": ("top",    "pla",   "0.90 0.90 0.90 1", False),
    "zhou":       ("top",    "steel", "0.70 0.70 0.72 1", False),
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
    """Find each part's triangle block in the assembly and its rigid transform."""
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
        print(f"  {name:11s} triangles {start:5d}-{start + n - 1:5d}  "
              f"match {scores[start]:6.1%}  max error {err:.4f} mm")
        if err > 0.01:
            raise RuntimeError(f"registration of {name} failed (error {err:.3f} mm)")
        placed[name] = moved
    return placed


def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def straighten_top_half(placed, pivot):
    """Rotation about the hinge (x axis through pivot) that puts the top half
    in the straight pose: stacked on the bottom half with the same footprint."""
    bottom = placed["dimokuai"]
    lo, hi = bottom.reshape(-1, 3).min(0), bottom.reshape(-1, 3).max(0)
    top = placed["dingmokuai"].reshape(-1, 3)
    best = None
    for deg in np.arange(-180.0, 180.0, 0.1):
        v = (top - pivot) @ rot_x(np.radians(deg)).T + pivot
        # footprint mismatch in y, plus penalty if it is not above the bottom half
        cost = abs(v[:, 1].min() - lo[1]) + abs(v[:, 1].max() - hi[1])
        cost += 0.0 if v[:, 2].max() > hi[2] + 5 else 1e3
        if best is None or cost < best[0]:
            best = (cost, deg)
    # refine to the exact angle
    fine = np.arange(best[1] - 0.1, best[1] + 0.1, 0.001)
    costs = [abs(((top - pivot) @ rot_x(np.radians(a)).T + pivot)[:, 1].min() - lo[1])
             + abs(((top - pivot) @ rot_x(np.radians(a)).T + pivot)[:, 1].max() - hi[1])
             for a in fine]
    return np.radians(fine[int(np.argmin(costs))])


# ---------------------------------------------------------------- geometry
def module_geometry():
    print("Registering parts into the assembly frame:")
    assembly = load_stl(ROOT / ASSEMBLY_STL)
    parts = {name: load_stl(ROOT / f"{name}-4.17.stl") for name in PARTS}
    placed = register_parts(assembly, parts)

    # hinge axis = assembly x, through the centre of the idler shaft
    shaft = placed["zhou"].reshape(-1, 3)
    pivot = np.array([0.0, *(shaft.min(0)[1:] + shaft.max(0)[1:]) / 2])

    straight = straighten_top_half(placed, pivot)
    print(f"  top half straightened by {np.degrees(straight):.2f} deg about the hinge")
    for name, (half, *_rest) in PARTS.items():
        if half == "top":
            placed[name] = (placed[name] - pivot) @ rot_x(straight).T + pivot

    # module frame: origin at the centre of the bottom face, z along the snake
    b = placed["dimokuai"].reshape(-1, 3)
    origin = np.array([(b[:, 0].min() + b[:, 0].max()) / 2,
                       (b[:, 1].min() + b[:, 1].max()) / 2,
                       b[:, 2].min()])
    top_face = placed["dingmokuai"].reshape(-1, 3)[:, 2].max()
    geo = {
        "pivot": (pivot - origin) / 1000.0,             # m, in module frame
        "length": (top_face - origin[2]) / 1000.0,      # m, module pitch
        "footprint": (b.max(0)[:2] - b.min(0)[:2]) / 1000.0,
        "meshes": {},
    }
    geo["pivot"][0] = 0.0
    for name, (half, *_rest) in PARTS.items():
        # bottom-half parts live in the module frame, top-half parts in the hinge frame
        ref = origin if half == "bottom" else origin + geo["pivot"] * 1000.0
        geo["meshes"][name] = placed[name] - ref
    print(f"  module length {geo['length'] * 1000:.1f} mm, footprint "
          f"{geo['footprint'][0] * 1000:.1f} x {geo['footprint'][1] * 1000:.1f} mm, "
          f"hinge at y={geo['pivot'][1] * 1000:.2f} z={geo['pivot'][2] * 1000:.2f} mm")
    return geo


def part_mass(name, tris):
    material = PARTS[name][1]
    if material == "servo":
        return SERVO_MASS
    density = PLA_DENSITY if material == "pla" else STEEL_DENSITY
    return abs(mesh_volume(tris)) * 1e-9 * density


# --------------------------------------------------------------------- MJCF
def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def quat_from_mat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).flatten())
    return q


def hill_pose(angle_deg, start=HILL_START, length=HILL_LENGTH, thickness=HILL_THICKNESS):
    """Centre and orientation of the hill box whose top face starts at (start, 0, 0)
    and rises along +x at angle_deg."""
    a = np.radians(angle_deg)
    along = np.array([np.cos(a), 0.0, np.sin(a)])
    normal = np.array([-np.sin(a), 0.0, np.cos(a)])
    pos = np.array([start, 0.0, 0.0]) + along * length / 2 - normal * thickness / 2
    quat = np.array([np.cos(-a / 2), 0.0, np.sin(-a / 2), 0.0])   # rotation about y by -a
    return pos, quat


def geoms_xml(half, geo, indent):
    lines = []
    for name, (h, _material, rgba, collides) in PARTS.items():
        if h != half:
            continue
        mass = part_mass(name, geo["meshes"][name])
        cls = "collision" if collides else "visual"
        lines.append(f'{indent}<geom class="{cls}" mesh="{name}" rgba="{rgba}" mass="{mass:.5f}"/>')
    return lines


def build_xml(geo, pattern):
    L = geo["length"]
    pivot = geo["pivot"]
    height = geo["footprint"].max() / 2      # lowest-hanging module rests on the floor
    # module frame -> world: hinge (x) along world y, module y up, snake axis (z) along world x
    root_quat = quat_from_mat(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    root_pos = np.array([-len(pattern) * L / 2, 0.0, height + 0.002])
    # absolute roll of each module about the snake axis: pitch 0 deg, yaw 90 deg
    roll = [0.0 if c == "p" else np.pi / 2 for c in pattern]

    lines = []
    indent = "    "
    for i, c in enumerate(pattern, start=1):
        if i == 1:
            lines.append(f'{indent}<body name="m1_bottom" pos="{fmt(root_pos)}" quat="{fmt(root_quat)}">')
            lines.append(f'{indent}  <freejoint name="root"/>')
        else:
            rel = roll[i - 1] - roll[i - 2]
            q = np.array([np.cos(rel / 2), 0, 0, np.sin(rel / 2)])
            pos = np.array([0.0, 0.0, L]) - pivot
            lines.append(f'{indent}<body name="m{i}_bottom" pos="{fmt(pos)}" quat="{fmt(q)}">')
        lines += geoms_xml("bottom", geo, indent + "  ")
        lines.append(f'{indent}  <body name="m{i}_top" pos="{fmt(pivot)}">')
        lines.append(f'{indent}    <joint name="j{i}" class="servo"/>')
        lines += geoms_xml("top", geo, indent + "    ")
        indent += "    "
    for i in range(len(pattern), 0, -1):
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

    meshes = "\n".join(f'    <mesh name="{name}" file="{name}.stl" scale="0.001 0.001 0.001"/>'
                       for name in PARTS)
    actuators = "\n".join(f'    <position name="servo{i}" joint="j{i}" class="servo"/>'
                          for i in range(1, len(pattern) + 1))
    rng = JOINT_RANGE_DEG
    hill_pos, hill_quat = hill_pose(HILL_ANGLE_DEG)
    hill_size = np.array([HILL_LENGTH, HILL_WIDTH, HILL_THICKNESS]) / 2
    return f"""<!-- Generated by build_model.py, do not edit by hand. -->
<mujoco model="modular_snake_{pattern}">
  <compiler angle="degree" meshdir="meshes" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>

  <custom>
    <numeric name="hill_start" data="{HILL_START}"/>
  </custom>

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
    <!-- S3003: damping = stall torque / no-load speed gives the torque-speed line -->
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
    <texture name="hill" type="2d" builtin="checker" rgb1="0.55 0.68 0.50" rgb2="0.48 0.60 0.44"
             width="512" height="512"/>
    <material name="hill" texture="hill" texrepeat="4 4" texuniform="true"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.95 0.95 0.97"
             width="256" height="256"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <geom name="hill" type="box" size="{fmt(hill_size)}" pos="{fmt(hill_pos)}" quat="{fmt(hill_quat)}"
          material="hill"/>
{bodies}
  </worldbody>

  <contact>
{excludes}
  </contact>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


def main():
    geo = module_geometry()
    MESH_DIR.mkdir(exist_ok=True)
    for name, tris in geo["meshes"].items():
        save_stl(MESH_DIR / f"{name}.stl", tris)
    XML_PATH.write_text(build_xml(geo, PATTERN))

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    print(f"Wrote {XML_PATH.name}: {model.nbody - 1} bodies, {model.njnt - 1} servo joints, "
          f"{model.nu} actuators, total mass {model.body_subtreemass[1] * 1000:.0f} g "
          f"({model.body_subtreemass[1] / len(PATTERN) * 1000:.1f} g per module)")


if __name__ == "__main__":
    main()
