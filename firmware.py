"""Python port of the ESP32 snake controller ("MMA4004 snake robot - main controller").

The control logic is ported function by function, with the firmware's own names
(camelCase variables, UPPER_CASE constants), so it can be compared with the
.ino side by side.  Only the hardware layer is replaced:

    millis()                 -> now_ms passed to tick()
    analogReadMilliVolts()   -> ir_left_mv / ir_right_mv passed to tick()
    MC6470 I2C read          -> head_acc / tail_acc (g, board axes) passed to tick()
    ledcWrite()              -> servoDeg[], the angle the PWM pulse asks the servo for
    Serial                   -> command(line) in, list of printed lines out (self.out)

It mirrors snake_robot/snake_robot.ino, which is the version pasted on 2026-10-01
plus additions that came out of the simulation (marked [SIM] there, NOT IN THE
PASTED .ino here; the serial command that toggles each one is in brackets):
  gaitVertical [v] - the travelling wave runs on the pitch joints instead of the yaw
                     joints.  False gives the original gait.
  irPeakHold       - walls are judged from the farthest IR reading of a wave period,
                     because the nose dips and the beams then see the floor.
  slopeSteer   [h] - where there are no walls, steer along the fall line with the
                     head accelerometer.
  edgeSteer    [e] - and away from a side where the beams find no board at the dips.
  bendFollow   [b] - the body bend follows the terrain (head pitch minus tail pitch).
  slopeBends   [k] - the original valley / crest kinks, off by default.
  wallTurn     [w] - at a wall, keep crawling and keep turning away for as long as the
                     wall is at the nose.  False gives the original: back up, then turn.

Joint order everywhere is the firmware's: head to tail, P1 Y2 P3 Y4 P5 Y6 P7.
Sign conventions: yaw +deg = body bends LEFT, pitch +deg = "valley" bend (the part
in front of the joint tilts UP), IMU pitch +deg = nose up.
"""
import math

JOINT_NAME = ["P1", "Y2", "P3", "Y4", "P5", "Y6", "P7"]
YAW_JOINTS = [1, 3, 5]        # Y2 Y4 Y6
PITCH_JOINTS = [0, 2, 4, 6]   # P1 P3 P5 P7

STOPPED, CRAWL, BACKING, TURNING = range(4)
MODE_NAME = ["STOPPED", "CRAWL", "BACKING", "TURNING"]

AVG_N = 100                   # samples in the pitch average (buffer size, fixed)
MAX_KINKS = 3                 # bends that can travel down the body at the same time


def constrain(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def approach(x, target, maxStep):
    if x < target:
        return min(x + maxStep, target)
    return max(x - maxStep, target)


class Imu:
    def __init__(self, name, fwdAxis, fwdSign, leftAxis, leftSign):
        self.name = name
        self.fwdAxis = fwdAxis        # board axis pointing toward the head: 0 = X, 1 = Y, 2 = Z
        self.fwdSign = fwdSign
        self.leftAxis = leftAxis      # board axis pointing to the robot's left (slope steering)
        self.leftSign = leftSign
        self.ok = False
        self.acc = [0.0, 0.0, 0.0]    # last reading in g
        self.pitch = 0.0              # last single sample, deg
        self.pitchAvg = 0.0           # averaged over one wave period, deg
        self.buf = [0.0] * AVG_N
        self.bufIdx = 0
        self.bufSum = 0.0
        self.primed = False
        # slope steering: forward and leftward acceleration, averaged like the pitch
        self.fwdAvg = 0.0             # g
        self.leftAvg = 0.0            # g
        self.bufFwd = [0.0] * AVG_N
        self.bufLeft = [0.0] * AVG_N
        self.sumFwd = 0.0
        self.sumLeft = 0.0


class Kink:
    def __init__(self):
        self.active = False
        self.bend = 0.0               # deg, + valley, - crest
        self.travelM = 0.0            # how far the robot has moved since the nose reached it


class Firmware:
    # ---- 2. servo calibration (Parallax 900-00005)
    PWM_BITS = 14
    SERVO_CENTER_US = 1500
    SERVO_US_PER_DEG = 11.1
    SERVO_MIN_US = 600
    SERVO_MAX_US = 2400
    JOINT_LIMIT_DEG = 60.0

    # ---- 3. gait
    LOOP_HZ = 50
    RAMP_PER_S = 0.5
    PITCH_RATE_DEG_S = 60.0

    # ---- 4. steering from the IR sensors
    IR_MIN_CM = 4.0
    IR_MAX_CM = 30.0
    IR_TAU_S = 0.15
    STEER_START_CM = 25.0
    STEER_GAIN = 2.5              # was 1.5
    STEER_MAX_DEG = 20.0
    STEER_TAU_S = 0.3
    BACKUP_CM = 7.0
    BACKUP_MS = 2500
    TURN_MS = 3000
    # NOT IN THE PASTED .ino, wallTurn (see wallCheck)
    WALL_CLEAR_CM = 10.0          # closer than this, the wall is still at the nose
    WALL_STEADY_CM = 3.0          # readings of one nose-up phase this close together: a real distance
    WALL_HOLD_MS = 6000           # no verdict for this long: end the turn anyway
    WALL_STUCK_MS = 30000         # still at the wall after this long (a turn takes up to 20 s): wedged

    # ---- 4b. steering on slopes (NOT IN THE PASTED .ino)
    # On the ramp there are no walls for the IR sensors.  On a slope gravity shows
    # which way is uphill, so the head accelerometer can hold the robot on the fall line.
    SLOPE_MIN_DEG = 6.0           # flatter than this: the accelerometer cannot tell the direction
    SLOPE_STEER_GAIN = 0.5        # deg of yaw offset per deg of heading error against the fall line
    OPEN_STEER_MAX_DEG = 8.0      # limit of the steering without walls: a body that is arched by the
                                  # wave and bent sideways as well rolls over onto its side
    # Gravity gives the heading but not the sideways position.  For that, use the nose
    # dips of the vertical wave: at each dip the beams hit the board beside the nose.
    # A side that finds nothing near during a whole period has no board under it.
    EDGE_CM = 25.0                # nearest reading of a period above this: no board on that side
    EDGE_STEER_DEG = 6.0          # yaw offset away from the edge
    EDGE_CREST_DEG = 10.0         # head pitched this much below the tail: over a crest, no edge steering

    # ---- 4c. body bend follows the terrain (NOT IN THE PASTED .ino)
    # Over a crest or in a valley the terrain bends the body against the pitch servos;
    # that takes their torque and the wave stops driving.  The bend shows up as a
    # difference between head and tail pitch.  Adopt it as the commanded shape (on the
    # two middle pitch joints), and let it relax when the terrain stops forcing it.
    BEND_GAIN_PER_S = 0.5         # how fast the commanded bend follows the measured one
    BEND_LEAK_TAU_S = 20.0        # relaxes back to straight with this time constant
    BEND_DEADBAND_DEG = 5.0       # ignore small differences (sensor offsets, the wave)
    BEND_MAX_DEG = 40.0           # total over the body

    # ---- 5. slope transitions
    BASELINE_TAU_S = 4.0
    KINK_TRIGGER_DEG = 6.0
    VALLEY_BEND_DEG = 30.0
    CREST_BEND_DEG = 20.0
    CRAWL_SPEED_MPS = 0.03
    PITCH_SPACING_M = 0.17
    REARM_MS = 2500

    def __init__(self):
        self.JOINT_TRIM_DEG = [0.0] * 7
        self.JOINT_DIR = [1] * 7

        self.gaitAmpDeg = 45.0        # was 30 in the pasted .ino; 45 with STEER_GAIN 2.5 gets over the course in the simulation
        self.gaitFreqHz = 0.5
        self.gaitPhaseDeg = 120.0
        # NOT IN THE PASTED .ino: which joints carry the travelling wave.
        #   True  - pitch joints (vertical "caterpillar" wave, runs tail to head).  Works
        #           with the same friction in every direction, i.e. without wheels.
        #   False - yaw joints, exactly as in the .ino (lateral undulation; only moves
        #           the robot if sliding sideways is harder than sliding forwards).
        self.gaitVertical = True
        self.slopeSteer = True        # NOT IN THE PASTED .ino: steer along the fall line on slopes

        self.headImu = Imu("head", 0, +1, 1, +1)
        self.tailImu = Imu("tail", 0, +1, 1, +1)

        self.mode = STOPPED
        self.modeStartMs = 0
        self.escapeLeft = True

        self.kinks = [Kink() for _ in range(MAX_KINKS)]
        self.pitchBaseline = 0.0
        self.slopeArmed = True
        self.lastTriggerMs = 0

        self.irLeftCm = self.IR_MAX_CM
        self.irRightCm = self.IR_MAX_CM
        self.irLeftNow = self.IR_MAX_CM       # low-pass filtered readings, before the peak hold
        self.irRightNow = self.IR_MAX_CM
        self.irPeakHold = True                # NOT IN THE PASTED .ino, see readIr()
        self.wallTurn = True                  # NOT IN THE PASTED .ino, see updateMode() and wallCheck()
        self.turnStartMs = 0                  # when this turn away from the wall began
        self.wallMinCm = self.IR_MAX_CM       # nearest and farthest reading of the nose-up phase so far
        self.wallMaxCm = 0.0
        self.wallN = 0
        self.irBufLeft = [self.IR_MAX_CM] * AVG_N
        self.irBufRight = [self.IR_MAX_CM] * AVG_N
        self.irIdx = 0
        self.irLeftMin = self.IR_MAX_CM       # nearest readings of the last wave period (edge steering)
        self.irRightMin = self.IR_MAX_CM
        self.edgeSteer = True                 # NOT IN THE PASTED .ino, see steerFromEdge()
        self.slopeBends = False               # NOT IN THE PASTED .ino, see updateTerrain()
        self.bendFollow = True                # NOT IN THE PASTED .ino, see updateBodyBend()
        self.bodyBendDeg = 0.0                # commanded total bend, + valley, - crest
        self.steerDeg = 0.0
        self.steerTarget = 0.0
        self.ampScale = 0.0
        self.ampTarget = 0.0
        self.gaitPhase = 0.0          # radians, runs forwards or backwards
        self.gaitDir = +1             # +1 forward, -1 backward

        self.pitchTarget = [0.0] * 4  # P1 P3 P5 P7
        self.pitchCmd = [0.0] * 4
        self.manualDeg = [0.0] * 7
        self.jointDeg = [0.0] * 7
        self.servoDeg = [0.0] * 7     # what the PWM pulse commands, in the joint convention

        self.printTelemetry = False
        self.lastPrintMs = 0
        self.now_ms = 0
        self.out = []                 # everything the firmware would print to Serial

        # setup()
        self.out.append("Snake robot starting")
        for i in range(7):
            self.setJoint(i, 0)       # everything straight
        self.out.append("Stopped. Type g to go, s to stop, p for telemetry.")

    def millis(self):
        return self.now_ms

    # ------------------------------------------------------------ servo output
    def usToDuty(self, us):
        return us * (1 << self.PWM_BITS) // 20000

    def servoWriteUs(self, i, us):
        us = constrain(us, self.SERVO_MIN_US, self.SERVO_MAX_US)
        duty = self.usToDuty(us)
        # the pulse the servo actually sees, turned back into a joint angle
        pulse_us = duty * 20000 / (1 << self.PWM_BITS)
        servo = (pulse_us - self.SERVO_CENTER_US) / self.SERVO_US_PER_DEG
        self.servoDeg[i] = (servo - self.JOINT_TRIM_DEG[i]) / self.JOINT_DIR[i]

    def setJoint(self, i, deg):
        """deg is the joint angle in the robot convention described at the top."""
        deg = constrain(deg, -self.JOINT_LIMIT_DEG, self.JOINT_LIMIT_DEG)
        self.jointDeg[i] = deg
        us = self.SERVO_CENTER_US + (self.JOINT_DIR[i] * deg + self.JOINT_TRIM_DEG[i]) * self.SERVO_US_PER_DEG
        self.servoWriteUs(i, int(math.floor(us + 0.5)))     # lroundf

    # ------------------------------------------------------------ IR sensors
    def sharpToCm(self, mv):
        volts = mv / 1000.0
        if volts < 0.35:
            return self.IR_MAX_CM                 # nothing in range
        cm = 12.08 * math.pow(volts, -1.058)      # curve fit for GP2Y0A41SK0F
        return constrain(cm, self.IR_MIN_CM, self.IR_MAX_CM)

    def readIr(self, dt, left_mv, right_mv):
        a = dt / self.IR_TAU_S
        if a > 1:
            a = 1
        self.irLeftNow += (self.sharpToCm(left_mv) - self.irLeftNow) * a
        self.irRightNow += (self.sharpToCm(right_mv) - self.irRightNow) * a

        # NOT IN THE PASTED .ino: with the vertical wave the nose dips once per period and
        # the beams then hit the floor (or the ramp) at a few cm.  A wall is there for the
        # whole period, the floor only for part of it, so use the farthest reading of the
        # last wave period.
        self.irBufLeft[self.irIdx] = self.irLeftNow
        self.irBufRight[self.irIdx] = self.irRightNow
        self.irIdx = (self.irIdx + 1) % AVG_N
        if self.irPeakHold:
            self.irLeftCm = max(self.irBufLeft)
            self.irRightCm = max(self.irBufRight)
        else:
            self.irLeftCm = self.irLeftNow
            self.irRightCm = self.irRightNow
        # the nearest reading of the period is the ground beside the nose at the dips
        self.irLeftMin = min(self.irBufLeft)
        self.irRightMin = min(self.irBufRight)

    # ------------------------------------------------------------ IMU
    def imuUpdate(self, m, acc):
        if not m.ok:
            return
        if acc is None:                           # imuRead() failed
            return
        m.acc = list(acc)

        # pitch = angle of the forward axis above the horizontal plane
        fwd = m.acc[m.fwdAxis] * m.fwdSign
        o1 = m.acc[(m.fwdAxis + 1) % 3]
        o2 = m.acc[(m.fwdAxis + 2) % 3]
        m.pitch = math.degrees(math.atan2(fwd, math.sqrt(o1 * o1 + o2 * o2)))

        left = m.acc[m.leftAxis] * m.leftSign

        # moving average over one wave period cancels the side to side wobble
        if not m.primed:
            for k in range(AVG_N):
                m.buf[k] = m.pitch
                m.bufFwd[k] = fwd
                m.bufLeft[k] = left
            m.bufSum = m.pitch * AVG_N
            m.sumFwd = fwd * AVG_N
            m.sumLeft = left * AVG_N
            m.primed = True
        m.bufSum += m.pitch - m.buf[m.bufIdx]
        m.buf[m.bufIdx] = m.pitch
        m.sumFwd += fwd - m.bufFwd[m.bufIdx]
        m.bufFwd[m.bufIdx] = fwd
        m.sumLeft += left - m.bufLeft[m.bufIdx]
        m.bufLeft[m.bufIdx] = left
        m.bufIdx = (m.bufIdx + 1) % AVG_N
        m.pitchAvg = m.bufSum / AVG_N
        m.fwdAvg = m.sumFwd / AVG_N
        m.leftAvg = m.sumLeft / AVG_N

    # ------------------------------------------------------------ mode logic
    def setMode(self, m):
        self.mode = m
        self.modeStartMs = self.millis()

    def steerFromIr(self):
        nearest = min(self.irLeftCm, self.irRightCm)
        if nearest > self.STEER_START_CM:
            return 0
        # left closer -> negative -> bend right -> turn away from the left obstacle
        off = self.STEER_GAIN * (self.irLeftCm - self.irRightCm)
        return constrain(off, -self.STEER_MAX_DEG, self.STEER_MAX_DEG)

    def imuOnSlope(self, m):
        """True when the averaged gravity component along the ground is big enough to steer by."""
        if not m.ok or not m.primed:
            return False
        return math.sqrt(m.fwdAvg * m.fwdAvg + m.leftAvg * m.leftAvg) > math.sin(math.radians(self.SLOPE_MIN_DEG))

    def imuSlopeErrorDeg(self, m):
        """Heading error of this end of the robot against the fall line: 0 = straight up
        (or straight down) the slope, + = it points to the left of it."""
        err = math.degrees(math.atan2(-m.leftAvg, m.fwdAvg))     # relative to straight uphill
        if m.fwdAvg < 0:                          # going down: relative to straight downhill
            err += -180.0 if err > 0 else 180.0
        return err

    def onSlope(self):
        return self.imuOnSlope(self.headImu) or self.imuOnSlope(self.tailImu)

    def slopeErrorDeg(self):
        """Mean over the IMUs that are on a slope.  While the nose is over a crest the
        head is level and tells nothing, but the tail is still on the slope behind."""
        total, n = 0.0, 0
        for m in (self.headImu, self.tailImu):
            if self.imuOnSlope(m):
                total += self.imuSlopeErrorDeg(m)
                n += 1
        return total / n if n else 0.0

    def steerFromSlope(self):
        # nose left of the fall line -> negative -> bend right -> back onto the fall line
        off = -self.SLOPE_STEER_GAIN * self.slopeErrorDeg()
        return constrain(off, -self.STEER_MAX_DEG, self.STEER_MAX_DEG)

    def steerFromEdge(self):
        # While the nose hangs in the air over a crest (head level, tail still pitched up)
        # the board falls away in front of it as well, and the beam that points more
        # forwards finds nothing: that is not a side edge, so do not steer by it.
        if (not self.imuOnSlope(self.headImu)
                and self.headImu.pitchAvg - self.tailImu.pitchAvg < -self.EDGE_CREST_DEG):
            return 0
        leftBoard = self.irLeftMin < self.EDGE_CM
        rightBoard = self.irRightMin < self.EDGE_CM
        if leftBoard and not rightBoard:
            return self.EDGE_STEER_DEG            # the board ends on the right -> bend left
        if rightBoard and not leftBoard:
            return -self.EDGE_STEER_DEG
        return 0                                  # board on both sides, or the nose is in the air

    def steerOpen(self):
        """Steering where there are no walls: hold the fall line, keep away from an edge."""
        off = 0.0
        if self.slopeSteer and self.onSlope():
            off += self.steerFromSlope()
        if self.edgeSteer:
            off += self.steerFromEdge()
        return constrain(off, -self.OPEN_STEER_MAX_DEG, self.OPEN_STEER_MAX_DEG)

    def startTurn(self):
        self.setMode(TURNING)
        self.turnStartMs = self.millis()
        self.wallMinCm = self.IR_MAX_CM
        self.wallMaxCm = 0
        self.wallN = 0

    def wallCheck(self):
        """wallTurn (NOT IN THE PASTED .ino): is the wall that the turn goes away from still
        at the nose?  +1 yes, -1 no, 0 cannot tell (yet).
        The peak hold cannot tell: closer than IR_MIN_CM the Sharp folds back and reads
        far, so with the nose against the wall the farthest reading of a period says "no
        wall".  But a folded back reading jumps with every mm the nose moves, and a wall
        at a real distance (or none) reads steadily.  So the readings of one nose-up
        phase of the wave are judged together (nose down, the beams see the floor)."""
        wall = self.irRightNow if self.escapeLeft else self.irLeftNow
        noseUp = not self.gaitVertical or math.sin(self.gaitPhase) > 0.5
        if noseUp:
            self.wallMinCm = min(self.wallMinCm, wall)
            self.wallMaxCm = max(self.wallMaxCm, wall)
            self.wallN += 1
        if self.wallN == 0 or (noseUp and self.wallN < AVG_N):
            return 0                              # still collecting
        seen = 0
        if self.wallMinCm < self.WALL_CLEAR_CM:
            seen = +1
        elif self.wallMaxCm - self.wallMinCm < self.WALL_STEADY_CM:
            seen = -1
        self.wallMinCm = self.IR_MAX_CM
        self.wallMaxCm = 0
        self.wallN = 0
        return seen

    def updateMode(self):
        inMode = self.millis() - self.modeStartMs
        nearest = min(self.irLeftCm, self.irRightCm)

        if self.mode == STOPPED:
            self.ampTarget = 0
            self.steerTarget = 0

        elif self.mode == CRAWL:
            self.gaitDir = +1
            self.ampTarget = 1
            if nearest > self.STEER_START_CM:
                self.steerTarget = self.steerOpen()         # no walls in range: open floor or ramp
            else:
                self.steerTarget = self.steerFromIr()
            if nearest < self.BACKUP_CM:
                self.escapeLeft = self.irLeftCm > self.irRightCm    # more room on the left
                # NOT IN THE PASTED .ino, wallTurn: no backing up.  Backing straightens the
                # body and gives the turn away again; keep crawling and keep turning instead.
                if self.wallTurn:
                    self.startTurn()
                else:
                    self.setMode(BACKING)

        elif self.mode == BACKING:
            self.gaitDir = -1                     # wave runs the other way, robot reverses
            self.steerTarget = 0
            if inMode > self.BACKUP_MS:
                self.startTurn()

        elif self.mode == TURNING:
            self.gaitDir = +1
            self.steerTarget = self.STEER_MAX_DEG if self.escapeLeft else -self.STEER_MAX_DEG
            if self.wallTurn:
                seen = self.wallCheck()
                if self.millis() - self.turnStartMs > self.WALL_STUCK_MS:
                    self.escapeLeft = not self.escapeLeft     # wedged: back up and try the other way
                    self.setMode(BACKING)
                elif seen > 0:
                    self.setMode(TURNING)         # still there: WALL_HOLD_MS starts over
                elif seen < 0 or inMode > self.WALL_HOLD_MS:
                    self.setMode(CRAWL)
            elif inMode > self.TURN_MS:
                self.setMode(CRAWL)

    # ------------------------------------------------------------ slope transitions
    def resetTerrain(self):
        for kk in self.kinks:
            kk.active = False
        for i in range(4):
            self.pitchTarget[i] = 0
        self.pitchBaseline = self.headImu.pitchAvg
        self.slopeArmed = True

    def activeKinks(self):
        return sum(1 for kk in self.kinks if kk.active)

    def startKink(self, bend):
        for kk in self.kinks:
            if not kk.active:
                kk.active = True
                kk.bend = bend
                kk.travelM = 0
                break
        self.slopeArmed = False
        self.lastTriggerMs = self.millis()
        self.out.append("[slope] %s detected, bend %+.1f deg" % ("valley" if bend > 0 else "crest", bend))

    def updateTerrain(self, dt):
        if not self.headImu.ok or not self.headImu.primed:
            return
        # NOT IN THE PASTED .ino: the slope bends can be switched off.  With the vertical
        # wave they misfire: every bend changes the head pitch that the detection watches.
        if not self.slopeBends:
            for kk in self.kinks:
                kk.active = False
            for i in range(4):
                self.pitchTarget[i] = 0
            return
        head = self.headImu.pitchAvg
        movingForward = self.mode in (CRAWL, TURNING)

        # 1. Detect a new kink under the nose
        if self.slopeArmed:
            self.pitchBaseline += (head - self.pitchBaseline) * dt / self.BASELINE_TAU_S
            change = head - self.pitchBaseline
            if movingForward and self.ampScale > 0.9:
                if change > self.KINK_TRIGGER_DEG:
                    self.startKink(+self.VALLEY_BEND_DEG)
                elif change < -self.KINK_TRIGGER_DEG:
                    self.startKink(-self.CREST_BEND_DEG)
        elif self.millis() - self.lastTriggerMs > self.REARM_MS:
            self.pitchBaseline = head     # the average now shows the new slope, measure from here
            self.slopeArmed = True

        # 2. Move every active bend backwards as the robot crawls forward
        for i in range(4):
            self.pitchTarget[i] = 0
        for kk in self.kinks:
            if not kk.active:
                continue
            kk.travelM += self.CRAWL_SPEED_MPS * self.gaitDir * self.ampScale * dt
            if kk.travelM < 0:
                kk.travelM = 0
            k = int(kk.travelM / self.PITCH_SPACING_M)      # which pitch joint is over this kink
            if k < 4:
                self.pitchTarget[k] += kk.bend
            else:
                kk.active = False
                self.out.append("[slope] bend has passed the tail")

    # ------------------------------------------------------------ body bend follows the terrain
    def updateBodyBend(self, dt):
        moving = self.mode != STOPPED and self.ampScale > 0.9
        if (not self.bendFollow or not moving or not self.headImu.primed or not self.tailImu.primed):
            self.bodyBendDeg = approach(self.bodyBendDeg, 0, self.PITCH_RATE_DEG_S * dt)
            return
        measured = self.headImu.pitchAvg - self.tailImu.pitchAvg      # the bend the body really has
        residual = measured - self.bodyBendDeg                         # what the terrain adds to the command
        if abs(residual) < self.BEND_DEADBAND_DEG:
            residual = 0.0
        else:
            residual -= math.copysign(self.BEND_DEADBAND_DEG, residual)
        self.bodyBendDeg += (self.BEND_GAIN_PER_S * residual - self.bodyBendDeg / self.BEND_LEAK_TAU_S) * dt
        self.bodyBendDeg = constrain(self.bodyBendDeg, -self.BEND_MAX_DEG, self.BEND_MAX_DEG)

    # ------------------------------------------------------------ gait and joint output
    def updateGait(self, dt):
        self.ampScale = approach(self.ampScale, self.ampTarget, self.RAMP_PER_S * dt)

        s = dt / self.STEER_TAU_S
        if s > 1:
            s = 1
        self.steerDeg += (self.steerTarget - self.steerDeg) * s

        self.gaitPhase += self.gaitDir * 2 * math.pi * self.gaitFreqHz * dt
        if self.gaitPhase > 2 * math.pi:
            self.gaitPhase -= 2 * math.pi
        if self.gaitPhase < 0:
            self.gaitPhase += 2 * math.pi

        # Yaw: travelling wave from head to tail, plus a steering offset
        # (gaitVertical: no wave here, the yaw joints only steer)
        lag = math.radians(self.gaitPhaseDeg)
        for k in range(3):
            j = YAW_JOINTS[k]
            wave = 0.0 if self.gaitVertical else self.gaitAmpDeg * math.sin(self.gaitPhase - k * lag)
            self.setJoint(j, self.ampScale * (wave + self.steerDeg) + self.manualDeg[j])

        # Pitch: straight, except the joint that sits over a slope kink
        # (gaitVertical: plus the travelling wave, running from tail to head so that
        # the robot moves head first; gaitDir = -1 reverses it like the yaw wave)
        for k in range(4):
            j = PITCH_JOINTS[k]
            wave = self.gaitAmpDeg * math.sin(self.gaitPhase + k * lag) if self.gaitVertical else 0.0
            self.pitchCmd[k] = approach(self.pitchCmd[k], self.pitchTarget[k], self.PITCH_RATE_DEG_S * dt)
            bend = 0.5 * self.bodyBendDeg if k in (1, 2) else 0.0      # P3 and P5 share the body bend
            self.setJoint(j, self.ampScale * wave + self.pitchCmd[k] + bend + self.manualDeg[j])

    # ------------------------------------------------------------ serial commands and telemetry
    def printRawImu(self, m):
        if not m.ok:
            self.out.append("%s IMU: not found" % m.name)
            return
        self.out.append("%s IMU  x %+.2f g   y %+.2f g   z %+.2f g   pitch %+.1f deg   avg fwd %+.3f g  left %+.3f g"
                        % (m.name, m.acc[0], m.acc[1], m.acc[2], m.pitch, m.fwdAvg, m.leftAvg))

    def command(self, line):
        """handleSerial() for one received line."""
        line = line.strip()
        if not line:
            return
        cmd = line[0]
        v = []
        for tok in line[1:].split()[:2]:          # sscanf("%f %f")
            try:
                v.append(float(tok))
            except ValueError:
                break
        n = len(v)
        v1 = v[0] if n >= 1 else 0.0
        v2 = v[1] if n >= 2 else 0.0

        if cmd == "g":
            for i in range(7):
                self.manualDeg[i] = 0
            self.resetTerrain()
            self.setMode(CRAWL)
            self.out.append("GO")
        elif cmd == "s":
            self.setMode(STOPPED)
            self.resetTerrain()
            self.out.append("STOP")
        elif cmd == "p":
            self.printTelemetry = not self.printTelemetry
        elif cmd == "r":
            self.printRawImu(self.headImu)
            self.printRawImu(self.tailImu)
        elif cmd == "j":
            if self.mode != STOPPED:
                self.out.append("Stop first (s)")
            elif n == 2 and 1 <= v1 <= 7:
                i = int(v1) - 1
                self.manualDeg[i] = v2
                self.out.append("%s -> %.1f deg" % (JOINT_NAME[i], v2))
            else:
                self.out.append("Use: j N DEG   (N = 1..7)")
        elif cmd == "a":
            if n >= 1:
                self.gaitAmpDeg = constrain(v1, 0.0, 45.0)
            self.out.append("amplitude %.1f deg" % self.gaitAmpDeg)
        elif cmd == "f":
            if n >= 1:
                self.gaitFreqHz = constrain(v1, 0.05, 2.0)
            self.out.append("frequency %.2f Hz" % self.gaitFreqHz)
        elif cmd == "v":
            self.gaitVertical = not self.gaitVertical
            self.out.append("wave on the %s joints" % ("pitch" if self.gaitVertical else "yaw"))
        elif cmd == "h":
            self.slopeSteer = not self.slopeSteer
            self.out.append("slope steering %s" % ("on" if self.slopeSteer else "off"))
        elif cmd == "e":
            self.edgeSteer = not self.edgeSteer
            self.out.append("edge steering %s" % ("on" if self.edgeSteer else "off"))
        elif cmd == "b":
            self.bendFollow = not self.bendFollow
            self.out.append("body bend follows the terrain: %s" % ("on" if self.bendFollow else "off"))
        elif cmd == "k":
            self.slopeBends = not self.slopeBends
            self.out.append("slope bends (valley / crest kinks) %s" % ("on" if self.slopeBends else "off"))
        elif cmd == "w":
            self.wallTurn = not self.wallTurn
            self.out.append("at a wall: %s" % ("keep turning" if self.wallTurn else "back up, then turn"))
        else:
            self.out.append("Commands: g s p r v h e b k w  j N DEG  a DEG  f HZ")

    def telemetryLine(self):
        line = ("%-7s IR L %5.1f R %5.1f  steer %+5.1f  pitch head %+5.1f tail %+5.1f  base %+5.1f  bends"
                % (MODE_NAME[self.mode], self.irLeftCm, self.irRightCm, self.steerDeg,
                   self.headImu.pitchAvg, self.tailImu.pitchAvg, self.pitchBaseline))
        for k in range(4):
            line += " %s %+4.0f" % (JOINT_NAME[PITCH_JOINTS[k]], self.pitchTarget[k])
        line += "  body %+4.0f  edge L %4.1f R %4.1f" % (self.bodyBendDeg, self.irLeftMin, self.irRightMin)
        return line

    def telemetry(self):
        if not self.printTelemetry:
            return
        if self.millis() - self.lastPrintMs < 200:        # 5 lines per second
            return
        self.lastPrintMs = self.millis()
        self.out.append(self.telemetryLine())

    # ------------------------------------------------------------ loop()
    def tick(self, dt, now_ms, ir_left_mv, ir_right_mv, head_acc, tail_acc):
        """One pass of loop() at LOOP_HZ.  Returns servoDeg (head to tail)."""
        self.now_ms = now_ms
        self.readIr(dt, ir_left_mv, ir_right_mv)
        self.imuUpdate(self.headImu, head_acc)
        self.imuUpdate(self.tailImu, tail_acc)
        self.updateMode()
        self.updateTerrain(dt)
        self.updateBodyBend(dt)
        self.updateGait(dt)
        self.telemetry()
        return self.servoDeg
