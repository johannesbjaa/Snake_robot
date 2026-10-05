/*
  MMA4004 snake robot - main controller
  ------------------------------------------------------------
  Board : ESP32 Dev Module (ESP32-WROOM-32, 38 pin DevKit)
  Core  : Arduino-ESP32 by Espressif (3.x preferred, 2.x also works)

  Body, head to tail:   P1  Y2  P3  Y4  P5  Y6  P7
  The ESP32 sits in Y4 and drives all 7 servos directly with PWM.

  Sensors
    2 x Sharp GP2Y0A41SK0F IR (4 to 30 cm) in the nose, angled about +-30 deg
        [SIM] the sensor seats in Nase_modul_IRsensorer.stl point about 56 deg off straight ahead
    2 x MC6470 (MikroE 6DOF IMU 13 Click), head (front of P1) and tail (rear of P7)
        Both boards have the same I2C address, so each one gets its own I2C bus.

  Sign conventions (flip JOINT_DIR for any joint that moves the wrong way)
    Yaw joint   +deg = body bends to the LEFT
    Pitch joint +deg = "valley" bend, the part in front of the joint tilts UP
    IMU pitch   +deg = nose up

  Serial Monitor: 115200 baud, line ending "Newline"
    g          go (start crawling)
    s          stop (all joints ramp back to 0)
    p          toggle telemetry printing
    r          print raw IMU axes once (use it to find the forward and the left axis)
    j N DEG    set joint N (1..7) to DEG degrees, only while stopped
    a DEG      set wave amplitude in degrees
    f HZ       set wave frequency in Hz
    v          [SIM] toggle the wave between the pitch joints and the yaw joints
    h          [SIM] toggle steering along the slope (accelerometer)
    e          [SIM] toggle steering away from the board edge (IR at the nose dips)
    b          [SIM] toggle the body bend that follows the terrain
    k          [SIM] toggle the original slope bends (valley / crest kinks)
    w          [SIM] toggle turning away at a wall without backing up first

  ------------------------------------------------------------
  Changes that came out of the MuJoCo simulation are marked [SIM].  They were
  developed in the simulation only and have NOT run on the real robot.
  firmware.py in the simulation folder is a function by function port of this file.

  Why they are there (the robot has no wheels, so friction is the same in every direction):
    - A wave on the yaw joints alone does not move the robot.  A travelling wave on the
      pitch joints, running tail to head, does (gaitVertical).
    - That wave dips the nose once per period and the IR beams then see the floor.  Walls
      are judged from the farthest reading of a period (irPeakHold), the ground beside
      the nose from the nearest one.
    - On the ramp there are no walls.  The robot steers along the fall line with the head
      accelerometer (slopeSteer) and away from a side where the beams find no board
      during the dips (edgeSteer).
    - Over a crest the terrain bends the body against the pitch servos and the wave stops
      driving.  The bend shows as a difference between head and tail pitch; the firmware
      adopts it as the commanded shape (bendFollow).
    - With the vertical wave the original valley / crest kinks misfire, because every
      bend changes the head pitch that their detection watches (slopeBends, off).
    - At a wall the original backs up with the body straight and only then turns.  The
      robot now keeps crawling and keeps the turn for as long as the wall is at the nose
      (wallTurn).  With the nose against the wall the Sharp is closer than its 4 cm and
      reads far; such a reading jumps about, a real distance is steady (wallCheck).
      If the wall is still there after WALL_STUCK_MS the robot is wedged: it then backs
      up and turns the other way.
*/

#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// =====================================================================
// 1. PINS
// =====================================================================
//                         P1  Y2  P3  Y4  P5  Y6  P7
const int SERVO_PIN[7]  = {13, 14, 16, 17, 18, 19, 23};
const char* JOINT_NAME[7] = {"P1", "Y2", "P3", "Y4", "P5", "Y6", "P7"};

const int IR_LEFT_PIN  = 34;   // ADC1, input only
const int IR_RIGHT_PIN = 35;   // ADC1, input only

const int HEAD_SDA = 21, HEAD_SCL = 22;   // I2C bus 0 (Wire)
const int TAIL_SDA = 25, TAIL_SCL = 26;   // I2C bus 1 (Wire1)
const uint32_t I2C_HZ = 100000;           // slow and safe for long wires through the joints

// =====================================================================
// 2. SERVO CALIBRATION  (Parallax 900-00005)
// =====================================================================
const int   PWM_FREQ_HZ      = 50;     // 20 ms frame
const int   PWM_BITS         = 14;     // 16384 steps per 20 ms, about 1.2 us per step
const int   SERVO_CENTER_US  = 1500;
const float SERVO_US_PER_DEG = 11.1;   // about 1000 us per 90 deg. Check with the "j" command
const int   SERVO_MIN_US     = 600;
const int   SERVO_MAX_US     = 2400;
const float JOINT_LIMIT_DEG  = 60.0;   // keep inside what the printed module allows

// Per joint fixes, in the same order P1 Y2 P3 Y4 P5 Y6 P7
float JOINT_TRIM_DEG[7] = {0, 0, 0, 0, 0, 0, 0};   // horn not perfectly centred
int   JOINT_DIR[7]      = {1, 1, 1, 1, 1, 1, 1};   // -1 if the joint turns the wrong way

const int YAW_JOINTS[3]   = {1, 3, 5};      // Y2 Y4 Y6
const int PITCH_JOINTS[4] = {0, 2, 4, 6};   // P1 P3 P5 P7

// =====================================================================
// 3. GAIT
// =====================================================================
const int   LOOP_HZ = 50;
const int   LOOP_MS = 1000 / LOOP_HZ;

float gaitAmpDeg   = 45.0;    // wave amplitude  [SIM] was 30; 45 with STEER_GAIN 2.5 gets over the course in the simulation
float gaitFreqHz   = 0.5;     // wave frequency
float gaitPhaseDeg = 120.0;   // phase lag between neighbouring joints of the wave

// [SIM] which joints carry the travelling wave
//   true  - pitch joints (vertical wave, runs tail to head).  Works without wheels.
//   false - yaw joints (lateral undulation, the original gait).  Only moves the robot
//           if sliding sideways is harder than sliding forwards.
bool gaitVertical = true;

const float RAMP_PER_S       = 0.5;    // amplitude ramps 0 -> 100 % in 2 s (softer on the battery)
const float PITCH_RATE_DEG_S = 60.0;   // max speed of pitch joints when the bend moves

// =====================================================================
// 4. STEERING FROM THE IR SENSORS
// =====================================================================
const float IR_MIN_CM      = 4.0;    // below this the Sharp output folds back, mount it 4 cm inside the nose
const float IR_MAX_CM      = 30.0;
const float IR_TAU_S       = 0.15;   // smoothing time constant
const float STEER_START_CM = 25.0;   // start steering when something is closer than this
const float STEER_GAIN     = 2.5;    // deg of yaw offset per cm of left/right difference (was 1.5)
const float STEER_MAX_DEG  = 20.0;
const float STEER_TAU_S    = 0.3;
const float BACKUP_CM      = 7.0;    // closer than this: back up and turn away
const unsigned long BACKUP_MS = 2500;
const unsigned long TURN_MS   = 3000;

// [SIM] use the farthest reading of the last wave period for the walls (see readIr)
bool irPeakHold = true;

// [SIM] at a wall: keep crawling and keep turning away, do not back up first (see updateMode
// and wallCheck)
bool wallTurn = true;
const float WALL_CLEAR_CM  = 10.0;   // closer than this, the wall is still at the nose
const float WALL_STEADY_CM = 3.0;    // readings of one nose-up phase this close together: a real distance
const unsigned long WALL_HOLD_MS  = 6000;    // no verdict for this long: end the turn anyway
const unsigned long WALL_STUCK_MS = 30000;   // still at the wall after this long (a turn takes up to 20 s): wedged

// =====================================================================
// 4b. [SIM] STEERING WHERE THERE ARE NO WALLS  (open floor, the ramp)
// =====================================================================
// On a slope gravity shows which way is uphill, so the head accelerometer can hold
// the robot on the fall line.  That gives the heading but not the sideways position.
// For the position: at each nose dip of the vertical wave the IR beams hit the board
// beside the nose.  A side that finds nothing near during a whole period has no board.
const float SLOPE_MIN_DEG    = 6.0;    // flatter than this: the accelerometer cannot tell the direction
const float SLOPE_STEER_GAIN = 0.5;    // deg of yaw offset per deg of heading error against the fall line
const float OPEN_STEER_MAX_DEG = 8.0;  // limit of the steering without walls: a body that is arched by the
                                       // wave and bent sideways as well rolls over onto its side
const float EDGE_CM          = 25.0;   // nearest reading of a period above this: no board on that side
const float EDGE_STEER_DEG   = 6.0;    // yaw offset away from the edge
const float EDGE_CREST_DEG   = 10.0;   // head pitched this much below the tail: over a crest, no edge steering
bool slopeSteer = true;
bool edgeSteer  = true;

// =====================================================================
// 4c. [SIM] BODY BEND FOLLOWS THE TERRAIN
// =====================================================================
// Over a crest or in a valley the terrain bends the body against the pitch servos;
// that takes their torque and the wave stops driving.  The bend shows up as a
// difference between head and tail pitch.  Adopt it as the commanded shape (on the
// two middle pitch joints P3 and P5), and let it relax when the terrain stops forcing it.
const float BEND_GAIN_PER_S   = 0.5;    // how fast the commanded bend follows the measured one
const float BEND_LEAK_TAU_S   = 20.0;   // relaxes back to straight with this time constant
const float BEND_DEADBAND_DEG = 5.0;    // ignore small differences (sensor offsets, the wave)
const float BEND_MAX_DEG      = 40.0;   // total over the body
bool  bendFollow  = true;
float bodyBendDeg = 0;                  // commanded total bend, + valley, - crest

// =====================================================================
// 5. SLOPE TRANSITIONS  (valley = down then up, crest = up then down)
// =====================================================================
const int   AVG_N            = 100;    // samples in the pitch average. 100 at 50 Hz = 2 s = one wave at 0.5 Hz
const float BASELINE_TAU_S   = 4.0;    // slow reference pitch
const float KINK_TRIGGER_DEG = 6.0;    // nose pitch change that counts as reaching a kink
const float VALLEY_BEND_DEG  = 30.0;   // a + b, set from the real course slopes
const float CREST_BEND_DEG   = 20.0;
const float CRAWL_SPEED_MPS  = 0.03;   // MEASURE THIS on flat ground with the gait you will use
                                       // [SIM] about 0.03 with the vertical wave at 40 deg, 0.5 Hz
const float PITCH_SPACING_M  = 0.17;   // distance between pitch joints
const unsigned long REARM_MS = 2500;   // after a trigger, wait for the pitch average to flush
const int   MAX_KINKS        = 3;      // bends that can travel down the body at the same time

// [SIM] off: with the vertical wave these bends misfire (see updateTerrain).  Section 4c replaces them.
bool slopeBends = false;

// =====================================================================
// 6. MC6470 ACCELEROMETER (registers from the MC6470 datasheet and MikroE driver)
// =====================================================================
const uint8_t MC_ADDR       = 0x4C;   // 0x6C if the A5 / ADDR pin is high
const uint8_t MC_REG_MODE   = 0x07;
const uint8_t MC_REG_SRTFR  = 0x08;
const uint8_t MC_REG_XOUT   = 0x0D;   // X, Y, Z, low byte first, 6 bytes
const uint8_t MC_REG_OUTCFG = 0x20;
const uint8_t MC_MODE_STBY  = 0x00;
const uint8_t MC_MODE_WAKE  = 0x01;
const uint8_t MC_RATE_64HZ  = 0x08;
const uint8_t MC_OUT_2G_14B = 0x05;   // +-2 g range, 14 bit
const float   MC_LSB_PER_G  = 4096.0; // 8192 counts per 2 g

struct Imu {
  const char* name;
  TwoWire*    bus;
  uint8_t     addr;
  int         fwdAxis;     // board axis pointing toward the head: 0 = X, 1 = Y, 2 = Z
  int         fwdSign;     // +1 or -1
  int         leftAxis;    // [SIM] board axis pointing to the robot's left (slope steering)
  int         leftSign;    // +1 or -1
  bool        ok = false;
  float       acc[3] = {0, 0, 0};   // last reading in g
  float       pitch = 0;            // last single sample, deg
  float       pitchAvg = 0;         // averaged over one wave period, deg
  float       buf[AVG_N] = {};
  int         bufIdx = 0;
  float       bufSum = 0;
  bool        primed = false;
  // [SIM] forward and leftward acceleration, averaged like the pitch
  float       fwdAvg = 0;           // g
  float       leftAvg = 0;          // g
  float       bufFwd[AVG_N] = {};
  float       bufLeft[AVG_N] = {};
  float       sumFwd = 0;
  float       sumLeft = 0;

  Imu(const char* n, TwoWire* b, uint8_t a, int axis, int sign, int lAxis, int lSign)
    : name(n), bus(b), addr(a), fwdAxis(axis), fwdSign(sign), leftAxis(lAxis), leftSign(lSign) {}
};

// Change the axes and signs after testing with the "r" command:
// tilt the nose up -> "avg fwd" must go positive; roll the robot onto its right side
// (left side up) -> "avg left" must go positive.
Imu headImu("head", &Wire,  MC_ADDR, 0, +1, 1, +1);
Imu tailImu("tail", &Wire1, MC_ADDR, 0, +1, 1, +1);

// =====================================================================
// STATE
// =====================================================================
enum Mode { STOPPED, CRAWL, BACKING, TURNING };
const char* MODE_NAME[] = {"STOPPED", "CRAWL", "BACKING", "TURNING"};
Mode mode = STOPPED;
unsigned long modeStartMs = 0;
bool escapeLeft = true;

// One slope kink that the body is crawling over
struct Kink {
  bool  active;
  float bend;       // deg, + valley, - crest
  float travelM;    // how far the robot has moved since the nose reached it
};
Kink kinks[MAX_KINKS];
float pitchBaseline = 0;
bool  slopeArmed = true;
unsigned long lastTriggerMs = 0;

float irLeftCm = IR_MAX_CM, irRightCm = IR_MAX_CM;
// [SIM] low-pass filtered readings before the peak hold, their last wave period, and its nearest value
float irLeftNow = IR_MAX_CM, irRightNow = IR_MAX_CM;
float irBufLeft[AVG_N], irBufRight[AVG_N];
int   irIdx = 0;
float irLeftMin = IR_MAX_CM, irRightMin = IR_MAX_CM;
// [SIM] wallTurn: when this turn away from the wall began, and the nearest and farthest
// reading of the nose-up phase so far (see wallCheck)
unsigned long turnStartMs = 0;
float wallMinCm = IR_MAX_CM, wallMaxCm = 0;
int   wallN = 0;

float steerDeg = 0, steerTarget = 0;
float ampScale = 0, ampTarget = 0;
float gaitPhase = 0;             // radians, runs forwards or backwards
int   gaitDir = +1;              // +1 forward, -1 backward

float pitchTarget[4] = {0, 0, 0, 0};   // P1 P3 P5 P7
float pitchCmd[4]    = {0, 0, 0, 0};
float manualDeg[7]   = {0, 0, 0, 0, 0, 0, 0};
float jointDeg[7]    = {0, 0, 0, 0, 0, 0, 0};

bool printTelemetry = false;
unsigned long lastTickMs = 0;
unsigned long lastPrintMs = 0;

// =====================================================================
// SERVO OUTPUT
// =====================================================================
uint32_t usToDuty(int us) {
  return (uint32_t)us * (1UL << PWM_BITS) / 20000UL;
}

void servoAttach(int i) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(SERVO_PIN[i], PWM_FREQ_HZ, PWM_BITS);
#else
  ledcSetup(i, PWM_FREQ_HZ, PWM_BITS);
  ledcAttachPin(SERVO_PIN[i], i);
#endif
}

void servoWriteUs(int i, int us) {
  us = constrain(us, SERVO_MIN_US, SERVO_MAX_US);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(SERVO_PIN[i], usToDuty(us));
#else
  ledcWrite(i, usToDuty(us));
#endif
}

// deg is the joint angle in the robot convention described at the top
void setJoint(int i, float deg) {
  deg = constrain(deg, -JOINT_LIMIT_DEG, JOINT_LIMIT_DEG);
  jointDeg[i] = deg;
  float us = SERVO_CENTER_US + (JOINT_DIR[i] * deg + JOINT_TRIM_DEG[i]) * SERVO_US_PER_DEG;
  servoWriteUs(i, (int)lroundf(us));
}

// =====================================================================
// IR SENSORS
// =====================================================================
// Median of 5 quick reads removes single spikes from servo noise
int readMilliVoltsMedian(int pin) {
  int v[5];
  for (int k = 0; k < 5; k++) v[k] = analogReadMilliVolts(pin);
  for (int a = 1; a < 5; a++) {            // small insertion sort
    int x = v[a], b = a - 1;
    while (b >= 0 && v[b] > x) { v[b + 1] = v[b]; b--; }
    v[b + 1] = x;
  }
  return v[2];
}

float sharpToCm(int mv) {
  float volts = mv / 1000.0f;
  if (volts < 0.35f) return IR_MAX_CM;     // nothing in range
  float cm = 12.08f * powf(volts, -1.058f); // curve fit for GP2Y0A41SK0F
  return constrain(cm, IR_MIN_CM, IR_MAX_CM);
}

void readIr(float dt) {
  float a = dt / IR_TAU_S;
  if (a > 1) a = 1;
  irLeftNow  += (sharpToCm(readMilliVoltsMedian(IR_LEFT_PIN))  - irLeftNow)  * a;
  irRightNow += (sharpToCm(readMilliVoltsMedian(IR_RIGHT_PIN)) - irRightNow) * a;

  // [SIM] With the vertical wave the nose dips once per period and the beams then hit
  // the floor (or the ramp) at a few cm.  A wall is there for the whole period, the
  // floor only for part of it, so use the farthest reading of the last wave period.
  irBufLeft[irIdx]  = irLeftNow;
  irBufRight[irIdx] = irRightNow;
  irIdx = (irIdx + 1) % AVG_N;
  float maxL = irBufLeft[0], maxR = irBufRight[0], minL = irBufLeft[0], minR = irBufRight[0];
  for (int k = 1; k < AVG_N; k++) {
    maxL = max(maxL, irBufLeft[k]);   minL = min(minL, irBufLeft[k]);
    maxR = max(maxR, irBufRight[k]);  minR = min(minR, irBufRight[k]);
  }
  if (irPeakHold) {
    irLeftCm  = maxL;
    irRightCm = maxR;
  } else {
    irLeftCm  = irLeftNow;
    irRightCm = irRightNow;
  }
  // the nearest reading of the period is the ground beside the nose at the dips
  irLeftMin  = minL;
  irRightMin = minR;
}

// =====================================================================
// IMU
// =====================================================================
bool mcWrite(Imu& m, uint8_t reg, uint8_t val) {
  m.bus->beginTransmission(m.addr);
  m.bus->write(reg);
  m.bus->write(val);
  return m.bus->endTransmission() == 0;
}

bool imuBegin(Imu& m) {
  bool ok = mcWrite(m, MC_REG_MODE, MC_MODE_STBY);       // registers can only be set in standby
  ok = mcWrite(m, MC_REG_SRTFR, MC_RATE_64HZ) && ok;
  ok = mcWrite(m, MC_REG_OUTCFG, MC_OUT_2G_14B) && ok;
  ok = mcWrite(m, MC_REG_MODE, MC_MODE_WAKE) && ok;
  m.ok = ok;
  return ok;
}

int16_t to14bit(uint8_t lo, uint8_t hi) {
  int32_t raw = ((int32_t)hi << 8 | lo) & 0x3FFF;
  if (raw & 0x2000) raw -= 0x4000;          // sign extend 14 bit
  return (int16_t)raw;
}

// Reads all 3 axes. The MC6470 needs a STOP between reads, endTransmission() sends one.
bool imuRead(Imu& m) {
  m.bus->beginTransmission(m.addr);
  m.bus->write(MC_REG_XOUT);
  if (m.bus->endTransmission() != 0) return false;
  if (m.bus->requestFrom((uint8_t)m.addr, (uint8_t)6) != 6) return false;
  for (int k = 0; k < 3; k++) {
    uint8_t lo = m.bus->read();
    uint8_t hi = m.bus->read();
    m.acc[k] = to14bit(lo, hi) / MC_LSB_PER_G;
  }
  return true;
}

void imuUpdate(Imu& m) {
  if (!m.ok) return;
  if (!imuRead(m)) return;

  // pitch = angle of the forward axis above the horizontal plane
  float fwd = m.acc[m.fwdAxis] * m.fwdSign;
  float o1  = m.acc[(m.fwdAxis + 1) % 3];
  float o2  = m.acc[(m.fwdAxis + 2) % 3];
  m.pitch = atan2f(fwd, sqrtf(o1 * o1 + o2 * o2)) * RAD_TO_DEG;

  float left = m.acc[m.leftAxis] * m.leftSign;   // [SIM]

  // moving average over one wave period cancels the side to side wobble
  if (!m.primed) {
    for (int k = 0; k < AVG_N; k++) {
      m.buf[k] = m.pitch;
      m.bufFwd[k] = fwd;
      m.bufLeft[k] = left;
    }
    m.bufSum = m.pitch * AVG_N;
    m.sumFwd = fwd * AVG_N;
    m.sumLeft = left * AVG_N;
    m.primed = true;
  }
  m.bufSum += m.pitch - m.buf[m.bufIdx];
  m.buf[m.bufIdx] = m.pitch;
  m.sumFwd += fwd - m.bufFwd[m.bufIdx];
  m.bufFwd[m.bufIdx] = fwd;
  m.sumLeft += left - m.bufLeft[m.bufIdx];
  m.bufLeft[m.bufIdx] = left;
  m.bufIdx = (m.bufIdx + 1) % AVG_N;
  m.pitchAvg = m.bufSum / AVG_N;
  m.fwdAvg = m.sumFwd / AVG_N;
  m.leftAvg = m.sumLeft / AVG_N;
}

// =====================================================================
// MODE LOGIC  (stop / crawl / back up / turn away)
// =====================================================================
void setMode(Mode m) {
  mode = m;
  modeStartMs = millis();
}

float steerFromIr() {
  float nearest = min(irLeftCm, irRightCm);
  if (nearest > STEER_START_CM) return 0;
  // left closer -> negative -> bend right -> turn away from the left obstacle
  float off = STEER_GAIN * (irLeftCm - irRightCm);
  return constrain(off, -STEER_MAX_DEG, STEER_MAX_DEG);
}

// [SIM] true when the averaged gravity component along the ground is big enough to steer by
bool imuOnSlope(Imu& m) {
  if (!m.ok || !m.primed) return false;
  return sqrtf(m.fwdAvg * m.fwdAvg + m.leftAvg * m.leftAvg) > sinf(SLOPE_MIN_DEG * DEG_TO_RAD);
}

// [SIM] heading error of this end of the robot against the fall line: 0 = straight up
// (or straight down) the slope, + = it points to the left of it
float imuSlopeErrorDeg(Imu& m) {
  float err = atan2f(-m.leftAvg, m.fwdAvg) * RAD_TO_DEG;      // relative to straight uphill
  if (m.fwdAvg < 0) err += (err > 0) ? -180.0f : 180.0f;      // going down: relative to straight downhill
  return err;
}

// [SIM]
bool onSlope() {
  return imuOnSlope(headImu) || imuOnSlope(tailImu);
}

// [SIM] mean over the IMUs that are on a slope.  While the nose is over a crest the
// head is level and tells nothing, but the tail is still on the slope behind.
float slopeErrorDeg() {
  float total = 0;
  int n = 0;
  if (imuOnSlope(headImu)) { total += imuSlopeErrorDeg(headImu); n++; }
  if (imuOnSlope(tailImu)) { total += imuSlopeErrorDeg(tailImu); n++; }
  return n ? total / n : 0;
}

// [SIM]
float steerFromSlope() {
  // nose left of the fall line -> negative -> bend right -> back onto the fall line
  float off = -SLOPE_STEER_GAIN * slopeErrorDeg();
  return constrain(off, -STEER_MAX_DEG, STEER_MAX_DEG);
}

// [SIM]
float steerFromEdge() {
  // While the nose hangs in the air over a crest (head level, tail still pitched up)
  // the board falls away in front of it as well, and the beam that points more
  // forwards finds nothing: that is not a side edge, so do not steer by it.
  if (!imuOnSlope(headImu) && headImu.pitchAvg - tailImu.pitchAvg < -EDGE_CREST_DEG) return 0;
  bool leftBoard  = irLeftMin  < EDGE_CM;
  bool rightBoard = irRightMin < EDGE_CM;
  if (leftBoard && !rightBoard) return EDGE_STEER_DEG;     // the board ends on the right -> bend left
  if (rightBoard && !leftBoard) return -EDGE_STEER_DEG;
  return 0;                                                // board on both sides, or the nose is in the air
}

// [SIM] steering where there are no walls: hold the fall line, keep away from an edge
float steerOpen() {
  float off = 0;
  if (slopeSteer && onSlope()) off += steerFromSlope();
  if (edgeSteer) off += steerFromEdge();
  return constrain(off, -OPEN_STEER_MAX_DEG, OPEN_STEER_MAX_DEG);
}

// [SIM]
void startTurn() {
  setMode(TURNING);
  turnStartMs = millis();
  wallMinCm = IR_MAX_CM;
  wallMaxCm = 0;
  wallN = 0;
}

// [SIM] wallTurn: is the wall that the turn goes away from still at the nose?
// +1 yes, -1 no, 0 cannot tell (yet).
// The peak hold cannot tell: closer than IR_MIN_CM the Sharp folds back and reads far, so
// with the nose against the wall the farthest reading of a period says "no wall".  But a
// folded back reading jumps with every mm the nose moves, and a wall at a real distance
// (or none) reads steadily.  So the readings of one nose-up phase of the wave are judged
// together (nose down, the beams see the floor).
int wallCheck() {
  float wall = escapeLeft ? irRightNow : irLeftNow;
  bool noseUp = !gaitVertical || sinf(gaitPhase) > 0.5f;
  if (noseUp) {
    wallMinCm = min(wallMinCm, wall);
    wallMaxCm = max(wallMaxCm, wall);
    wallN++;
  }
  if (wallN == 0 || (noseUp && wallN < AVG_N)) return 0;   // still collecting
  int seen = 0;
  if (wallMinCm < WALL_CLEAR_CM)                   seen = +1;
  else if (wallMaxCm - wallMinCm < WALL_STEADY_CM) seen = -1;
  wallMinCm = IR_MAX_CM;
  wallMaxCm = 0;
  wallN = 0;
  return seen;
}

void updateMode() {
  unsigned long inMode = millis() - modeStartMs;
  float nearest = min(irLeftCm, irRightCm);

  switch (mode) {
    case STOPPED:
      ampTarget = 0;
      steerTarget = 0;
      break;

    case CRAWL:
      gaitDir = +1;
      ampTarget = 1;
      if (nearest > STEER_START_CM) steerTarget = steerOpen();   // [SIM] no walls in range: open floor or ramp
      else                          steerTarget = steerFromIr();
      if (nearest < BACKUP_CM) {
        escapeLeft = irLeftCm > irRightCm;   // more room on the left
        // [SIM] wallTurn: no backing up.  Backing straightens the body and gives the
        // turn away again; keep crawling and keep turning instead.
        if (wallTurn) startTurn();
        else          setMode(BACKING);
      }
      break;

    case BACKING:
      gaitDir = -1;                          // wave runs the other way, robot reverses
      steerTarget = 0;
      if (inMode > BACKUP_MS) startTurn();
      break;

    case TURNING:
      gaitDir = +1;
      steerTarget = escapeLeft ? STEER_MAX_DEG : -STEER_MAX_DEG;
      if (wallTurn) {                        // [SIM]
        int seen = wallCheck();
        if (millis() - turnStartMs > WALL_STUCK_MS) {
          escapeLeft = !escapeLeft;          // wedged: back up and try the other way
          setMode(BACKING);
        } else if (seen > 0) {
          setMode(TURNING);                  // still there: WALL_HOLD_MS starts over
        } else if (seen < 0 || inMode > WALL_HOLD_MS) {
          setMode(CRAWL);
        }
      } else if (inMode > TURN_MS) {
        setMode(CRAWL);
      }
      break;
  }
}

// =====================================================================
// SLOPE TRANSITIONS
// =====================================================================
void resetTerrain() {
  for (int i = 0; i < MAX_KINKS; i++) kinks[i].active = false;
  for (int i = 0; i < 4; i++) pitchTarget[i] = 0;
  pitchBaseline = headImu.pitchAvg;
  slopeArmed = true;
}

int activeKinks() {
  int n = 0;
  for (int i = 0; i < MAX_KINKS; i++) if (kinks[i].active) n++;
  return n;
}

void startKink(float bend) {
  for (int i = 0; i < MAX_KINKS; i++) {
    if (!kinks[i].active) {
      kinks[i].active = true;
      kinks[i].bend = bend;
      kinks[i].travelM = 0;
      break;
    }
  }
  slopeArmed = false;
  lastTriggerMs = millis();
  Serial.printf("[slope] %s detected, bend %+.1f deg\n", bend > 0 ? "valley" : "crest", bend);
}

void updateTerrain(float dt) {
  if (!headImu.ok || !headImu.primed) return;

  // [SIM] the slope bends can be switched off.  With the vertical wave they misfire:
  // every bend changes the head pitch that the detection watches.
  if (!slopeBends) {
    for (int i = 0; i < MAX_KINKS; i++) kinks[i].active = false;
    for (int i = 0; i < 4; i++) pitchTarget[i] = 0;
    return;
  }

  float head = headImu.pitchAvg;
  bool movingForward = (mode == CRAWL || mode == TURNING);

  // 1. Detect a new kink under the nose
  if (slopeArmed) {
    pitchBaseline += (head - pitchBaseline) * dt / BASELINE_TAU_S;
    float change = head - pitchBaseline;
    if (movingForward && ampScale > 0.9f) {
      if (change > KINK_TRIGGER_DEG)       startKink(+VALLEY_BEND_DEG);
      else if (change < -KINK_TRIGGER_DEG) startKink(-CREST_BEND_DEG);
    }
  } else if (millis() - lastTriggerMs > REARM_MS) {
    pitchBaseline = head;       // the average now shows the new slope, measure from here
    slopeArmed = true;
  }

  // 2. Move every active bend backwards as the robot crawls forward (backing up moves it forward again)
  for (int i = 0; i < 4; i++) pitchTarget[i] = 0;
  for (int i = 0; i < MAX_KINKS; i++) {
    Kink& kk = kinks[i];
    if (!kk.active) continue;
    kk.travelM += CRAWL_SPEED_MPS * gaitDir * ampScale * dt;
    if (kk.travelM < 0) kk.travelM = 0;
    int k = (int)(kk.travelM / PITCH_SPACING_M);   // which pitch joint is over this kink
    if (k < 4) {
      pitchTarget[k] += kk.bend;
    } else {
      kk.active = false;
      Serial.println("[slope] bend has passed the tail");
    }
  }
}

// =====================================================================
// [SIM] BODY BEND FOLLOWS THE TERRAIN
// =====================================================================
float approach(float x, float target, float maxStep) {
  if (x < target) return min(x + maxStep, target);
  return max(x - maxStep, target);
}

void updateBodyBend(float dt) {
  bool moving = (mode != STOPPED) && ampScale > 0.9f;
  if (!bendFollow || !moving || !headImu.primed || !tailImu.primed) {
    bodyBendDeg = approach(bodyBendDeg, 0, PITCH_RATE_DEG_S * dt);
    return;
  }
  float measured = headImu.pitchAvg - tailImu.pitchAvg;   // the bend the body really has
  float residual = measured - bodyBendDeg;                // what the terrain adds to the command
  if (fabsf(residual) < BEND_DEADBAND_DEG) residual = 0;
  else residual -= copysignf(BEND_DEADBAND_DEG, residual);
  bodyBendDeg += (BEND_GAIN_PER_S * residual - bodyBendDeg / BEND_LEAK_TAU_S) * dt;
  bodyBendDeg = constrain(bodyBendDeg, -BEND_MAX_DEG, BEND_MAX_DEG);
}

// =====================================================================
// GAIT AND JOINT OUTPUT
// =====================================================================
void updateGait(float dt) {
  ampScale = approach(ampScale, ampTarget, RAMP_PER_S * dt);

  float s = dt / STEER_TAU_S;
  if (s > 1) s = 1;
  steerDeg += (steerTarget - steerDeg) * s;

  gaitPhase += gaitDir * TWO_PI * gaitFreqHz * dt;
  if (gaitPhase > TWO_PI) gaitPhase -= TWO_PI;
  if (gaitPhase < 0)      gaitPhase += TWO_PI;

  // Yaw: travelling wave from head to tail, plus a steering offset
  // [SIM] gaitVertical: no wave here, the yaw joints only steer
  float lag = gaitPhaseDeg * DEG_TO_RAD;
  for (int k = 0; k < 3; k++) {
    int j = YAW_JOINTS[k];
    float wave = gaitVertical ? 0.0f : gaitAmpDeg * sinf(gaitPhase - k * lag);
    setJoint(j, ampScale * (wave + steerDeg) + manualDeg[j]);
  }

  // Pitch: straight, except the joint that sits over a slope kink
  // [SIM] gaitVertical: plus the travelling wave, running from tail to head so that the
  // robot moves head first (gaitDir = -1 reverses it like the yaw wave), and P3 and P5
  // share the body bend
  for (int k = 0; k < 4; k++) {
    int j = PITCH_JOINTS[k];
    float wave = gaitVertical ? gaitAmpDeg * sinf(gaitPhase + k * lag) : 0.0f;
    pitchCmd[k] = approach(pitchCmd[k], pitchTarget[k], PITCH_RATE_DEG_S * dt);
    float bend = (k == 1 || k == 2) ? 0.5f * bodyBendDeg : 0.0f;
    setJoint(j, ampScale * wave + pitchCmd[k] + bend + manualDeg[j]);
  }
}

// =====================================================================
// SERIAL COMMANDS AND TELEMETRY
// =====================================================================
void printRawImu(Imu& m) {
  if (!m.ok) {
    Serial.printf("%s IMU: not found\n", m.name);
    return;
  }
  Serial.printf("%s IMU  x %+.2f g   y %+.2f g   z %+.2f g   pitch %+.1f deg   avg fwd %+.3f g  left %+.3f g\n",
                m.name, m.acc[0], m.acc[1], m.acc[2], m.pitch, m.fwdAvg, m.leftAvg);
}

void handleSerial() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  char cmd = line.charAt(0);
  float v1 = 0, v2 = 0;
  int n = sscanf(line.c_str() + 1, "%f %f", &v1, &v2);

  switch (cmd) {
    case 'g':
      for (int i = 0; i < 7; i++) manualDeg[i] = 0;
      resetTerrain();
      setMode(CRAWL);
      Serial.println("GO");
      break;

    case 's':
      setMode(STOPPED);
      resetTerrain();
      Serial.println("STOP");
      break;

    case 'p':
      printTelemetry = !printTelemetry;
      break;

    case 'r':
      printRawImu(headImu);
      printRawImu(tailImu);
      break;

    case 'j':
      if (mode != STOPPED) { Serial.println("Stop first (s)"); break; }
      if (n == 2 && v1 >= 1 && v1 <= 7) {
        int i = (int)v1 - 1;
        manualDeg[i] = v2;
        Serial.printf("%s -> %.1f deg\n", JOINT_NAME[i], v2);
      } else {
        Serial.println("Use: j N DEG   (N = 1..7)");
      }
      break;

    case 'a':
      if (n >= 1) gaitAmpDeg = constrain(v1, 0.0f, 45.0f);
      Serial.printf("amplitude %.1f deg\n", gaitAmpDeg);
      break;

    case 'f':
      if (n >= 1) gaitFreqHz = constrain(v1, 0.05f, 2.0f);
      Serial.printf("frequency %.2f Hz\n", gaitFreqHz);
      break;

    // [SIM] switches
    case 'v':
      gaitVertical = !gaitVertical;
      Serial.printf("wave on the %s joints\n", gaitVertical ? "pitch" : "yaw");
      break;

    case 'h':
      slopeSteer = !slopeSteer;
      Serial.printf("slope steering %s\n", slopeSteer ? "on" : "off");
      break;

    case 'e':
      edgeSteer = !edgeSteer;
      Serial.printf("edge steering %s\n", edgeSteer ? "on" : "off");
      break;

    case 'b':
      bendFollow = !bendFollow;
      Serial.printf("body bend follows the terrain: %s\n", bendFollow ? "on" : "off");
      break;

    case 'k':
      slopeBends = !slopeBends;
      Serial.printf("slope bends (valley / crest kinks) %s\n", slopeBends ? "on" : "off");
      break;

    case 'w':
      wallTurn = !wallTurn;
      Serial.printf("at a wall: %s\n", wallTurn ? "keep turning" : "back up, then turn");
      break;

    default:
      Serial.println("Commands: g s p r v h e b k w  j N DEG  a DEG  f HZ");
  }
}

void telemetry() {
  if (!printTelemetry) return;
  if (millis() - lastPrintMs < 200) return;   // 5 lines per second
  lastPrintMs = millis();
  Serial.printf("%-7s IR L %5.1f R %5.1f  steer %+5.1f  pitch head %+5.1f tail %+5.1f  base %+5.1f  bends",
                MODE_NAME[mode], irLeftCm, irRightCm, steerDeg,
                headImu.pitchAvg, tailImu.pitchAvg, pitchBaseline);
  for (int k = 0; k < 4; k++) Serial.printf(" %s %+4.0f", JOINT_NAME[PITCH_JOINTS[k]], pitchTarget[k]);
  Serial.printf("  body %+4.0f  edge L %4.1f R %4.1f", bodyBendDeg, irLeftMin, irRightMin);   // [SIM]
  Serial.println();
}

// =====================================================================
// SETUP AND LOOP
// =====================================================================
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);            // don't block if the line ending is missing
  delay(300);
  Serial.println("\nSnake robot starting");

  for (int i = 0; i < 7; i++) {
    servoAttach(i);
    setJoint(i, 0);                 // everything straight, safe for fitting the horns
  }
  for (int k = 0; k < AVG_N; k++) { // [SIM] nothing seen yet
    irBufLeft[k] = IR_MAX_CM;
    irBufRight[k] = IR_MAX_CM;
  }

  analogReadResolution(12);
  pinMode(IR_LEFT_PIN, INPUT);
  pinMode(IR_RIGHT_PIN, INPUT);

  Wire.begin(HEAD_SDA, HEAD_SCL, I2C_HZ);
  Wire1.begin(TAIL_SDA, TAIL_SCL, I2C_HZ);
  Serial.printf("Head IMU: %s\n", imuBegin(headImu) ? "OK" : "NOT FOUND");
  Serial.printf("Tail IMU: %s\n", imuBegin(tailImu) ? "OK" : "NOT FOUND");
  delay(50);

  Serial.println("Stopped. Type g to go, s to stop, p for telemetry.");
  lastTickMs = millis();
}

void loop() {
  handleSerial();

  unsigned long now = millis();
  if (now - lastTickMs < (unsigned long)LOOP_MS) return;
  float dt = (now - lastTickMs) / 1000.0f;
  lastTickMs = now;

  readIr(dt);
  imuUpdate(headImu);
  imuUpdate(tailImu);
  updateMode();
  updateTerrain(dt);
  updateBodyBend(dt);
  updateGait(dt);
  telemetry();
}
