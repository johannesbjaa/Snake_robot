/*
  MMA4004 snake robot - main controller
  ------------------------------------------------------------
  Board : ESP32 Dev Module (ESP32-WROOM-32, 38 pin DevKit)
  Core  : Arduino-ESP32 by Espressif (3.x preferred, 2.x also works)

  Body, head to tail:   P1  Y2  P3  Y4  P5  Y6  P7
  The ESP32 sits in Y4 and drives all 7 servos directly with PWM.

  Sensors
    2 x Sharp GP2Y0A41SK0F IR (4 to 30 cm) in the nose, angled about +-30 deg
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
    r          print raw IMU axes once (use it to find the forward axis)
    j N DEG    set joint N (1..7) to DEG degrees, only while stopped
    a DEG      set wave amplitude in degrees
    f HZ       set wave frequency in Hz
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
// 3. GAIT  (lateral undulation on the yaw joints)
// =====================================================================
const int   LOOP_HZ = 50;
const int   LOOP_MS = 1000 / LOOP_HZ;

float gaitAmpDeg   = 30.0;    // wave amplitude
float gaitFreqHz   = 0.5;     // wave frequency
float gaitPhaseDeg = 120.0;   // phase lag between neighbouring yaw joints

const float RAMP_PER_S       = 0.5;    // amplitude ramps 0 -> 100 % in 2 s (softer on the battery)
const float PITCH_RATE_DEG_S = 60.0;   // max speed of pitch joints when the bend moves

// =====================================================================
// 4. STEERING FROM THE IR SENSORS
// =====================================================================
const float IR_MIN_CM      = 4.0;    // below this the Sharp output folds back, mount it 4 cm inside the nose
const float IR_MAX_CM      = 30.0;
const float IR_TAU_S       = 0.15;   // smoothing time constant
const float STEER_START_CM = 25.0;   // start steering when something is closer than this
const float STEER_GAIN     = 1.5;    // deg of yaw offset per cm of left/right difference
const float STEER_MAX_DEG  = 20.0;
const float STEER_TAU_S    = 0.3;
const float BACKUP_CM      = 7.0;    // closer than this: back up and turn away
const unsigned long BACKUP_MS = 2500;
const unsigned long TURN_MS   = 3000;

// =====================================================================
// 5. SLOPE TRANSITIONS  (valley = down then up, crest = up then down)
// =====================================================================
const int   AVG_N            = 100;    // samples in the pitch average. 100 at 50 Hz = 2 s = one wave at 0.5 Hz
const float BASELINE_TAU_S   = 4.0;    // slow reference pitch
const float KINK_TRIGGER_DEG = 6.0;    // nose pitch change that counts as reaching a kink
const float VALLEY_BEND_DEG  = 30.0;   // a + b, set from the real course slopes
const float CREST_BEND_DEG   = 20.0;
const float CRAWL_SPEED_MPS  = 0.03;   // MEASURE THIS on flat ground with the gait you will use
const float PITCH_SPACING_M  = 0.17;   // distance between pitch joints
const unsigned long REARM_MS = 2500;   // after a trigger, wait for the pitch average to flush
const int   MAX_KINKS        = 3;      // bends that can travel down the body at the same time

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
  bool        ok = false;
  float       acc[3] = {0, 0, 0};   // last reading in g
  float       pitch = 0;            // last single sample, deg
  float       pitchAvg = 0;         // averaged over one wave period, deg
  float       buf[AVG_N] = {};
  int         bufIdx = 0;
  float       bufSum = 0;
  bool        primed = false;

  Imu(const char* n, TwoWire* b, uint8_t a, int axis, int sign)
    : name(n), bus(b), addr(a), fwdAxis(axis), fwdSign(sign) {}
};

// Change fwdAxis and fwdSign after testing with the "r" command
Imu headImu("head", &Wire,  MC_ADDR, 0, +1);
Imu tailImu("tail", &Wire1, MC_ADDR, 0, +1);

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
  irLeftCm  += (sharpToCm(readMilliVoltsMedian(IR_LEFT_PIN))  - irLeftCm)  * a;
  irRightCm += (sharpToCm(readMilliVoltsMedian(IR_RIGHT_PIN)) - irRightCm) * a;
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

  // moving average over one wave period cancels the side to side wobble
  if (!m.primed) {
    for (int k = 0; k < AVG_N; k++) m.buf[k] = m.pitch;
    m.bufSum = m.pitch * AVG_N;
    m.primed = true;
  }
  m.bufSum += m.pitch - m.buf[m.bufIdx];
  m.buf[m.bufIdx] = m.pitch;
  m.bufIdx = (m.bufIdx + 1) % AVG_N;
  m.pitchAvg = m.bufSum / AVG_N;
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
      steerTarget = steerFromIr();
      if (nearest < BACKUP_CM) {
        escapeLeft = irLeftCm > irRightCm;   // more room on the left
        setMode(BACKING);
      }
      break;

    case BACKING:
      gaitDir = -1;                          // wave runs the other way, robot reverses
      steerTarget = 0;
      if (inMode > BACKUP_MS) setMode(TURNING);
      break;

    case TURNING:
      gaitDir = +1;
      steerTarget = escapeLeft ? STEER_MAX_DEG : -STEER_MAX_DEG;
      if (inMode > TURN_MS) setMode(CRAWL);
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
// GAIT AND JOINT OUTPUT
// =====================================================================
float approach(float x, float target, float maxStep) {
  if (x < target) return min(x + maxStep, target);
  return max(x - maxStep, target);
}

void updateGait(float dt) {
  ampScale = approach(ampScale, ampTarget, RAMP_PER_S * dt);

  float s = dt / STEER_TAU_S;
  if (s > 1) s = 1;
  steerDeg += (steerTarget - steerDeg) * s;

  gaitPhase += gaitDir * TWO_PI * gaitFreqHz * dt;
  if (gaitPhase > TWO_PI) gaitPhase -= TWO_PI;
  if (gaitPhase < 0)      gaitPhase += TWO_PI;

  // Yaw: travelling wave from head to tail, plus a steering offset
  float lag = gaitPhaseDeg * DEG_TO_RAD;
  for (int k = 0; k < 3; k++) {
    int j = YAW_JOINTS[k];
    float wave = gaitAmpDeg * sinf(gaitPhase - k * lag);
    setJoint(j, ampScale * (wave + steerDeg) + manualDeg[j]);
  }

  // Pitch: straight, except the joint that sits over a slope kink
  for (int k = 0; k < 4; k++) {
    int j = PITCH_JOINTS[k];
    pitchCmd[k] = approach(pitchCmd[k], pitchTarget[k], PITCH_RATE_DEG_S * dt);
    setJoint(j, pitchCmd[k] + manualDeg[j]);
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
  Serial.printf("%s IMU  x %+.2f g   y %+.2f g   z %+.2f g   pitch %+.1f deg\n",
                m.name, m.acc[0], m.acc[1], m.acc[2], m.pitch);
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

    default:
      Serial.println("Commands: g s p r  j N DEG  a DEG  f HZ");
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
  updateGait(dt);
  telemetry();
}
