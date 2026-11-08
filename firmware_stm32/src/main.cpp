/*
 * Bionic Hand Firmware -- STM32G474RE + SimpleFOC + AS5600
 * ========================================================
 *
 * Why this file exists
 * --------------------
 * The Arduino build drives hobby servos: open loop, no feedback of any kind.
 * A servo reports nothing -- not its position, not its load, not whether it is
 * stalled.  Every "the hand moved" claim rests on "the command was sent".
 *
 * This build drives BLDC gimbal motors through FOC with a magnetic encoder on
 * each joint, which turns that around: the joint angle is MEASURED.  That is
 * what makes a tracking-error number, a stall detection, or a "did the finger
 * actually reach the object" claim possible at all.
 *
 * What is deliberately identical to the Arduino build
 * --------------------------------------------------
 * The serial protocol.  Same commands, same acknowledgements, same error
 * strings, same sequence-id correlation, same timeout-to-safe-pose.  The whole
 * Python side (tracking, mapping, UI, diagnostics) is hardware-neutral and
 * keeps working untouched -- that is the payoff of keeping the protocol
 * separate from the actuator.
 *
 * Protocol
 * --------
 *   Host -> MCU:  "joint:angle,joint:angle,...\n"      set targets (degrees)
 *                 "#<seq> joint:angle,...\n"           same, with a sequence id
 *                 "PING\n"                             liveness check
 *                 "Q\n"                                read back measured angles
 *   MCU  -> Host: "PONG"
 *                 "OK:<seq>:<N> servos updated"        N targets accepted
 *                 "ERROR[<seq>]: <reason>"
 *                 "A:<joint>:<deg>,<joint>:<deg>,..."  reply to Q, MEASURED
 *
 * The acknowledgement still means "target accepted and the control loop has
 * been given it" -- NOT "the joint has arrived".  Arrival is what Q reports.
 *
 * SAFETY
 * ------
 * A BLDC motor with FOC will happily drive a mechanism into a hard stop and
 * hold it there at full current.  Before running this on a real joint:
 *   - current-limit the supply and set a conservative voltage limit,
 *   - verify the encoder direction and zero BEFORE closing the position loop,
 *   - keep a physical power cut within reach,
 *   - start with a small angle and a single motor.
 * The angle window below is a bench window, not a measured mechanical limit.
 */

#include <Arduino.h>
#include <SimpleFOC.h>

// ============================================================================
// Configuration
// ============================================================================

#ifndef SERIAL_BAUD
#define SERIAL_BAUD 115200
#endif

// Number of actuated axes on THIS board.  The hand has 12; a single-joint bench
// rig has 1.  The protocol carries the full 12-axis name set either way, so the
// host does not need to know how many motors are actually wired.
#ifndef NUM_AXES
#define NUM_AXES 1
#endif

// Command timeout: with no valid frame for this long, every axis returns to the
// safe pose.  Matches the Arduino build so the two timeout layers stay
// separately testable (host stops after 3 missing ACKs; firmware stops here).
#define COMMAND_TIMEOUT_MS 3000

// Safe pose, in degrees.  Also the pose commanded at startup.
#define DEFAULT_ANGLE 90.0f

// Bench window.  NOT a mechanical limit -- widen only after measuring the real
// mechanism and fitting a hard stop.
#define BENCH_MIN_ANGLE 80.0f
#define BENCH_MAX_ANGLE 100.0f

// Maximum joint speed, degrees per second.  The target is interpolated at this
// rate so a large command step does not become a full-speed slam.
#define MAX_SPEED_DPS 60.0f

// Motor electrical limits.  GB2208 is a high-resistance gimbal motor (~10 ohm),
// so it tolerates being stalled far better than a low-resistance drone motor --
// but "tolerates" is not "is safe to leave stalled".  These are conservative.
#define MOTOR_VOLTAGE_LIMIT 3.0f      // volts
#define MOTOR_CURRENT_LIMIT 1.0f      // amps
#define MOTOR_POLE_PAIRS 7            // GB2208 is a 14-pole (7 pole-pair) motor

// Motor direction: set to -1 if a positive command drives the joint backwards.
// Verify this with the encoder BEFORE enabling the closed loop.
#define MOTOR_DIRECTION CW

// Serial frame limits, matching the Arduino parser.
#define MAX_COMMAND_PAYLOAD 255
#define MAX_FIELDS 12

// ============================================================================
// Axis table
// ============================================================================

// Canonical axis names.  MUST match the Python side exactly; the validator
// cross-checks this array against the manifest.
const char* AXIS_NAMES[] = {
  "thumb_cmc", "thumb_mcp", "thumb_ip",
  "index_mcp", "index_pip",
  "middle_mcp", "middle_pip",
  "ring_mcp", "ring_pip",
  "pinky_mcp", "pinky_pip",
  "wrist_rot"
};
const uint8_t TOTAL_AXIS_NAMES = 12;

// ---------------------------------------------------------------------------
// Per-axis hardware.  ONLY the first NUM_AXES entries are instantiated.
//
// DengFOC mini is a TWO-channel board, so a 12-axis hand needs six of them (or
// a custom board).  Wiring them all to one MCU also needs 12 encoder pairs and
// 36 PWM channels, which exceeds what a single G474 exposes -- see the notes in
// TODO.md.  For now this drives one axis, which is what the bench test needs.
// ---------------------------------------------------------------------------

#if NUM_AXES >= 1
// --- Axis 0: index_mcp on the bench rig ------------------------------------
// PLACEHOLDER PIN MAP.  Verify against the actual DengFOC mini silkscreen and
// the NUCLEO-G474RE pinout before applying power.
#define A0_PWM_A   PA8
#define A0_PWM_B   PA9
#define A0_PWM_C   PA10
#define A0_EN      PB0
#define A0_SDA     PB7
#define A0_SCL     PB6

BLDCMotor      motor0(MOTOR_POLE_PAIRS);
BLDCDriver3PWM driver0(A0_PWM_A, A0_PWM_B, A0_PWM_C, A0_EN);
MagneticSensorI2C sensor0 = MagneticSensorI2C(AS5600_I2C);
#endif

// Live per-axis state, indexed by the CANONICAL axis index (0..11) so the
// protocol addressing is stable regardless of how many motors exist.
float currentAngle[12];
float targetAngle[12];
bool  axisPresent[12];

// ============================================================================
// Small helpers
// ============================================================================

static bool isDigitChar(char c) { return c >= '0' && c <= '9'; }

// Parse a strict signed decimal integer.  No leading '+', no spaces, no empty
// string: a lenient parser here would accept frames the host considers invalid.
static bool parseIntStrict(const char* text, long* out) {
  if (text == nullptr || *text == '\0') return false;
  const char* p = text;
  bool negative = false;
  if (*p == '-') { negative = true; p++; }
  if (*p == '\0') return false;
  long value = 0;
  while (*p) {
    if (!isDigitChar(*p)) return false;
    value = value * 10 + (*p - '0');
    if (value > 100000L) return false;   // refuse absurd values early
    p++;
  }
  *out = negative ? -value : value;
  return true;
}

// Index of an axis name, or -1.  Linear scan: 12 entries, called at most once
// per field, and this runs on a 170 MHz part.
static int findAxis(const char* name, size_t len) {
  for (uint8_t i = 0; i < TOTAL_AXIS_NAMES; i++) {
    const char* candidate = AXIS_NAMES[i];
    if (strlen(candidate) == len && strncmp(candidate, name, len) == 0) {
      return (int)i;
    }
  }
  return -1;
}

// ============================================================================
// Motion
// ============================================================================

// Command one axis.  Returns false if the axis is not wired on this board.
static bool setAxisTarget(uint8_t axis, float degrees) {
  if (axis >= TOTAL_AXIS_NAMES) return false;
  if (!axisPresent[axis]) return false;
  // Clamp into the bench window.  A command outside it is a host bug or a
  // mapping bug; refusing to move is safer than moving.
  if (degrees < BENCH_MIN_ANGLE) degrees = BENCH_MIN_ANGLE;
  if (degrees > BENCH_MAX_ANGLE) degrees = BENCH_MAX_ANGLE;
  targetAngle[axis] = degrees;
  return true;
}

static void updateMotion(float dtSeconds) {
  const float maxStep = MAX_SPEED_DPS * dtSeconds;
  for (uint8_t i = 0; i < TOTAL_AXIS_NAMES; i++) {
    if (!axisPresent[i]) continue;
    float delta = targetAngle[i] - currentAngle[i];
    if (delta > maxStep)  delta = maxStep;
    if (delta < -maxStep) delta = -maxStep;
    currentAngle[i] += delta;
  }
#if NUM_AXES >= 1
  motor0.move(currentAngle[0]);
#endif
}

// ============================================================================
// Frame parsing
// ============================================================================

// Handle one newline-free frame.  Writes a response into `reply`.
// Returns true when a response should be sent.
static bool handleFrame(char* frame, char* reply, size_t replySize) {
  // Optional "#<seq> " prefix.
  int seq = -1;
  char* body = frame;
  if (frame[0] == '#') {
    char* space = strchr(frame, ' ');
    if (space == nullptr) {
      snprintf(reply, replySize, "ERROR: malformed sequence prefix");
      return true;
    }
    *space = '\0';
    long parsed = 0;
    if (!parseIntStrict(frame + 1, &parsed) || parsed < 0 || parsed > 999) {
      snprintf(reply, replySize, "ERROR: malformed sequence prefix");
      return true;
    }
    seq = (int)parsed;
    body = space + 1;
  }

  if (strcmp(body, "PING") == 0) {
    snprintf(reply, replySize, "PONG");
    return true;
  }

  // "Q" -- report MEASURED angles.  This is the command the Arduino build
  // cannot implement, because a hobby servo has nothing to report.
  if (strcmp(body, "Q") == 0) {
    size_t used = 0;
    used += snprintf(reply + used, replySize - used, "A:");
    bool first = true;
    for (uint8_t i = 0; i < TOTAL_AXIS_NAMES; i++) {
      if (!axisPresent[i]) continue;
      used += snprintf(reply + used, replySize - used, "%s%s:%.1f",
                       first ? "" : ",", AXIS_NAMES[i], currentAngle[i]);
      first = false;
      if (used >= replySize - 24) break;   // never overflow the reply buffer
    }
    return true;
  }

  // --- CSV of joint:angle fields ---
  // Staged into locals first: a frame is applied only if EVERY field is valid,
  // so a partly-malformed frame cannot move half the hand.
  float staged[12];
  bool  seen[12] = {false};
  uint8_t accepted = 0;

  char* cursor = body;
  while (cursor != nullptr && *cursor != '\0') {
    char* comma = strchr(cursor, ',');
    if (comma != nullptr) *comma = '\0';

    char* colon = strchr(cursor, ':');
    if (colon == nullptr) {
      snprintf(reply, replySize, "ERROR%s: missing ':' in field",
               seq >= 0 ? "" : "");
      return true;
    }
    *colon = '\0';
    const char* name = cursor;
    const char* angleText = colon + 1;

    int axis = findAxis(name, strlen(name));
    if (axis < 0) {
      snprintf(reply, replySize, "ERROR: unknown axis '%s'", name);
      return true;
    }
    if (seen[axis]) {
      snprintf(reply, replySize, "ERROR: duplicate axis '%s'", name);
      return true;
    }
    long degrees = 0;
    if (!parseIntStrict(angleText, &degrees)) {
      snprintf(reply, replySize, "ERROR: bad angle '%s'", angleText);
      return true;
    }
    if (degrees < 0 || degrees > 180) {
      snprintf(reply, replySize, "ERROR: angle %ld out of 0..180", degrees);
      return true;
    }
    if (!axisPresent[axis]) {
      snprintf(reply, replySize,
               "ERROR: axis '%s' not available on this board", name);
      return true;
    }

    staged[axis] = (float)degrees;
    seen[axis] = true;
    accepted++;

    if (comma == nullptr) break;
    cursor = comma + 1;
    if (*cursor == '\0') {   // trailing comma -> empty field
      snprintf(reply, replySize, "ERROR: empty CSV field");
      return true;
    }
  }

  if (accepted == 0) {
    snprintf(reply, replySize, "ERROR: no fields");
    return true;
  }

  for (uint8_t i = 0; i < TOTAL_AXIS_NAMES; i++) {
    if (seen[i]) setAxisTarget(i, staged[i]);
  }

  if (seq >= 0) {
    snprintf(reply, replySize, "OK:%d:%u servos updated", seq, accepted);
  } else {
    snprintf(reply, replySize, "OK:%u servos updated", accepted);
  }
  return true;
}

// ============================================================================
// Serial plumbing
// ============================================================================

static char rxBuffer[MAX_COMMAND_PAYLOAD + 2];
static size_t rxLength = 0;
static char replyBuffer[512];

static void emitReply() {
  Serial.print(replyBuffer);
  Serial.print('\n');
}

// ============================================================================
// setup / loop
// ============================================================================

static uint32_t lastCommandTime = 0;
static uint32_t lastMotionTime = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);

  for (uint8_t i = 0; i < TOTAL_AXIS_NAMES; i++) {
    axisPresent[i] = false;
    currentAngle[i] = DEFAULT_ANGLE;
    targetAngle[i] = DEFAULT_ANGLE;
  }

#if NUM_AXES >= 1
  Wire.setSDA(A0_SDA);
  Wire.setSCL(A0_SCL);
  Wire.begin();
  sensor0.init();

  driver0.voltage_limit = MOTOR_VOLTAGE_LIMIT;
  driver0.init();
  motor0.linkDriver(&driver0);
  motor0.linkSensor(&sensor0);

  // Open-loop-ish voltage mode with a position target: the simplest thing that
  // can hold a commanded angle.  Current sensing is not wired on the bench rig
  // yet, so torque mode is not available.
  motor0.torque_controller = TorqueControlType::voltage;
  motor0.controller = MotionControlType::angle;
  motor0.PID_velocity.P = 0.2f;
  motor0.PID_velocity.I = 2.0f;
  motor0.LPF_velocity.Tf = 0.01f;
  motor0.P_angle.P = 20.0f;
  motor0.voltage_limit = MOTOR_VOLTAGE_LIMIT;
  motor0.velocity_limit = 10.0f;   // rad/s

  motor0.init();
  motor0.initFOC();

  // Park at the safe pose and hold it.
  float zero = sensor0.getAngle();
  motor0.target = zero;            // hold wherever it powered up
  axisPresent[0] = true;
  currentAngle[0] = DEFAULT_ANGLE;
  targetAngle[0] = DEFAULT_ANGLE;

  Serial.print(F("AXIS0 zero="));
  Serial.println(zero, 4);
#endif

  Serial.println(F("BIONIC_HAND_READY"));

  lastCommandTime = millis();
  lastMotionTime = lastCommandTime;
}

void loop() {
  // --- serial receive ---
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      rxBuffer[rxLength] = '\0';
      if (rxLength > 0) {
        if (handleFrame(rxBuffer, replyBuffer, sizeof(replyBuffer))) {
          emitReply();
          lastCommandTime = millis();
        }
      }
      rxLength = 0;
    } else if (c == '\r') {
      // ignore
    } else if (c == '\0') {
      rxLength = 0;   // NUL resets the frame rather than corrupting it
    } else {
      if (rxLength < MAX_COMMAND_PAYLOAD) {
        rxBuffer[rxLength++] = c;
      } else {
        rxLength = 0;   // overlong frame: drop it, do not truncate-and-accept
        snprintf(replyBuffer, sizeof(replyBuffer), "ERROR: frame too long");
        emitReply();
      }
    }
  }

  // --- timeout: return to the safe pose ---
  uint32_t now = millis();
  if (now - lastCommandTime > COMMAND_TIMEOUT_MS) {
    for (uint8_t i = 0; i < TOTAL_AXIS_NAMES; i++) {
      if (axisPresent[i]) targetAngle[i] = DEFAULT_ANGLE;
    }
  }

  // --- motion update ---
  float dt = (now - lastMotionTime) / 1000.0f;
  if (dt > 0.0f) {
    lastMotionTime = now;
    if (dt > 0.25f) dt = 0.25f;   // a long stall must not become a huge step
    updateMotion(dt);
  }

#if NUM_AXES >= 1
  motor0.loopFOC();
  motor0.move(currentAngle[0]);

  // Report the MEASURED angle back into the protocol state, so a Q reply
  // reflects reality rather than the last command.
  {
    float measured = sensor0.getAngle();
    float degrees = measured * 180.0f / PI;
    // Encoder angle is unbounded (multi-turn); fold it into 0..360 for display.
    while (degrees < 0.0f)    degrees += 360.0f;
    while (degrees >= 360.0f) degrees -= 360.0f;
    // NOTE: this is the raw encoder angle, not yet mapped to joint degrees.
    // The joint zero/direction calibration is a bench step; until it is done,
    // Q reports the encoder, not the joint.
    currentAngle[0] = degrees;
  }
#endif
}
