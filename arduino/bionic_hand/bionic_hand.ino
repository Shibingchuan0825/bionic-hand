/**
 * Bionic Dexterous Hand - Arduino Firmware
 * =========================================
 *
 * Receives joint angle commands via serial from the PC (Python/MediaPipe)
 * and controls multiple servo motors for a bionic prosthetic hand.
 *
 * Protocol (ASCII text, newline-terminated):
 *   COMMAND FORMAT:  "joint_name:angle,joint_name:angle,...\n"
 *   EXAMPLE:         "index_mcp:90,index_pip:170,middle_mcp:85,...\n"
 *
 *   A command may carry an optional sequence id: "#12 index_mcp:90\n".
 *   The id (0-999) is echoed as "OK:12:1 servos updated", which lets the host
 *   match an acknowledgment to the exact frame it sent.  A partial frame is
 *   accepted: it updates only the axes it names.
 *
 *   RESPONSE FORMAT: "OK:N servos updated\n" (N = recognized fields)
 *
 *   "OK:N" means the frame was valid, every named axis is available on this
 *   board, and an output attempt was issued to each of them.  The Adafruit
 *   driver's setPWM() returns void, so the firmware verifies the I2C device is
 *   present immediately before writing; "OK" does not prove the servo shaft
 *   moved.  It is an acknowledgment that the command was accepted and driven,
 *   not a position measurement.
 *
 *   ERROR RESPONSES:
 *     "ERROR: invalid command"                    malformed or unknown field
 *     "ERROR: command too long"                   frame exceeded the buffer
 *     "ERROR: axis not available on this board"   named axis has no output
 *     "ERROR: driver not ready"                   PCA9685 / driver unreachable
 *     "ERROR: actuator update failed"             output attempt failed
 *
 *   PING:            "PING\n" → responds "PONG\n"
 *
 * Hardware Requirements:
 *   - Arduino Uno/Mega/Nano or compatible
 *   - PCA9685 16-channel PWM driver (recommended for 12+ servos)
 *     OR direct PWM pins (max ~6 servos on Uno)
 *   - External 5-6V power supply for servos (do NOT power from Arduino 5V pin!)
 *
 * Connections (using PCA9685 via I2C):
 *   Arduino    PCA9685
 *   -------    -------
 *   SDA (A4) → SDA
 *   SCL (A5) → SCL
 *   VCC      → VCC (3.3V or 5V)
 *   GND      → GND
 *
 *   PCA9685   Servos
 *   -------   ------
 *   PWM 0-11 → 12x SG90/MG996R servo signal wires
 *   External 5-6V power → Servo V+ and PCA9685 V+
 *   GND → Servo GND
 *
 * Direct PWM mode (no PCA9685):
 *   The Servo library does NOT use the hardware PWM pins.  It bit-bangs each
 *   pulse from a Timer1 interrupt and only calls pinMode(pin, OUTPUT), so any
 *   digital pin works and the limit is the library's channel count, not the
 *   number of PWM-capable pins:
 *     Uno/Nano (ATmega328P): 1 timer x 12 = MAX_SERVOS 12  -> all 12 axes fit
 *     Mega (ATmega2560):     4 timers x 12 = MAX_SERVOS 48
 *   Avoid pins 0/1 (USB serial).  Note that pins 9/10 lose analogWrite() while
 *   the library holds Timer1; they still work fine as servo outputs.
 */

// Choose servo driver mode before conditional includes:
//   0 = Direct PWM (Servo library, all 12 axes on Uno)
//   1 = PCA9685 I2C PWM driver (16 channels, frees the Uno's timers)
#define DRIVER_MODE 0

#if DRIVER_MODE != 0 && DRIVER_MODE != 1
  #error "DRIVER_MODE must be 0 (direct PWM) or 1 (PCA9685 I2C)."
#endif

#include <stdlib.h>
#include <errno.h>
#include <limits.h>

#if DRIVER_MODE == 1
  #include <Wire.h>
  #include <Adafruit_PWMServoDriver.h>
#else
  #include <Servo.h>
#endif

// ============================================================================
// Configuration
// ============================================================================

// Serial baud rate (must match Python config)
#define SERIAL_BAUD 115200

// Servo pulse range (microseconds) - adjust for your servo specs
#define SERVO_MIN_US  500
#define SERVO_MAX_US  2500
#define SERVO_FREQ    50    // 50 Hz for standard analog servos

// PCA9685 I2C address
#define PCA9685_ADDR 0x40

// Number of servos
#define NUM_SERVOS 12

// ============================================================================
// Servo Configuration
// ============================================================================

// Servo name definitions (must match Python side!)
const char* SERVO_NAMES[NUM_SERVOS] = {
  "thumb_cmc", "thumb_mcp", "thumb_ip",
  "index_mcp", "index_pip",
  "middle_mcp", "middle_pip",
  "ring_mcp", "ring_pip",
  "pinky_mcp", "pinky_pip",
  "wrist_rot"
};

// PCA9685 channel assignments (or Arduino pin for direct mode)
#if DRIVER_MODE == 1
  // PCA9685 channel per servo
  const uint8_t SERVO_CHANNELS[NUM_SERVOS] = {0,1,2, 3,4, 5,6, 7,8, 9,10, 11};
#else
  // Direct PWM: one digital pin per axis.  These pins do NOT need to be
  // hardware-PWM capable -- see the header comment.  Pins 0/1 are avoided
  // because they carry the USB serial link.
  //
  // Pin 13 also drives the on-board LED, which will flicker with its servo
  // pulse.  That is cosmetic only; move that axis to A0 (digital 14) if it is
  // distracting.
  const int16_t SERVO_PINS[NUM_SERVOS] = {2,3,4, 5,6, 7,8, 9,10, 11,12, 13};
#endif

// Default angle (relaxed open hand position)
#define DEFAULT_ANGLE 90

// Smoothing: maximum joint speed in degrees per second.  A fixed step per loop
// made the real speed depend on loop timing, so an I2C retry or a serial burst
// could silently change the motion profile.
#define MAX_SPEED_DPS 180.0f

// Longest interval that may be integrated in one step.  A stall (blocked I2C,
// host pause, boot delay) must not be turned into one large commanded jump.
#define MAX_INTERPOLATION_DT 0.1f

// Timeout: if no command received for this long, return to default pose (ms)
#define COMMAND_TIMEOUT_MS 3000

// ============================================================================
// Globals
// ============================================================================

#if DRIVER_MODE == 1
  Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);
#endif

#if DRIVER_MODE == 0
  Servo servos[NUM_SERVOS];
#endif

bool driverReady = false;

// Per-axis availability.  In direct-PWM mode an axis whose pin is absent or
// whose attach() failed must be rejected at parse time instead of silently
// accepting a frame that can never reach an actuator.
bool servoAttached[NUM_SERVOS];

// Servo state
float currentAngles[NUM_SERVOS];
float targetAngles[NUM_SERVOS];
unsigned long lastCommandTime = 0;
unsigned long lastInterpolationTime = 0;

// Forward declarations keep the sketch valid for non-Arduino C++ builds too.
// The Arduino IDE auto-generates prototypes for .ino files, but a direct
// avr-g++ compile does not, so every call made before its definition is
// declared here.
bool refreshDriverHealth();
bool servoAvailable(uint8_t index);
bool parseAngleText(const char* text, int* angleOut);
bool setServoAngle(uint8_t index, float angle);
bool interpolateServos();
void processSerial();
void parseCommand(const char* command);
void printError(int seq, const __FlashStringHelper* message);
int8_t findServoIndex(const char* name);
void checkTimeout();
void printServoStatus();

// ============================================================================
// Driver health and channel helpers
// ============================================================================

bool refreshDriverHealth() {
#if DRIVER_MODE == 1
  Wire.beginTransmission(PCA9685_ADDR);
  uint8_t status = Wire.endTransmission();
  if (status != 0) {
    driverReady = false;
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
      servoAttached[i] = false;
    }
    return false;
  }

  if (!driverReady) {
    pwm.begin();
    pwm.setPWMFreq(SERVO_FREQ);
  }
  driverReady = true;
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    servoAttached[i] = true;
  }
#endif
  return driverReady;
}

bool servoAvailable(uint8_t index) {
  if (index >= NUM_SERVOS || !driverReady) {
    return false;
  }
  return servoAttached[index];
}

bool parseAngleText(const char* text, int* angleOut) {
  if (text == NULL || *text == '\0') {
    return false;
  }

  const char* cursor = text;
  if (*cursor == '+' || *cursor == '-') {
    cursor++;
  }
  if (*cursor < '0' || *cursor > '9') {
    return false;
  }
  for (; *cursor != '\0'; cursor++) {
    if (*cursor < '0' || *cursor > '9') {
      return false;
    }
  }

  errno = 0;
  char* end = NULL;
  long value = strtol(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' || value < 0 || value > 180) {
    return false;
  }

  *angleOut = (int)value;
  return true;
}

// ============================================================================
// Setup
// ============================================================================

void setup() {
  Serial.begin(SERIAL_BAUD);

  // Wait for serial to be ready (important for Leonardo/Micro)
  while (!Serial && millis() < 3000);

  Serial.println(F("BIONIC_HAND_BOOTING"));

  // Initialize servos
#if DRIVER_MODE == 1
  Wire.begin();
  driverReady = refreshDriverHealth();
  if (driverReady) {
    Serial.print(F("PCA9685 initialized at "));
    Serial.print(SERVO_FREQ);
    Serial.println(F(" Hz"));
  } else {
    Serial.println(F("ERROR: PCA9685 not detected"));
  }
#else
  // Initialize direct servo objects.  Unsupported axes remain disconnected.
  // On the Uno only 6 of the 12 axes have a PWM-capable pin, so this mode is
  // deliberately partial: it reports ready when at least one axis is usable and
  // rejects commands for the axes that have no pin.
  uint8_t attachedCount = 0;
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    servoAttached[i] = false;
    if (SERVO_PINS[i] < 0) {
      continue;
    }
    // attach() returns the assigned channel, or INVALID_SERVO (255) when the
    // library has no channel left for this object.
    uint8_t channel = servos[i].attach(SERVO_PINS[i], SERVO_MIN_US, SERVO_MAX_US);
    if (channel != INVALID_SERVO && servos[i].attached()) {
      servoAttached[i] = true;
      attachedCount++;
    }
  }
  if (attachedCount < NUM_SERVOS) {
    Serial.print(F("WARNING: direct PWM supports only "));
    Serial.print(attachedCount);
    Serial.print(F(" of "));
    Serial.print(NUM_SERVOS);
    Serial.println(F(" axes on this board"));
  }
  driverReady = attachedCount > 0;
#endif

  if (driverReady) {
    Serial.println(F("BIONIC_HAND_READY"));
  } else {
    Serial.println(F("BIONIC_HAND_NOT_READY"));
  }

  // Set all servos to default position
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    currentAngles[i] = DEFAULT_ANGLE;
    targetAngles[i] = DEFAULT_ANGLE;
    setServoAngle(i, DEFAULT_ANGLE);
  }

  lastCommandTime = millis();
  lastInterpolationTime = lastCommandTime;
  Serial.println(F("All servos initialized to default position"));
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
  // --- 1. Process incoming serial commands ---
  processSerial();

  // --- 2. Smooth servo interpolation ---
  interpolateServos();

  // --- 3. Command timeout safety ---
  checkTimeout();

  // Small delay for stability
  delay(5);
}

// ============================================================================
// Serial Protocol
// ============================================================================

void processSerial() {
  // Buffer for incoming line.  Reject overlong or malformed frames instead of
  // executing a retained prefix.
  static char buffer[256];
  static uint8_t bufIndex = 0;
  static bool overflow = false;
  static bool malformed = false;

  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (overflow) {
        Serial.println(F("ERROR: command too long"));
      } else if (malformed) {
        Serial.println(F("ERROR: invalid command"));
      } else if (bufIndex > 0) {
        buffer[bufIndex] = '\0';
        parseCommand(buffer);
      }
      overflow = false;
      malformed = false;
      bufIndex = 0;
    } else if (!overflow && !malformed) {
      uint8_t byteValue = static_cast<uint8_t>(c);
      if (byteValue < 32 || byteValue > 126) {
        // C-string parsing would truncate at NUL and accept a valid prefix.
        malformed = true;
      } else if (bufIndex < sizeof(buffer) - 1) {
        buffer[bufIndex++] = c;
      } else {
        overflow = true;
      }
    }
  }
}

void parseCommand(const char* command) {
  // Optional leading sequence id: "#<digits> <frame>".  The id is echoed on
  // the response so the host can match an acknowledgment to the exact frame it
  // sent; without it, two commands that update the same number of axes produce
  // indistinguishable ACKs.
  int seq = -1;
  if (command[0] == '#') {
    const char* cursor = command + 1;
    long value = 0;
    uint8_t digits = 0;
    while (*cursor >= '0' && *cursor <= '9' && digits < 3) {
      value = value * 10 + (*cursor - '0');
      cursor++;
      digits++;
    }
    if (digits > 0 && *cursor == ' ') {
      seq = (int)value;
      command = cursor + 1;
    }
  }

  // PING is a health probe, not a joint command, and intentionally does not
  // refresh the motion watchdog.
  if (strcmp(command, "PING") == 0) {
    if (refreshDriverHealth()) {
      Serial.println(F("PONG"));
    } else {
      printError(seq, F("driver not ready"));
    }
    return;
  }

  // Attempt recovery before rejecting a command after a transient disconnect.
  if (!driverReady && !refreshDriverHealth()) {
    printError(seq, F("driver not ready"));
    return;
  }

  // Parse into a temporary copy first.  No target is changed until every
  // field in the frame has passed validation.
  char cmdCopy[256];
  strncpy(cmdCopy, command, sizeof(cmdCopy) - 1);
  cmdCopy[sizeof(cmdCopy) - 1] = '\0';

  float stagedTargets[NUM_SERVOS];
  bool seen[NUM_SERVOS] = {false};
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    stagedTargets[i] = targetAngles[i];
  }

  char* cursor = cmdCopy;
  uint8_t updatedCount = 0;
  bool valid = true;
  bool unavailable = false;

  while (valid) {
    if (*cursor == '\0') {
      valid = false;
      break;
    }

    // Split one field without strtok(): strtok() skips empty fields and would
    // accept leading, consecutive, or trailing commas.
    char* nextComma = strchr(cursor, ',');
    if (nextComma != NULL) {
      *nextComma = '\0';
    }

    char* colonPos = strchr(cursor, ':');
    char* lastColon = strrchr(cursor, ':');
    if (colonPos == NULL || colonPos != lastColon || colonPos == cursor ||
        colonPos[1] == '\0') {
      valid = false;
      break;
    }

    *colonPos = '\0';
    int angle = 0;
    int8_t idx = findServoIndex(cursor);
    if (idx < 0 || seen[idx] || !parseAngleText(colonPos + 1, &angle)) {
      valid = false;
      break;
    }
    // A named axis that exists in the protocol but has no usable output on
    // this board must be reported, not silently accepted.
    if (!servoAvailable(idx)) {
      unavailable = true;
      valid = false;
      break;
    }

    seen[idx] = true;
    stagedTargets[idx] = angle;
    updatedCount++;

    if (nextComma == NULL) {
      break;
    }
    cursor = nextComma + 1;
  }

  if (!valid || updatedCount == 0) {
    if (unavailable) {
      printError(seq, F("axis not available on this board"));
    } else {
      printError(seq, F("invalid command"));
    }
    return;
  }

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    if (seen[i]) {
      targetAngles[i] = stagedTargets[i];
    }
  }

  // Confirm at least one physical output attempt before acknowledging the
  // frame.  If the driver is unavailable, leave the target staged for retry
  // but do not refresh the motion watchdog or claim success.
  if (!interpolateServos()) {
    printError(seq, F("actuator update failed"));
    return;
  }
  lastCommandTime = millis();

  Serial.print(F("OK:"));
  if (seq >= 0) {
    Serial.print(seq);
    Serial.print(':');
  }
  Serial.print(updatedCount);
  Serial.println(F(" servos updated"));
}

void printError(int seq, const __FlashStringHelper* message) {
  Serial.print(F("ERROR"));
  if (seq >= 0) {
    Serial.print('[');
    Serial.print(seq);
    Serial.print(']');
  }
  Serial.print(F(": "));
  Serial.println(message);
}

int8_t findServoIndex(const char* name) {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    if (strcmp(name, SERVO_NAMES[i]) == 0) {
      return i;
    }
  }
  return -1;
}

// ============================================================================
// Servo Control
// ============================================================================

bool interpolateServos() {
  unsigned long now = millis();
  float dt = (now - lastInterpolationTime) / 1000.0f;
  lastInterpolationTime = now;
  if (dt < 0.0f) {
    dt = 0.0f;
  }
  if (dt > MAX_INTERPOLATION_DT) {
    dt = MAX_INTERPOLATION_DT;
  }
  // A zero step still writes the current angle, so every axis gets a real
  // output attempt and the caller's success check stays meaningful.
  float maxStep = MAX_SPEED_DPS * dt;

  bool allSucceeded = true;

  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    float delta = targetAngles[i] - currentAngles[i];

    if (abs(delta) < 0.5) {
      // Close enough - snap to target only after confirming the output path.
      if (setServoAngle(i, targetAngles[i])) {
        currentAngles[i] = targetAngles[i];
      } else {
        allSucceeded = false;
      }
      continue;
    }

    // Rate-limited movement.  Keep the old state until the driver accepts the
    // write, so a disconnected PCA9685 can retry the same target after recovery.
    float nextAngle = currentAngles[i];
    if (delta > maxStep) {
      nextAngle += maxStep;
    } else if (delta < -maxStep) {
      nextAngle -= maxStep;
    } else {
      nextAngle += delta;
    }

    if (setServoAngle(i, nextAngle)) {
      currentAngles[i] = nextAngle;
    } else {
      allSucceeded = false;
    }
  }

  return allSucceeded;
}

bool setServoAngle(uint8_t index, float angle) {
  // Clamp angle
  angle = constrain(angle, 0.0f, 180.0f);

  if (index >= NUM_SERVOS) {
    return false;
  }

#if DRIVER_MODE == 1
  // Refresh the I2C presence check before output.  A disconnected driver must
  // not be reported as a successful command by the protocol.
  if (!refreshDriverHealth()) {
    return false;
  }
  // PCA9685: convert angle to pulse length
  uint16_t pulseLen = map(angle, 0, 180, SERVO_MIN_US, SERVO_MAX_US);
  // PCA9685 tick count = pulse_us * 4096 * frequency / 1,000,000.
  uint32_t pulseTicksWide = static_cast<uint32_t>(pulseLen) * 4096UL * SERVO_FREQ / 1000000UL;
  uint16_t pulseTicks = static_cast<uint16_t>(constrain(pulseTicksWide, 0UL, 4095UL));
  pwm.setPWM(SERVO_CHANNELS[index], 0, pulseTicks);
  return true;
#else
  // Direct PWM
  if (!driverReady || SERVO_PINS[index] < 0) {
    return false;
  }
  servos[index].write((int)angle);
  return true;
#endif
}

// ============================================================================
// Safety
// ============================================================================

void checkTimeout() {
  static bool timeoutActive = false;

  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    // Return all servos to default position once per timeout event.
    if (!timeoutActive) {
      timeoutActive = true;
      Serial.println(F("WARNING: Command timeout - returning to default pose"));

      for (uint8_t i = 0; i < NUM_SERVOS; i++) {
        targetAngles[i] = DEFAULT_ANGLE;
      }
    }
  } else {
    // Reset timeout state when a valid command resumes.
    timeoutActive = false;
  }
}

// ============================================================================
// Debug / Diagnostic Helpers (not used in normal operation)
// ============================================================================

void printServoStatus() {
  Serial.println(F("\n--- Servo Status ---"));
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    Serial.print(SERVO_NAMES[i]);
    Serial.print(F(": cur="));
    Serial.print(currentAngles[i], 1);
    Serial.print(F("° tgt="));
    Serial.print(targetAngles[i], 1);
    Serial.println(F("°"));
  }
  Serial.println(F("--------------------"));
}
