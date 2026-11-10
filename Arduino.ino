#include <ESP32Servo.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// 1.3 Inch OLED (SH1106G chipset at address 0x3C)
Adafruit_SH1106G display = Adafruit_SH1106G(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

Servo panServo;
Servo tiltServo;

// ESP32 GPIO Pins
const int PAN_PIN = 18;      // Pan Servo signal -> GPIO 18
const int TILT_PIN = 19;     // Tilt Servo signal -> GPIO 19
const int LASER_PIN = 5;     // Physical Laser Diode Control -> GPIO 5
const int BTN_PIN = 4;       // Physical Laser Push Button -> GPIO 4 (Active LOW)

int currentPan = 90;
int currentTilt = 90;
bool laserState = true;      // Laser defaults to ON
bool lastBtnState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

bool oledAvailable = false;
unsigned long lastDisplayUpdate = 0;

void setup() {
  Serial.begin(115200);
  
  // Laser Pin & Physical Button Setup
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, HIGH); // Laser ON initially
  
  pinMode(BTN_PIN, INPUT_PULLUP); // 2-Pin Push Button to GND

  // Allocate ESP32 PWM Timers for Servos
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  panServo.setPeriodHertz(50);
  tiltServo.setPeriodHertz(50);
  
  panServo.attach(PAN_PIN, 500, 2400);
  tiltServo.attach(TILT_PIN, 500, 2400);
  
  panServo.write(90);
  tiltServo.write(90);

  // Initialize I2C on ESP32 (SDA=GPIO 21, SCL=GPIO 22)
  Wire.begin(21, 22);

  if (display.begin(0x3C, true)) {
    oledAvailable = true;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);
    display.setCursor(14, 18);
    display.println("ESP32 TURRET");
    display.setCursor(24, 34);
    display.println("STUDIO v2.0");
    display.display();
    delay(1000);
  }
}

void loop() {
  // 1. PHYSICAL BUTTON TOGGLE DEBOUNCE LOGIC
  int reading = digitalRead(BTN_PIN);
  if (reading != lastBtnState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    static bool buttonState = HIGH;
    if (reading != buttonState) {
      buttonState = reading;
      if (buttonState == LOW) {
        laserState = !laserState;
        digitalWrite(LASER_PIN, laserState ? HIGH : LOW);
      }
    }
  }
  lastBtnState = reading;

  // 2. SERIAL RECEIVE LOGIC (pan,tilt or pan,tilt,laser or L1/L0)
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    if (data.startsWith("L1")) {
      laserState = true;
      digitalWrite(LASER_PIN, HIGH);
    } else if (data.startsWith("L0")) {
      laserState = false;
      digitalWrite(LASER_PIN, LOW);
    } else {
      int commaIndex = data.indexOf(',');
      if (commaIndex > 0) {
        int panAngle = data.substring(0, commaIndex).toInt();
        int secondComma = data.indexOf(',', commaIndex + 1);
        
        int tiltAngle;
        if (secondComma > 0) {
          tiltAngle = data.substring(commaIndex + 1, secondComma).toInt();
          int lState = data.substring(secondComma + 1).toInt();
          laserState = (lState != 0);
          digitalWrite(LASER_PIN, laserState ? HIGH : LOW);
        } else {
          tiltAngle = data.substring(commaIndex + 1).toInt();
        }
        
        currentPan = constrain(panAngle, 0, 180);
        currentTilt = constrain(tiltAngle, 0, 180);

        panServo.write(currentPan);
        tiltServo.write(currentTilt);
      }
    }
  }

  // 3. Non-blocking OLED Telemetry Update (every 150ms)
  if (oledAvailable && (millis() - lastDisplayUpdate >= 150)) {
    lastDisplayUpdate = millis();
    display.clearDisplay();
    
    display.drawRect(0, 0, 128, 64, SH110X_WHITE);
    display.drawFastHLine(0, 14, 128, SH110X_WHITE);

    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);
    display.setCursor(4, 4);
    display.print("ESP32 HUD");
    
    display.setCursor(72, 4);
    if (laserState) {
      display.print("[LSR:ON]");
    } else {
      display.print("[LSR:OFF]");
    }

    display.setCursor(4, 20);
    display.print("PAN : ");
    display.print(currentPan);
    display.print(" deg");
    
    int panBarW = map(currentPan, 0, 180, 0, 118);
    display.drawRect(4, 30, 120, 7, SH110X_WHITE);
    display.fillRect(5, 31, panBarW, 5, SH110X_WHITE);

    display.setCursor(4, 42);
    display.print("TILT: ");
    display.print(currentTilt);
    display.print(" deg");

    int tiltBarW = map(currentTilt, 0, 180, 0, 118);
    display.drawRect(4, 52, 120, 7, SH110X_WHITE);
    display.fillRect(5, 53, tiltBarW, 5, SH110X_WHITE);

    display.display();
  }
}