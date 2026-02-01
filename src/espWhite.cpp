#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// ==========================================
// 1. 네트워크 및 MQTT 설정
// ==========================================
const char* ssid = " "         // 와이파이 이름
const char* password = " "; // 와이파이 비번 
const char* mqtt_server = "192.168.0.123";     // 리더(라즈베리파이) IP 주소

const char* topic_speed = "leader/speed";

WiFiClient espClient;
PubSubClient client(espClient);

// ==========================================
// 2. 핀 및 상수 정의
// ==========================================
#define MOTOR_A_a D5      
#define MOTOR_A_b D6      
#define MOTOR_B_a D7      
#define MOTOR_B_b D8      

#define TRIG_PIN D4       
#define ECHO_PIN D10      

#define LINESENS_L A0     
#define LINESENS_R D2     

#define TARGET_DIST 25    
#define DIST_MARGIN 5     
#define STOP_DISTANCE 10  

// ==========================================
// 3. 전역 변수
// ==========================================

// speed_base: 팔로워가 실제로 “기본으로 달릴 PWM”
int speed_base = 0;

// 매핑 기준값
// 리더가 12cm/s일 때 PWM 100이 나오도록 계산된 값 (30.6) -> 리더 속도를 PWM으로 바꾸는 스케일 기준값
const float MAX_LEADER_SPEED_CMS = 30.6;

// 최소 동작 PWM (안전장치)
const int MIN_MOVING_PWM = 60;

// ==========================================
// 4. Setup
// ==========================================
void setup_wifi();
void callback(char* topic, byte* payload, unsigned int length);
void reconnect();
long get_distance();
void motor_drive(int spd_l, int spd_r);
void motor_stop();

void setup() {
    Serial.begin(115200);

    pinMode(MOTOR_A_a, OUTPUT); pinMode(MOTOR_A_b, OUTPUT);
    pinMode(MOTOR_B_a, OUTPUT); pinMode(MOTOR_B_b, OUTPUT);
    pinMode(LINESENS_R, INPUT);
    pinMode(TRIG_PIN, OUTPUT); pinMode(ECHO_PIN, INPUT);

    analogWriteRange(255);
    analogWriteFreq(1000);

    setup_wifi();
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
}

// ==========================================
// 5. Main Loop
// ==========================================
void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();

    // 1. 센서 값 읽기
    boolean line_l = (analogRead(LINESENS_L) > 500) ? 1 : 0;
    boolean line_r = digitalRead(LINESENS_R);
    long distance = get_distance();

    int current_pwm = speed_base;

    // (A) 안전 정지
    if (distance > 0 && distance < STOP_DISTANCE) {
        motor_stop();
        return;
    }

    // (B) 리더 정지 시
    if (speed_base < 30) {
        motor_stop();
        return;
    }

    // (C) 거리 제어 (ACC)
    if (distance < TARGET_DIST - DIST_MARGIN) {
        // 가까우면 감속 (70%)
        current_pwm = (int)(speed_base * 0.7);
    }
    else if (distance > TARGET_DIST + DIST_MARGIN && distance < 100) {
        // 멀어지면 가속 (1.3배)
        current_pwm = (int)(speed_base * 1.3);
        if (current_pwm > 255) current_pwm = 255;
    }
    else {
        // 적정 거리 유지
        current_pwm = speed_base;
    }

    // (D) 최소 PWM 보정
    if (current_pwm > 0 && current_pwm < MIN_MOVING_PWM) {
        current_pwm = MIN_MOVING_PWM;
    }

    // 3. 주행 (차선 유지)
    if (line_l == 1 && line_r == 1) {
        motor_drive(current_pwm, current_pwm);
    }
    else if (line_l == 0 && line_r == 1) {
        // 왼쪽 이탈 -> 오른쪽 턴 (안쪽 바퀴 감속)
        motor_drive(current_pwm, (int)(current_pwm * 0.5));
    }
    else if (line_l == 1 && line_r == 0) {
        // 오른쪽 이탈 -> 왼쪽 턴
        motor_drive((int)(current_pwm * 0.5), current_pwm);
    }
    else {
        motor_stop();
    }

    delay(10);
}

// ==========================================
// 6. 보조 함수들
// ==========================================

void setup_wifi() {
    delay(10);
    Serial.print("Connecting to ");
    Serial.println(ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
}

void callback(char* topic, byte* payload, unsigned int length) {
    char msg[length + 1];
    memcpy(msg, payload, length);
    msg[length] = '\0';

    float leader_speed_cms = atof(msg);

    // 매핑 계산: (리더속도 / 30.6) * 255
    // 예: 리더 12cm/s -> (12/30.6)*255 = 약 100 PWM
    int mapped_pwm = (int)((leader_speed_cms / MAX_LEADER_SPEED_CMS) * 255.0);

    speed_base = constrain(mapped_pwm, 0, 255);

    Serial.print("Leader: ");
    Serial.print(leader_speed_cms);
    Serial.print(" cm/s -> My PWM: ");
    Serial.println(speed_base);
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Attempting MQTT connection...");
        // 다른 차량이면 ID 변경 ("FollowerCar2" 등)
        if (client.connect("FollowerCar2")) {
            Serial.println("connected");
            client.subscribe(topic_speed);
        }
        else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            delay(2000);
        }
    }
}

long get_distance() {
    digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    long duration = pulseIn(ECHO_PIN, HIGH, 23200);
    if (duration == 0) return 999;
    return duration * 0.034 / 2;
}

void motor_drive(int spd_l, int spd_r) {
    spd_r = constrain(spd_r, 0, 255);
    spd_l = constrain(spd_l, 0, 255);
    analogWrite(MOTOR_A_a, spd_l); digitalWrite(MOTOR_A_b, LOW);
    analogWrite(MOTOR_B_a, spd_r); digitalWrite(MOTOR_B_b, LOW);
}

void motor_stop() {
    digitalWrite(MOTOR_A_a, LOW); digitalWrite(MOTOR_A_b, LOW);
    digitalWrite(MOTOR_B_a, LOW); digitalWrite(MOTOR_B_b, LOW);
}