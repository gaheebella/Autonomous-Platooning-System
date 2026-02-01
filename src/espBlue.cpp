#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// Wi-Fi로 라즈베리파이(MQTT 브로커)에 연결하고
// leader/speed 토픽을 구독해서 리더의 실제 속도(cm/s) 를 받으면
// 그 속도를 자기 모터 PWM(0~255) 로 변환해 기본 속도(speed_base)로 삼고
// 초음파로 앞차 거리 기반 ACC(가감속) 를 하고
// 라인센서로 차선 유지(좌/우 차등 구동) 를 수행

// ==========================================
// 1. 네트워크 및 MQTT 설정
// ==========================================
const char* ssid = " ";         // 와이파이 이름
const char* password = " ";    // 와이파이 비번 
const char* mqtt_server = "192.168.0.123";     // MQTT 브로커 IP (리더 라즈베리파이 IP 주소)
const char* topic_speed = "leader/speed";

WiFiClient espClient;
PubSubClient client(espClient);

// ==========================================
// 2. 핀 및 상수 정의
// ==========================================

// 모터 핀
#define MOTOR_A_a D5      
#define MOTOR_A_b D6      
#define MOTOR_B_a D7      
#define MOTOR_B_b D8      

// 초음파 핀
#define TRIG_PIN D4       
#define ECHO_PIN D10      

// 라인센서 핀 -> 왼: 아날로그 / 오: 디지털
#define LINESENS_L A0     
#define LINESENS_R D2     

// 거리 제어 기준값 -> 목표거리: 25cm / 허용 오차 += 5cm / 10cm 이하면 강제 정지
#define TARGET_DIST 25    
#define DIST_MARGIN 5     
#define STOP_DISTANCE 10  

// ==========================================
// 3. 전역 변수 (속도 제어용)
// ==========================================

// 리더 속도를 MQTT로 받음 -> 그 값을 PWM으로 바꾼 결과가 speed_base (즉, 팔로워의 기본 주행 속도 PWM))
int speed_base = 0;

// 속도(m/s) 와 PWM 매핑
// 매핑 기준값 변경
// 리더가 12cm/s(9~15의 중간)일 때 PWM 80이 나오도록 계산된 값 (약 38.0)
const float MAX_LEADER_SPEED_CMS = 38.0;

// 최소 동작 PWM (PWM이 너무 낮으면 모터가 웅~ 소리만 나고 안 돌 때를 대비한 최소값)
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

// MQTT 연결 유지
void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();

    // 1. 센서 값 읽기
    // 왼쪽 라인센서: 아날로그 값이 500보다 크면 1(라인 감지)로 판단
    // 오른쪽 라인센서: 디지털 입력 그대로 사용
    // 초음파 거리: cm로 반환
    boolean line_l = (analogRead(LINESENS_L) > 500) ? 1 : 0;
    boolean line_r = digitalRead(LINESENS_R);
    long distance = get_distance();

    // 기본은 MQTT로 받은 속도 기반 PWM(speed_base)
    // 이후 거리 제어(ACC)로 가감속되며 current_pwm이 바뀜
    int current_pwm = speed_base;

    // (A) 안전 정지 -> 앞차와 10cm 미만이면 즉시 정지 후 이번 루프 종료
    if (distance > 0 && distance < STOP_DISTANCE) {
        motor_stop();
        return;
    }

    // (B) 리더 정지 처리
    // PWM이 너무 낮으면 팔로워도 아예 정지
    if (speed_base < 30) {
        motor_stop();
        return;
    }

    // (C) 거리 제어 (ACC)
    if (distance < TARGET_DIST - DIST_MARGIN) {
        // 가까우면 감속 (현재 속도의 70%)
        current_pwm = (int)(speed_base * 0.7);
    }
    else if (distance > TARGET_DIST + DIST_MARGIN && distance < 100) {
        // 멀어지면 가속 (현재 속도의 1.3배, 최대 255)
        current_pwm = (int)(speed_base * 1.3);
        if (current_pwm > 255) current_pwm = 255;
    }
    else {
        // 적정 거리 유지
        current_pwm = speed_base;
    }

    // (D) 최소 PWM 보정 (너무 느려서 모터가 멈추는 것 방지)
    if (current_pwm > 0 && current_pwm < MIN_MOVING_PWM) {
        current_pwm = MIN_MOVING_PWM;
    }

    // 3. 주행 (차선 유지 / 라인 트레이싱)
    // 조향을 P제어가 아닌, 한 쪽 바퀴를 절반으로 줄이는 규칙 기반
    // 양쪽 라인 감지(1,1) : 직진
    if (line_l == 1 && line_r == 1) {
        motor_drive(current_pwm, current_pwm);
    }
    else if (line_l == 0 && line_r == 1) {
        // 왼쪽 이탈(0,1) -> 오른쪽으로 턴 (오른쪽 바퀴 감속)
        motor_drive(current_pwm, (int)(current_pwm * 0.5));
    }
    else if (line_l == 1 && line_r == 0) {
        // 오른쪽 이탈(1,0) -> 왼쪽으로 턴 (왼쪽 바퀴 감속)
        motor_drive((int)(current_pwm * 0.5), current_pwm);
    }
    // 둘 다 못 봄(0,0): 정지
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

    // 매핑 계산: (리더속도 / 38.0) * 255
    // 예: 리더 12cm/s -> (12/38)*255 = 80 PWM
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
        // 클라이언트 ID는 FollowerCar1, FollowerCar2 등으로 변경 권장
        if (client.connect("FollowerCar1")) {
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