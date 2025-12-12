// Arduino Sketch Template for TinyML

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);
  Serial.println("TinyML Model Initialized");
}

void loop() {
  // Main loop
  delay(1000);
}
