#include "camera.h"
#include "himax.h"
HM01B0 himax;
Camera cam(himax);

#define IMAGE_MODE CAMERA_GRAYSCALE
FrameBuffer fb;

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Initialize the camera
  // QVGA = 320x240
  // IMAGE_GRAYSCALE creates a grayscale image. Change to IMAGE_RGB565 for color.
  if (!cam.begin(CAMERA_R320x240, IMAGE_MODE, 30)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  Serial.println("Camera initialized.");
}

void loop() {
  
  if (cam.grabFrame(fb, 3000) == 0) {
    // Get the frame buffer
    Serial.write(fb.getBuffer());    
  } else {
    Serial.println("Failed to capture image.");
  }
  delay(1000); // Wait for 2 seconds before next capture
}
