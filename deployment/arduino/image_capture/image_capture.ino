#include "camera.h"
#include "himax.h"
HM01B0 himax;
Camera cam(himax);

#define IMAGE_MODE CAMERA_GRAYSCALE
FrameBuffer fb;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  if (!cam.begin(CAMERA_R160x120, IMAGE_MODE, 30)) {// Resoluciones disponibles CAMERA_R160x120     = 0,   /* QQVGA Resolution   */ CAMERA_R320x240     = 1,   /* QVGA Resolution    */ CAMERA_R320x320     = 2,   /* 320x320 Resolution */
    Serial.println("Failed to initialize camera!");
    while (1);
  }
}

void loop() {
  
  if (cam.grabFrame(fb, 3000) == 0) {
    Serial.write(fb.getBuffer(),cam.frameSize());    
  } else {
    Serial.println("Failed to capture image.");
  }
  delay(40);
}
