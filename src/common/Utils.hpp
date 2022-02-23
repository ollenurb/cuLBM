//
// Created by matteo on 7/18/21.
//
#pragma once

namespace utils {

  /*
  * Returns the pixel value as an unsigned integer
  * RGB components can be retrieved using simple bit shifting
  * R: val << 16
  * G: val << 8
  * B: val <<
  */
  unsigned int HSBtoRGB(float hue, float saturation, float brightness);
}