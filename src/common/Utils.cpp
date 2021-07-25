//
// Created by matteo on 7/18/21.
//

#include "Utils.hpp"
#include <cmath>

namespace utils {
  unsigned int HSBtoRGB(float hue, float saturation, float brightness) {
    int r = 0, g = 0, b = 0;
    if (saturation == 0) {
      r = g = b = (int) (brightness * 255.0f + 0.5f);
    } else {
      float h = (hue - (float) std::floor(hue)) * 6.0f;
      float f = h - (float) std::floor(h);
      float p = brightness * (1.0f - saturation);
      float q = brightness * (1.0f - saturation * f);
      float t = brightness * (1.0f - (saturation * (1.0f - f)));
      switch ((unsigned) h) {
        case 0:
          r = std::lround(brightness * 255.0f + 0.5f);
          g = (unsigned) (t * 255.0f + 0.5f);
          b = (unsigned) (p * 255.0f + 0.5f);
          break;
        case 1:
          r = (unsigned) (q * 255.0f + 0.5f);
          g = (unsigned) (brightness * 255.0f + 0.5f);
          b = (unsigned) (p * 255.0f + 0.5f);
          break;
        case 2:
          r = (unsigned) (p * 255.0f + 0.5f);
          g = (unsigned) (brightness * 255.0f + 0.5f);
          b = (unsigned) (t * 255.0f + 0.5f);
          break;
        case 3:
          r = (unsigned) (p * 255.0f + 0.5f);
          g = (unsigned) (q * 255.0f + 0.5f);
          b = (unsigned) (brightness * 255.0f + 0.5f);
          break;
        case 4:
          r = (unsigned) (t * 255.0f + 0.5f);
          g = (unsigned) (p * 255.0f + 0.5f);
          b = (unsigned) (brightness * 255.0f + 0.5f);
          break;
        case 5:
          r = (unsigned) (brightness * 255.0f + 0.5f);
          g = (unsigned) (p * 255.0f + 0.5f);
          b = (unsigned) (q * 255.0f + 0.5f);
          break;
      }
    }
    return ((0xFF000000 | (r << 16) | (g << 8) | b));
  }
}