//
// Created by matteo on 7/18/21.
//

#include "Utils.hpp"
#include <cmath>

namespace utils {
  unsigned int HSBtoRGB(float hue, float saturation, float brightness) {
    int r = 0, g = 0, b = 0;
    if (saturation == 0) {
      r = g = b = lround(brightness * 255.0f + 0.5f);
    } else {
      float h = (hue - (float) std::floor(hue)) * 6.0f;
      float f = h - (float) std::floor(h);
      float p = brightness * (1.0f - saturation);
      float q = brightness * (1.0f - saturation * f);
      float t = brightness * (1.0f - (saturation * (1.0f - f)));
      switch ((unsigned) h) {
        case 0:
          r = lround(brightness * 255.0f + 0.5f);
          g = lround(t * 255.0f + 0.5f);
          b = lround(p * 255.0f + 0.5f);
          break;
        case 1:
          r = lround(q * 255.0f + 0.5f);
          g = lround(brightness * 255.0f + 0.5f);
          b = lround(p * 255.0f + 0.5f);
          break;
        case 2:
          r = lround(p * 255.0f + 0.5f);
          g = lround(brightness * 255.0f + 0.5f);
          b = lround(t * 255.0f + 0.5f);
          break;
        case 3:
          r = lround(p * 255.0f + 0.5f);
          g = lround(q * 255.0f + 0.5f);
          b = lround(brightness * 255.0f + 0.5f);
          break;
        case 4:
          r = lround(t * 255.0f + 0.5f);
          g = lround(p * 255.0f + 0.5f);
          b = lround(brightness * 255.0f + 0.5f);
          break;
        case 5:
          r = lround(brightness * 255.0f + 0.5f);
          g = lround(p * 255.0f + 0.5f);
          b = lround(q * 255.0f + 0.5f);
          break;
      }
    }
    return ((0xFF000000 | (r << 16) | (g << 8) | b));
  }
}