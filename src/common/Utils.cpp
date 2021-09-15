//
// Created by matteo on 7/18/21.
//

#include "Utils.hpp"
#include <cmath>

namespace utils {
  unsigned int HSBtoRGB(float hue, float saturation, float brightness) {
    int r = 0, g = 0, b = 0;
    if (saturation == 0) {
      r = g = b = std::floor(brightness * 255.0f + 0.5f);
    } else {
      float h = (hue - (float) std::floor(hue)) * 6.0f;
      float f = h - (float) std::floor(h);
      float p = brightness * (1.0f - saturation);
      float q = brightness * (1.0f - saturation * f);
      float t = brightness * (1.0f - (saturation * (1.0f - f)));
      switch ((unsigned) h) {
        case 0:
          r = std::floor(brightness * 255.0f + 0.5f);
          g = std::floor(t * 255.0f + 0.5f);
          b = std::floor(p * 255.0f + 0.5f);
          break;
        case 1:
          r = std::floor(q * 255.0f + 0.5f);
          g = std::floor(brightness * 255.0f + 0.5f);
          b = std::floor(p * 255.0f + 0.5f);
          break;
        case 2:
          r = std::floor(p * 255.0f + 0.5f);
          g = std::floor(brightness * 255.0f + 0.5f);
          b = std::floor(t * 255.0f + 0.5f);
          break;
        case 3:
          r = std::floor(p * 255.0f + 0.5f);
          g = std::floor(q * 255.0f + 0.5f);
          b = std::floor(brightness * 255.0f + 0.5f);
          break;
        case 4:
          r = std::floor(t * 255.0f + 0.5f);
          g = std::floor(p * 255.0f + 0.5f);
          b = std::floor(brightness * 255.0f + 0.5f);
          break;
        case 5:
          r = std::floor(brightness * 255.0f + 0.5f);
          g = std::floor(p * 255.0f + 0.5f);
          b = std::floor(q * 255.0f + 0.5f);
          break;
      }
    }
    return ((0xFF000000 | (r << 16) | (g << 8) | b));
  }
}