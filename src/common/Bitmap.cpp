//
// Created by matteo on 3/18/22.
//

#include "Bitmap.hpp"
#include <algorithm>
#include <cmath>

void center_circle(Bitmap<Host> &bitmap) {
    for (int x = 0; x < bitmap.w; x++) {
        for (int y = 0; y < bitmap.h; y++) {
            unsigned rel_x = bitmap.w / 2 - x;
            unsigned rel_y = bitmap.h / 2 - y;
            double r = sqrt(rel_x * rel_x + rel_y * rel_y);
            if (r < std::min(bitmap.w, bitmap.h) * 0.1) { bitmap(x, y) = true; }
        }
    }
}

void center_square(Bitmap<Host> &bitmap) {
    // TODO: implement
}
