//
// Created by matteo on 3/18/22.
//

#ifndef LBM_BITMAP_HPP
#define LBM_BITMAP_HPP

#include "Array.hpp"

template<typename Allocation>
using Bitmap = Array<bool, 2, Allocation>;

/* Put some geometric shapes inside a bitmap */
void center_circle(Bitmap<Host>& bitmap);
void center_square(Bitmap<Host>& bitmap);

#endif//LBM_BITMAP_HPP