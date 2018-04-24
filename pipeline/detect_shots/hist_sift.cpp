#include <iostream>

#include "jp_img.hpp"

/**
 * I think this is already used by libm. Anyway, the inlining seems to help.
 * V slow on intel processors: Latency ~ 250 clocks, throughput ~ 170 clocks.
 */
inline
float
fast_atan2f(float x, float y)
{
  float ret;
  asm ("fpatan" : "=t"(ret) : "0"(y), "u"(x));
  return ret;
}

void
hist_sift(const unsigned char* data, int w, int h, unsigned char* desco)
{
  jp_img_wrapper<const unsigned char> img(data, w, h, 3);

  jp_img<float> grey_img
    = jp_img_rgb2grey<jp_img<float> >(img);

  jp_img<float> img_smooth = grey_img;
  //  = jp_img_smooth<jp_img<float> >(grey_img, 2.0f);

  std::pair<jp_img<float>, jp_img<float> > img_dx_dy =
    jp_img_calc_x_y_derivative<jp_img<float> >(img_smooth);

  jp_img<float> dx = img_dx_dy.first;
  jp_img<float> dy = img_dx_dy.second;

  jp_img<float> img_mag(w, h);
  jp_img<float> img_ori(w, h);

  for (int y=0; y<h; ++y) {
    for (int x=0; x<w; ++x) {
      img_mag(x,y) = sqrt(dx(x,y)*dx(x,y) + dy(x,y)*dy(x,y));
      img_ori(x,y) = fast_atan2f(dy(x,y), dx(x,y));
    }
  }

  float desc[128];
  for (int i=0; i<128; ++i) desc[i] = 0.0f;

  for (int sq = 0; sq < 16; ++sq) {
    int row = sq/4;
    int col = sq%4;

    int x1 = (w * col)/4;
    int x2 = (w * (col+1))/4;
    int y1 = (h * row)/4;
    int y2 = (h * (row+1))/4;


    for (int y=y1; y<y2; ++y) {
      for (int x=x1; x<x2; ++x) {
        int bin = (int)(8 * (img_ori(x,y) + M_PI + 0.001)/(2.0 * M_PI));
        desc[sq*8 + bin] += img_mag(x,y);
      }
    }
  }

  float sqlen = 0.0;
  for (int i=0; i<128; ++i) sqlen += (desc[i]*desc[i]);
  sqlen = 1.0 / sqrt(sqlen);
  for (int i=0; i<128; ++i) desc[i] *= sqlen;

  unsigned char descuc[128];
  for (int i=0; i<128; ++i) {
    int ival = (int)(512.0*desc[i]);
    descuc[i] = (ival < 256) ? (unsigned char)ival : (unsigned char)255;
  }

  std::copy(descuc, descuc + 128, desco);

//  for (int i=0; i<128; ++i) {
//    std::cout << (int)descuc[i] << " ";
//  }
//  std::cout << std::endl;
  //getchar();

  //

  //jp_img_display(img_smooth);
}
