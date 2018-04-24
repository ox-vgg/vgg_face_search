/**
 * Provides routines for reading and writing to jpegs.
 *
 * d = jp_jpeg_read(fn, w, h) - Loads,decodes and returns the image stored in fn as RGB.
 *                              The width and height are stored in w and h.
 *                              Returns 0 on error.
 *
 * jp_jpeg_write(fn, d, w, h) - Encodes and saves the raw RGB data stored at d, with size (w,h) to fn.
 *                              Returns 0 on error.
 *
 * EXAMPLES
 * -Read a jpeg from a file.
 *   int w, h;
 *   unsigned char* d = jp_jpeg_read("blah.jpg", w, h);
 *
 */
#ifndef __JP_JPEG_HPP
#define __JP_JPEG_HPP

#include <iostream>
#include <string>

#include <setjmp.h>

extern "C" {
#include "jpeglib.h"
}

struct jp_jpeg_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;

  static
  void
  fatal_error_handler(j_common_ptr cinfo)
  {
    std::cerr << cinfo->err;
    longjmp(((jp_jpeg_error_mgr*)cinfo->err)->setjmp_buffer, 1);
    return;
  }
};



inline
unsigned char*
jp_jpeg_read(const std::string& fname, int& width, int& height)
{
  struct jpeg_decompress_struct cinfo;
  struct jp_jpeg_error_mgr jerr;

  FILE* fobj = fopen(fname.c_str(), "rb");
  if (!fobj) {
    std::cerr << "Can't load " << fname << std::endl;
    perror(0);
    width = 0; height = 0; return 0;
  }

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jerr.fatal_error_handler;
  if (setjmp(jerr.setjmp_buffer))
  {
    jpeg_destroy_decompress(&cinfo);
    fclose(fobj);
    return 0;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fobj);
  jpeg_read_header(&cinfo, TRUE);

  cinfo.do_fancy_upsampling = FALSE;
  cinfo.do_block_smoothing = FALSE;

  jpeg_start_decompress(&cinfo);

  width = cinfo.output_width;
  height = cinfo.output_height;

  unsigned char* ret = new unsigned char[width*height*3];
  unsigned char* rowp[32];

  while (cinfo.output_scanline < cinfo.output_height) {
    for (int y=0; y<32; ++y)
      rowp[y] = &ret[(y+cinfo.output_scanline)*width*3];
    jpeg_read_scanlines(&cinfo, rowp, 32); // ?
  }

  if (cinfo.output_components == 1) {
    for (int y=0; y<height; ++y) {
      for (int x=width-1; x>=0; --x) {
        ret[3*width*y + 3*x + 0] = ret[3*width*y + x];
        ret[3*width*y + 3*x + 1] = ret[3*width*y + x];
        ret[3*width*y + 3*x + 2] = ret[3*width*y + x];
      }
    }
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  fclose(fobj);

  return ret;
}



inline
int
jp_jpeg_write(const std::string& fn, const unsigned char* d, int width, int height, int quality=100)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  FILE* fobj;
  fobj = fopen(fn.c_str(), "wb");
  if (!fobj) return 0;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  jpeg_stdio_dest(&cinfo, fobj);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);

  jpeg_set_quality(&cinfo, quality, TRUE);
  cinfo.dct_method = JDCT_ISLOW;
  cinfo.smoothing_factor = 0; /*-1; <-- this values black causes holes in some images*/
  cinfo.optimize_coding = TRUE;
  jpeg_start_compress(&cinfo, TRUE);

  const unsigned char* rowp[1];
  while (cinfo.next_scanline < cinfo.image_height) {
    rowp[0] = &d[cinfo.next_scanline*width*3];
    jpeg_write_scanlines(&cinfo, (unsigned char**)rowp, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(fobj);
  jpeg_destroy_compress(&cinfo);

  return 1;
}

#endif
