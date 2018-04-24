#ifndef __HIST_COL_HPP
#define __HIST_COL_HPP

/*
 *  Computes a simple color histogram. Assumes RGBRGB... data. You must allocate
 *  3*bins*sizeof(unsigned int) memory for the descriptor.
 */
void
hist_col(const unsigned char* data, int w, int h, int bins, unsigned int* desc);

#endif
