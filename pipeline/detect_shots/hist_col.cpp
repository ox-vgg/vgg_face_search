
#include "hist_col.hpp"

void
hist_col(const unsigned char* data, int w, int h, int bins, unsigned int* desc)
{
    const int MAXVAL = 256;

    int lut[MAXVAL];
    for (int i=0; i<MAXVAL; ++i) lut[i] = i * bins / MAXVAL;

    unsigned int* rdesc = desc;
    unsigned int* gdesc = desc + bins;
    unsigned int* bdesc = desc + 2*bins;
    for (int j=0; j<3*bins; ++j) desc[j] = 0;

    for (int y=0; y<h; ++y)
        for (int x=0; x<w; ++x)
        {
            ++rdesc[lut[*data++]];
            ++gdesc[lut[*data++]];
            ++bdesc[lut[*data++]];
        }
}
