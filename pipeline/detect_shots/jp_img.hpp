#ifndef __JP_IMG_HPP
#define __JP_IMG_HPP

#include <cmath>
#include <boost/shared_array.hpp>

template<class F>
class
jp_img_wrapper
{
public:
  typedef F value_type;
private:
  F* data_;
  const int width_;
  const int height_;
  const int chans_;

public:
  jp_img_wrapper(F* data, int width, int height, int chans = 1)
   : data_(data), width_(width), height_(height), chans_(chans)
  { }

  inline
  const F&
  operator()(int x, int y, int c = 0) const
  { return data_[chans_*(width_*y + x) + c]; }

  inline
  F&
  operator()(int x, int y, int c = 0)
  { return data_[chans_*(width_*y + x) + c]; }

  inline
  int
  width() const
  { return width_; }

  inline
  int
  height() const
  { return height_; }

  inline
  int
  chans() const
  { return chans_; }

  inline
  F*
  data()
  { return data_; }
  
  inline
  const F*
  data() const
  { return data_; }
};

template<class F>
class
jp_img
{
public:
  typedef F value_type;
private:
  boost::shared_array<F> data_;
  const int width_;
  const int height_;
  const int chans_;

public:
  jp_img(int width, int height, int chans = 1)
   : data_(new F[width*height*chans]), width_(width), height_(height), chans_(chans)
  { }

  template<class InType>
  jp_img(const InType& img)
   : data_(new F[img.width() * img.height() * img.chans()]), width_(img.width()), height_(img.height()), chans_(img.chans())
  {
    for (int y=0; y<height_; ++y) {
      for (int x=0; x<width_; ++x) {
        for (int c=0; c<chans_; ++c) {
          operator()(x,y,c) = (value_type)img(x,y,c);
        }
      }
    }
  }

  inline
  const F&
  operator()(int x, int y, int c = 0) const
  { return data_[chans_*(width_*y + x) + c]; }

  inline
  F&
  operator()(int x, int y, int c = 0)
  { return data_[chans_*(width_*y + x) + c]; }

  inline
  int
  width() const
  { return width_; }

  inline
  int
  height() const
  { return height_; }

  inline
  int
  chans() const
  { return chans_; }

  inline
  F*
  data()
  { return data_.get(); }
  
  inline
  const F*
  data() const
  { return data_.get(); }
};

template<class ImgFloat,
         class KerFloat>
void
jp_img_conv_buffer(ImgFloat* buffer, const KerFloat* kernel, int bsize, int ksize)
{
  for (int i=0; i<bsize; ++i) {
    KerFloat sum = KerFloat();
    for (int j=0; j<ksize; ++j)
      sum += (buffer[i+j]*kernel[j]);
    buffer[i] = sum;
  }
}

template<class OutType,
         class InType,
         class KerFloat>
OutType
jp_img_conv_horiz(const InType& img, const KerFloat* kernel, int ksize)
{
  typedef typename OutType::value_type value_type;

  value_type buffer[2000];
  int halfsize = ksize/2;

  OutType ret(img.width(), img.height(), img.chans());

  for (int c=0; c<img.chans(); ++c) {
    for (int y=0; y<img.height(); ++y) {
      for (int i=0; i<halfsize; ++i) {
        buffer[i] = img(0,y,c);
      }
      for (int i=0; i<img.width(); ++i) {
        buffer[halfsize + i] = img(i,y,c);
      }
      for (int i=0; i<halfsize; ++i) {
        buffer[img.width() + halfsize + i] = img(img.width()-1,y,c);
      }

      jp_img_conv_buffer(buffer, kernel, img.width(), ksize);

      for (int i=0; i<img.width(); ++i) {
        ret(i,y,c) = buffer[i];
      }
    }
  }
  return ret;
}

template<class OutType,
         class InType,
         class KerFloat>
OutType
jp_img_conv_vert(const InType& img, const KerFloat* kernel, int ksize)
{
  typedef typename OutType::value_type value_type;

  value_type buffer[2000];
  int halfsize = ksize/2;

  OutType ret(img.width(), img.height(), img.chans());

  for (int c=0; c<img.chans(); ++c) {
    for (int x=0; x<img.width(); ++x) {
      for (int i=0; i<halfsize; ++i) {
        buffer[i] = img(x,0,c);
      }
      for (int i=0; i<img.height(); ++i) {
        buffer[halfsize + i] = img(x,i,c);
      }
      for (int i=0; i<halfsize; ++i) {
        buffer[img.height() + halfsize + i] = img(x,img.height()-1,c);
      }

      jp_img_conv_buffer(buffer, kernel, img.height(), ksize);

      for (int i=0; i<img.height(); ++i) {
        ret(x,i,c) = buffer[i];
      }
    }
  }
  return ret;
}


template<class OutType,
         class InType,
         class Float>
OutType
jp_img_smooth(const InType& img, const Float& sigma)
{
  static const int trunc = 3;

  Float kernel[100];
  int ksize = (int)(2.0*trunc*sigma + 1.0);
  if (ksize % 2 == 0)
    ksize++;

  Float sum = Float();
  for (int i=0; i<ksize; ++i) {
    Float x = i - ksize/2;
    kernel[i] = std::exp(-x*x/(2.0*sigma*sigma));
    sum += kernel[i];
  }

  for (int i=0; i<ksize; ++i)
    kernel[i]/=sum;

  return jp_img_conv_vert<OutType>(jp_img_conv_horiz<OutType>(img, kernel, ksize), kernel, ksize);
}

template<class OutType,
         class InType>
OutType
jp_img_rgb2grey(const InType& img)
{
  OutType ret(img.width(), img.height(), 1);
  typedef typename OutType::value_type value_type;

  for (int y=0; y<img.height(); ++y) {
    for (int x=0; x<img.width(); ++x) {
      ret(x,y) = (img(x,y,0) + img(x,y,1) + img(x,y,2)) / (value_type)3;
    }
  }

  return ret;
}

template<class OutType,
         class InType>
std::pair<OutType, OutType>
jp_img_calc_x_y_derivative(const InType& img)
{
  OutType dx(img.width(), img.height(), 1);
  OutType dy(img.width(), img.height(), 1);

  int w = img.width(), h = img.height();

  for (int y=0; y<h; ++y) {
    for (int x=0; x<w; ++x) {
      if (x == 0)
        dx(x,y) = 2.0*(img(x+1,y) - img(x,y));
      else if (x == w-1)
        dx(x,y) = 2.0*(img(x,y) - img(x-1,y));
      else
        dx(x,y) = img(x+1,y) - img(x-1,y);

      if (y == 0)
        dy(x,y) = 2.0*(img(x,y+1) - img(x,y));
      else if (y == h-1)
        dy(x,y) = 2.0*(img(x,y) - img(x,y-1));
      else
        dy(x,y) = img(x,y+1) - img(x,y-1);
    }
  }

  return std::make_pair(dx, dy);
}

#if 0
#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_RGB_Image.H>
template<class InType>
void
jp_img_display(const InType& img)
{
  jp_img<unsigned char> img_uc = img;

  Fl_Window window(img.width()+10, img.height()+10);
  Fl_Box box(5, 5, img.width()+5, img.height()+5);
  Fl_RGB_Image fl_img(img_uc.data(), img.width(), img.height(), img.chans());
  box.image(fl_img);
  window.show();
  Fl::run();
}
#endif

#endif
