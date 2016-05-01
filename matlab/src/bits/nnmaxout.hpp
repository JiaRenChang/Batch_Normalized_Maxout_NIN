// @file nnpooling.hpp
// @brief Pooling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnmaxout__
#define __vl__nnmaxout__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  vl::Error
  nnmaxout_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor data,
                    int numunit, int numpiece) ;

  vl::Error
  nnmaxout_backward(vl::Context& context,
                     vl::Tensor derData,
                     vl::Tensor data,
                     vl::Tensor derOutput,
					 int numunit, int numpiece);
}

#endif /* defined(__vl__nnpooling__) */
