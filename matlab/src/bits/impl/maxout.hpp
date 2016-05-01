// @file pooling.hpp
// @brief Pooling block implementation
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef VL_maxout_H
#define VL_maxout_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

	/* Max pooling */

	template<vl::Device dev, typename type> vl::Error
		maxout_forward(type* pooled,
		type const* data,
		size_t height, size_t width, size_t depth,
		size_t numunit, size_t numpiece);


	template<vl::Device dev, typename type> vl::Error
		maxout_backward(type* derData,
		type const* data,
		type const* derPooled,
		size_t height, size_t width, size_t depth,
		size_t numunit, size_t numpiece);



#if ENABLE_GPU
  template<> vl::Error
  maxout_forward<vl::GPU, float>(float* pooled,
                                      float const* data,
                                      size_t height, size_t width, size_t depth,
									  size_t numunit, size_t numpiece);

  template<> vl::Error
  maxout_backward<vl::GPU, float>(float* derData,
                                       float const* data,
                                       float const* derPooled,
                                       size_t height, size_t width, size_t depth,
									   size_t numunit, size_t numpiece);

#endif

} }

#endif /* defined(VL_NNPOOLING_H) */
