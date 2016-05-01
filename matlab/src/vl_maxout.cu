// @file vl_nnpool.cu
// @brief Pooling block MEX wrapper
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-15 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnmaxout.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT=3, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int numunit ;
  int numpiece ;
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;


  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 2) { mexErrMsgTxt("The arguments are less than two.") ;}

  if (nin < 4) {
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  if (backMode) { derOutput.init(in[IN_DEROUTPUT]) ; }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }

      numunit =  mxGetPr(in[1])[0] ;
      numpiece =  mxGetPr(in[2])[0] ;


  /* Basic compatibility of geometry */
   //mexPrintf("input data: %d ",data[0]);
 // mexPrintf("numpiece:%d ",numpiece);
  //mexPrintf("data depth:%d ",data.getDepth());
  if (numunit * numpiece != data.getDepth()) {
    mexErrMsgTxt("hidden unit not equal to maxout layer.") ;
  }

  /* Get the output geometry */
  vl::TensorGeometry outputGeom(data.getHeight(),
                                data.getWidth(),
                                numunit,
                                data.getSize()) ;
   //mexPrintf("input data: %d %d %d %d",data.getHeight(), data.getWidth(),data.getDepth(),data.getSize());

  if (backMode && (derOutput != outputGeom)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::Device type = data.getMemoryType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;
  vl::MexTensor derFilters(context) ;
  vl::MexTensor derBiases(context) ;

  if (!backMode) {
    output.init(type, outputGeom, 0) ;
  } else {
    derData.init(type, data.getGeometry(), 0) ;
  }

  
  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::Error error ;
  if (!backMode) {
    error = vl::nnmaxout_forward(context,
                                  output, data,
                                  numunit, numpiece);
  } else {
    error = vl::nnmaxout_backward(context,
                                   derData, data, derOutput,
								   numunit, numpiece);
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::vlSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
