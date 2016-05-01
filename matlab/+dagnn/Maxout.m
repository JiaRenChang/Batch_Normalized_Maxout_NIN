classdef Maxout < dagnn.Filter
  properties  
    numunit = 16
    numpiece = 2
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_maxout(inputs{1},self.numunit,self.numpiece) ;
%      fprintf('---------------maxout-----------------\n')
%      size(outputs{1})
%      fprintf('---------------------------------------\n')
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_maxout(inputs{1},self.numunit,self.numpiece, derOutputs{1}) ;
      derParams = {} ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes,self)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3)/self.numpiece;
    end
    
  end
end
