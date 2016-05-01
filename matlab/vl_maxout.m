%  function y = vl_maxout(x,numunit,numpiece,dzdy)
% %VL_maxout  CNN  maxout unit implemention
% %The maxout unit : 
% %Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013). Maxout networks. arXiv preprint arXiv:1302.4389.
% %
% %   Y =  vl_maxout(X,numunit,numpiece) applies the maxout unit to the data
% %   X. X can have arbitrary size.
% %   
% %
% %   DZDX =  vl_maxout(X,numunit,numpiece DZDY) computes the network derivative DZDX
% %   with respect to the input X given the derivative DZDY with respect
% %   to the output Y. DZDX has the same dimension as X.
% %
% %   2015 Jia Ren Chang (NCTU, Taiwan)
% %
%  if nargin <= 3 || isempty(dzdy)
%   x=gpuArray(x);  
%   sz = size(x); %4D
%   
%   %check input
%   if sz(3) ~= numunit*numpiece
%      fprintf('numunit*numpiece ~= inputsize');
%   end
%   feamap=gpuArray(zeros(sz(1),sz(2),numunit,sz(4),'single'));
%   %feamap=zeros(sz(1),sz(2),numunit,sz(4),'single');
%  
%   for i = 1:numunit
%        %seq = (i-1)*numpiece+1 : i*numpiece; 
%        feamap(:,:,i,:)= max(x(:,:,(i-1)*numpiece+1 : i*numpiece,:),[],3); 
%   end
%   %size(feamap)
%   y = feamap; 
%   clear x feamap;
% else
%  x=gpuArray(x);
%  dzdy=gpuArray(dzdy);
%  %th=0;
%   for i = 1:numunit 
%        seq = (i-1)*numpiece+1 : i*numpiece;
%        L=max(x(:,:,seq,:),[],3);
%        %for j = seq
%         mask =[];
%         mask = bsxfun(@eq,x(:,:,seq,:),L); 
%         %th=th+1;
%         error(:,:,seq,:) = bsxfun(@times,mask,dzdy(:,:,i,:));
%        %end  
%   end  
%   y =error;
%   clear x error
% end
% end