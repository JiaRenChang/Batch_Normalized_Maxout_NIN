function weight = orthonorm(h,w,fil1,fil2)
  initw = randn(h,w*fil1*fil2);
  [U,S,V] = svd(initw,'econ');
  
  if numel(V) == h*w*fil1*fil2
   weight = 1*reshape(V,h,w,fil1,fil2);
  else
   weight = 1*reshape(U,h,w,fil1,fil2);
  end
