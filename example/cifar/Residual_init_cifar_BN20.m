function net = Residual_init_cifar_BN20(varargin)
opts.networkType = 'dagnn' ;
opts = vl_argparse(opts, varargin) ;

import dagnn.*
net = DagNN() ;

convblock = Conv();  convblock.size = [3 3 3 16];        convblock.hasBias=true;     convblock.pad=1;  convblock.stride=1; %32
net.addLayer('Gconv1',convblock,{'input'},{'x1'},{'Gconv1f','Gconv1b'});
net.params(end-1).value = sqrt(2/(9*16))*randn(3,3,3,16,'single');
net.params(end).value = zeros(1,16,'single');
net.params(end-1).learningRate = .1;
net.params(end).learningRate = .1;
net.params(end-1).weightDecay = 1;
net.params(end).weightDecay = 0;

bnblock = BatchNorm() ;
net.addLayer('Gbn1',bnblock,{'x1'},{'x2'},{'Gbn1a','Gbn1b','Gbn1mom'});
net.params(end-2).value = ones(16,1,'single') ;
net.params(end-1).value = zeros(16,1,'single') ;
net.params(end).value = zeros(16,2,'single') ;
net.params(end-2).learningRate = 1 ;
net.params(end-1).learningRate = 1;
net.params(end).learningRate = .5 ;
net.params(end-2).weightDecay = 0;
net.params(end-1).weightDecay = 0;
net.params(end).weightDecay = 0 ;

actblock = ReLU() ;
net.addLayer('Grelu1',actblock,{'x2'},{'x3'},{});

way1 = 3;


%%

net = add_block(net, 1001, way1, way1, way1 ,3, 3, 16, 16, 1, 1) ; %
way1 = way1+7*1;

for idl = 1:2
net = add_block(net, idl  , way1, way1,way1, 3, 3, 16, 16, 1, 1) ; %
way1 = way1+7*1;
end

%
startway1 = way1;

%% trasnform layer

convblock = Conv(); convblock.size = [1 1 16 32];   convblock.hasBias=true;    convblock.pad=0;    convblock.stride=2; %16
net.addLayer('tranconv1',convblock,{sprintf('x%d',startway1)},{sprintf('x%d',startway1+1)},{'tranconv1f'});
net.params(end).value = sqrt(2/(9*32))*randn(1,1,16,32,'single');
net.params(end-1).learningRate = .1;
net.params(end).learningRate = .1;
net.params(end-1).weightDecay = 1;
net.params(end).weightDecay = 0;

bnblock = BatchNorm() ;
net.addLayer('tranbn1',bnblock,{sprintf('x%d',startway1+1)},{sprintf('x%d',startway1+2)},{'tranbn1f','tranbn1b','tranbn1mom'});
net.params(end-2).value = ones(32,1,'single') ;
net.params(end-1).value = zeros(32,1,'single') ;
net.params(end).value = zeros(32,2,'single') ;
net.params(end-2).learningRate = 1 ;
net.params(end-1).learningRate = 1;
net.params(end).learningRate = .5 ;
net.params(end-2).weightDecay = 0;
net.params(end-1).weightDecay = 0;
net.params(end).weightDecay = 0 ;

way1 = startway1+2;

net = add_block(net, 1003, way1, startway1 ,way1 ,3, 3, 16, 32, 2, 1) ; %startid 9 output 17
way1 = way1+7*1;

for idl = 3:4
    net = add_block(net, idl  , way1 , way1 ,way1,3, 3, 32, 32, 1, 1) ; % o -> 30
    way1 = way1+7*1;
end

startway1 = way1; 

%% trasnform layer
convblock = Conv(); convblock.size = [1 1 32 64];   convblock.hasBias=true;    convblock.pad=0;    convblock.stride=2; %16
net.addLayer('tranconv1_2',convblock,{sprintf('x%d',startway1)},{sprintf('x%d',startway1+1)},{'tranconv1f_2'});
net.params(end).value = sqrt(2/(9*64))*randn(1,1,32,64,'single');
net.params(end-1).learningRate = .1;
net.params(end).learningRate = .1;
net.params(end-1).weightDecay = 1;
net.params(end).weightDecay = 0;

bnblock = BatchNorm() ;
net.addLayer('tranbn1_2',bnblock,{sprintf('x%d',startway1+1)},{sprintf('x%d',startway1+2)},{'tranbn1f_2','tranbn1b_2','tranbn1mom_2'});
net.params(end-2).value = ones(64,1,'single') ;
net.params(end-1).value = zeros(64,1,'single') ;
net.params(end).value = zeros(64,2,'single') ;
net.params(end-2).learningRate = 1 ;
net.params(end-1).learningRate = 1;
net.params(end).learningRate = .5 ;
net.params(end-2).weightDecay = 0;
net.params(end-1).weightDecay = 0;
net.params(end).weightDecay = 0 ;

way1 = startway1+2;

net = add_block(net, 1005, way1, startway1,way1,3, 3, 32, 64, 2, 1) ; %startid 9 output 17
way1 = way1+7*1;

for idl = 5:6
net = add_block(net, idl  , way1, way1,way1,3, 3, 64, 64, 1, 1) ; % o -> 30
way1 = way1+7*1;
end

%%
outway = way1;

poolbock = Pooling('method','avg','poolSize',[8 8],'pad',0,'stride',8); %8
net.addLayer('pool1',poolbock,{sprintf('x%d',outway)},{sprintf('x%d',outway+1)},{});

%% ending layer
convblock = Conv(); convblock.size = [1 1 64 10];   convblock.hasBias=true;    convblock.pad=0;    convblock.stride=1; %
net.addLayer('end_conv1',convblock,{sprintf('x%d',outway+1)},{sprintf('x%d',outway+2)},{'end_conv1f','end_conv1b'});
net.params(end-1).value = sqrt(2/(1*10))*randn(1,1,64,10,'single');
net.params(end).value = zeros(1,10,'single');
net.params(end-1).learningRate = .1;
net.params(end).learningRate = .1;
net.params(end-1).weightDecay = 1;
net.params(end).weightDecay = 0;

bnblock = BatchNorm() ;
net.addLayer('end_bn1',bnblock,{sprintf('x%d',outway+2)},{'prediction'},{'end_bn1f','end_bn1b','end_bn1mom'});
net.params(end-2).value = ones(10,1,'single') ;
net.params(end-1).value = zeros(10,1,'single') ;
net.params(end).value = zeros(10,2,'single') ;
net.params(end-2).learningRate = 1 ;
net.params(end-1).learningRate = 1;
net.params(end).learningRate = .5 ;
net.params(end-2).weightDecay = 0;
net.params(end-1).weightDecay = 0;
net.params(end).weightDecay = 0 ;



%%
      softmaxblock = Loss('loss', 'softmaxlog') ;
      net.addLayer('softmaxloss',softmaxblock,{'prediction','label'},{'objective'},{});
  
%%

net.meta.trainOpts.learningRate = [ones(1,80) 0.1*ones(1,42) 0.01*ones(1,42)];
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 128;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.trainOpts.augmentation= true;
net.meta.trainOpts.momentum = 0.9;


% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
      {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end


function net = add_block(net, id , sumid, inputid,startid , h, w, in, out, stride, pad)
import dagnn.*

convblock = Conv('size',[h w in out],'hasBias',true,'pad',pad,'stride',stride);
net.addLayer(sprintf('res_Conv%d', id),convblock,{sprintf('x%d',inputid)},{sprintf('x%d',startid+1)},{sprintf('resf%d',id),sprintf('resb%d',id)});
net.params(end-1).value = sqrt(2/(h*h*out))*randn(h,w,in,out,'single');
net.params(end).value = zeros(1,out,'single');
net.params(end-1).learningRate = .1;
net.params(end).learningRate = .1;
net.params(end-1).weightDecay = 1;
net.params(end).weightDecay = 0;

bnblock = BatchNorm() ;
net.addLayer(sprintf('res_bn%d',id),bnblock,{sprintf('x%d',startid+1)},{sprintf('x%d',startid+2)},{sprintf('bnf%d',id),sprintf('bnb%d',id),sprintf('bnmom%d',id)});
net.params(end-2).value = ones(out,1,'single') ;
net.params(end-1).value = zeros(out,1,'single') ;
net.params(end).value = zeros(out,2,'single') ;
net.params(end-2).learningRate = 1 ;
net.params(end-1).learningRate = 1;
net.params(end).learningRate = .5 ;
net.params(end-2).weightDecay = 0;
net.params(end-1).weightDecay = 0;
net.params(end).weightDecay = 0 ;

actblock = ReLU() ;
net.addLayer(sprintf('res_relu%d',id),actblock,{sprintf('x%d',startid+2)},{sprintf('x%d',startid+3)},{});

convblock = Conv('size',[h w out out],'hasBias',true,'pad',pad,'stride',1);
net.addLayer(sprintf('res_Conv2_%d', id),convblock,{sprintf('x%d',startid+3)},{sprintf('x%d',startid+4)},{sprintf('resf_2_%d',id),sprintf('resb_2_%d',id)});
net.params(end-1).value = sqrt(2/(h*h*out))*randn(h,w,out,out,'single');
net.params(end).value = zeros(1,out,'single');
net.params(end-1).learningRate = .1;
net.params(end).learningRate = .1;
net.params(end-1).weightDecay = 1;
net.params(end).weightDecay = 0;

bnblock = BatchNorm() ;
net.addLayer(sprintf('res_bn2_%d',id),bnblock,{sprintf('x%d',startid+4)},{sprintf('x%d',startid+5)},{sprintf('bnf_2_%d',id),sprintf('bnb_2_%d',id),sprintf('bnmom_2_%d',id)});
net.params(end-2).value = ones(out,1,'single') ;
net.params(end-1).value = zeros(out,1,'single') ;
net.params(end).value = zeros(out,2,'single') ;
net.params(end-2).learningRate = 1 ;
net.params(end-1).learningRate = 1;
net.params(end).learningRate = .5 ;
net.params(end-2).weightDecay = 0;
net.params(end-1).weightDecay = 0;
net.params(end).weightDecay = 0 ;

sumblock = Sum();
net.addLayer(sprintf('res_sum%d',id),sumblock,{sprintf('x%d',sumid),sprintf('x%d',startid+5)},{sprintf('x%d',startid+6)},{});

actblock = ReLU() ;
net.addLayer(sprintf('res_relu2_%d',id),actblock,{sprintf('x%d',startid+6)},{sprintf('x%d',startid+7)},{});