# Batch_Normalized_Maxout_NIN
Paper on arxiv: http://arxiv.org/abs/1511.02583

This is my implmentation of Batch Normalized Maxout NIN.

Original Matconvnet: https://github.com/vlfeat/matconvnet

This respository is my modification of Matconvnet.
You can install this modification as same as original installation:
http://www.vlfeat.org/matconvnet/install/
I used VS2013, CUDA-6.5 and cudnn-v4.

I added followwing functions:
Maxout units (GPU supported only) 
Data augmentations (horizontal flipping / pad zeros and random cropping)

<h1>Usage:</h1>
<h2>use maxout units as pooling layers</h2>
for example: a batch normalized maxout layer consist of a convolutional layer, a BN layer, and a maxout layer

"unit1"  is the number of maxout units
"piece1" is the number of maxout pieces

net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'maxoutconv1', ...
                           'weights', {{single(orthonorm(1,1,unit0,unit1*piece1)), b*ones(1,unit1*piece1,'single')}}, ...
                           'stride', 1, ...
                           'learningRate', [.1 1], ...
                           'weightDecay', [1 0], ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'bnorm', 'name', 'bn2', ...
                           'weights', {{ones(unit1*piece1, 1, 'single'), zeros(unit1*piece1, 1, 'single')}},'learningRate', [1 1 .5],'weightDecay', [0 0]) ;   

net.layers{end+1} = struct('type', 'maxout','numunit',unit1,'numpiece',piece1) ; 


<h2>Data augmentations:</h2>
<p>add following to your net opts</p>
<p>-> net.meta.trainOpts.augmentation= true;</p>

####Using this implmentation, I achieved 8.13+-0.19% test error without augmentation in CIFAR-10 datasets.
####DATA preprocessing: GCN and Whitening.

