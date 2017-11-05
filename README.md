# Batch_Normalized_Maxout_NIN
Paper on arxiv: http://arxiv.org/abs/1511.02583

Paper on ICML visualization workshop: 

This is my implmentation of Batch Normalized Maxout NIN.

Original Matconvnet: https://github.com/vlfeat/matconvnet

This respository is my modification of Matconvnet.

You can install this modification as same as original installation:
http://www.vlfeat.org/matconvnet/install/
<blockquote>
vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', ...
'cudaRoot', your-cuda-toolkit\CUDA\v6.5', ...
'enableCudnn', true, 'cudnnRoot', 'your-cudnn-root\cuda') ;
</blockquote>

I used VS2013, CUDA-6.5 and cudnn-v4.

I added followwing functions:
<blockquote>
<ul>Maxout units (GPU supported only) </ul>
<ul>Data augmentations (horizontal flipping / pad zeros and random cropping)</ul>
</blockquote>

<h1>Usage:</h1>
##To run BN Maxout NIN
<blockquote>
After installation, run "\example\cifar\cnn_cifar.m"
</blockquote>

<h2>Use maxout units as pooling layers</h2>
for example: a batch normalized maxout layer consist of a convolutional layer, a BN layer, and a maxout layer

"unit1"  is the number of maxout units

"piece1" is the number of maxout pieces
<blockquote>
<p>net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'maxoutconv1', ...
                           'weights', {{single(orthonorm(1,1,unit0,unit1*piece1)), b*ones(1,unit1*piece1,'single')}}, ...
                           'stride', 1, ...
                           'learningRate', [.1 1], ...
                           'weightDecay', [1 0], ...
                           'pad', 0) ;</p>

<p>net.layers{end+1} = struct('type', 'bnorm', 'name', 'bn2', ...
                           'weights', {{ones(unit1*piece1, 1, 'single'), zeros(unit1*piece1, 1, 'single')}},'learningRate', [1 1 .5],'weightDecay', [0 0]) ;</p>   

<p>net.layers{end+1} = struct('type', 'maxout','numunit',unit1,'numpiece',piece1) ; </p>
</blockquote>

<h2>Data augmentations:</h2>
<blockquote>
<p>add following to your net opts</p>
<p>-> net.meta.trainOpts.augmentation= true;</p>
</blockquote>
####Using this implmentation, I achieved 8.13+-0.19% test error without augmentation in CIFAR-10 datasets.
####DATA preprocessing: GCN and Whitening.

