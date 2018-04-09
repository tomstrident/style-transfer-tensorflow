# Style Transfer (Tensorflow Implementation)

An implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in tensorflow.

## Requirements:
* Tensorflow
* numpy, scipy
* [VGG19 model](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

A demo file `style_transfer_demo.py` is provided.

## Reference Images
Settings: `alpha=1.0, beta=1.0, num_iter=1000`
<p align="center">
<img src="Images/Tuebingen_Neckarfront.jpg" height="192px">
<img src="Images/vangogh_starry_night.jpg" height="192px">
</p>

<p align="center">
<img src="Output/sample.png" height="192px">
</p>

## Example Images
You can find the source images in `/content-images/` and `/style-images/`.
<p align="center">
<img src="content-images/hans.jpg" height="192px">
<img src="example-images/hans5.png" height="192px">
<img src="example-images/hans6.png" height="192px">
</p>
<p align="center">
<img src="example-images/hans7.png" height="192px">
<img src="example-images/hans8.png" height="192px">
<img src="example-images/hans9.png" height="192px">
</p>
<p align="center">
<img src="example-images/hans10.png" height="192px">
<img src="example-images/hans11.png" height="192px">
<img src="example-images/hans12.png" height="192px">
</p>

Tested on Win10 and Ubuntu 16.04
