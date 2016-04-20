# 2016_Art_Style
A Neural Algorithm of Artistic style arXiv:1508.06576v2

I include train and test code in master branch.
The model 'vgg-verydeep-19' is also included in /data.
(Your GPU memory should >= 4GB)

<img width="540" alt="f7bdb46c-80d4-45c3-a70f-8c2ba1c13472" src="https://cloud.githubusercontent.com/assets/8390471/14660630/a353b504-06d9-11e6-899c-a811413a2b53.png">

# Result
![](https://github.com/layumi/2016_Artist_Style/blob/master/4.jpg)
 
![](https://github.com/layumi/2016_Artist_Style/blob/master/1.jpg) 
![](https://github.com/layumi/2016_Artist_Style/blob/master/demo.jpg) 


# How to train & test
1.You may compile matconvnet first by running gpu_compile.m  (you need to change some setting in it)

For more compile information, you can learn it from www.vlfeat.org/matconvnet/install/#compiling

2.Use test_vgg19.m to have fun~
