
## 问题回答

### 任务一: 生成 train/test.txt

* `gen_train_test.py`: 扩增人脸，生成训练集和测试集

### 任务二：网络搭建

* 数据在网络中的维度顺序是什么?

N x C x H x W。N：个数，C：通道数， H：高， W：宽

* nn.Conv2d()中参数含义与顺序?

`in_channels` (int): Number of channels in the input image 
`out_channels` (int): Number of channels produced by the convolution 
`kernel_size` (int or tuple): Size of the convolving kernel 
`stride` (int or tuple, optional): Stride of the convolution. Default: 1 
`padding` (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0 
`padding_mode` (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
`dilation` (int or tuple, optional): Spacing between kernel elements. Default: 1
`groups` (int, optional): Number of blocked connections from input channels to output channels. Default: 1
`bias` (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

* nn.Linear()是什么意思?参数含义与顺序?

线形变换

`in_features`: size of each input sample
`out_features`: size of each output sample
`bias`: If set to ``False``, the layer will not learn an additive bias. Default: ``True``

* nn.PReLU()与 nn.ReLU()的区别?示例中定义了很多 nn.PReLU()，能否只定义一个
PReLU?

<img src="https://render.githubusercontent.com/render/math?math=\text{ReLU}(x)= \max(0, x)">
<img src="https://render.githubusercontent.com/render/math?math=\text{PReLU}(x) = \max(0,x) + a * \min(0,x)">



* nn.AvgPool2d()中参数含义?还有什么常用的 pooling 方式?
* view()的作用?