
## Stage 1

* `stage1\generate_train_test.py`: 扩增人脸，生成训练集和测试集
* `stage1\data_pil.py`: PIL读取图片，灰度图
* `stage1\data_cv2.py`: cv2读取数据，彩色图
* `stage1\detector_t.py`: 网络搭建

### 任务一: 生成 train/test.txt

`stage1\gen_train_test.py`

### 任务二：网络搭建

* 数据在网络中的维度顺序是什么?

N x C x H x W。N：个数，C：通道数， H：高， W：宽

* nn.Conv2d()中参数含义与顺序?

`in_channels` (int): Number of channels in the input image <br>
`out_channels` (int): Number of channels produced by the convolution <br>
`kernel_size` (int or tuple): Size of the convolving kernel <br>
`stride` (int or tuple, optional): Stride of the convolution. Default: 1 <br>
`padding` (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0 <br>
`padding_mode` (string, optional). Accepted values `zeros` and `circular` Default: `zeros`<br>
`dilation` (int or tuple, optional): Spacing between kernel elements. Default: 1<br>
`groups` (int, optional): Number of blocked connections from input channels to output channels. Default: 1<br>
`bias` (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``<br>

* nn.Linear()是什么意思?参数含义与顺序?

线形变换<br>
`in_features`: size of each input sample<br>
`out_features`: size of each output sample<br>
`bias`: If set to ``False``, the layer will not learn an additive bias. Default: ``True``<br>

* nn.PReLU()与 nn.ReLU()的区别?示例中定义了很多 nn.PReLU()，能否只定义一个
PReLU?

<img src="https://render.githubusercontent.com/render/math?math=\text{ReLU}(x)= \max(0, x)">
<img src="https://render.githubusercontent.com/render/math?math=%5Ctext%7BPReLU%7D(x)%20%3D%20%5Cmax(0%2Cx)%20%20%2B%20%20a%20*%20%5Cmin(0%2Cx)">

PReLU和ReLU的区别在于，x小于0时的取值，ReLU为0，PReLU为ax，其中a是一个可以学习的参数。
可以只定义一个PReLU函数，但为了forward计算一步到位，我们一般定义多个激活函数。

* nn.AvgPool2d()中参数含义?还有什么常用的 pooling 方式?

`kernel_size`: the size of the window
<br>`stride`: the stride of the window. Default value is :attr:`kernel_size`
<br>`padding`: implicit zero padding to be added on both sides
<br>`ceil_mode`: when True, will use `ceil` instead of `floor` to compute the output shape
<br>`count_include_pad`: when True, will include the zero-padding in the averaging calculation
<br>`divisor_override`: if specified, it will be used as divisor, otherwise attr:`kernel_size` will be used

常用的还有max pooling, power-average pooling, adaptive average pooling

* view()的作用?

flatten降维，把数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor


### 任务五：Train部分

* `optimizer.zero()`与 `optimizer.step()`的作用是什么？<br>

`ptimizer.zero()`每次计算新的grad时，要把原来的梯度清0。optimizer.zero_grad()可以自动完成这个操作，把所有Variable的grad成员数值变为0<br>
`optimizer.step()`是在每个Variable的grad都被计算出来后，更新每个Variable的数值。<br>

* `model.eval()`产生的效果?<br>

`model.train()`启用BatchNormalization和Dropout，而`model.eval()`用于验证时，不启用BatchNormalization和Dropout。<br>

* `model.state_dict()`的目的是？<br>

将每一层与它的对应参数建立映射关系。<br>

* 何时系统自动进行bp？<br>

调用loss.backward()后并且Tensor的requires_grad为True。<br>

* 如果自己的层需要bp，如何实现？如何调用？<br>

通过设置requires_grad参数，训练需要bp的层而冻结其他层。<br>