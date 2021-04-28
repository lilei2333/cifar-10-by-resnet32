# A cifar-10 project by using Resnet32


本项目是神经网络和深度学习课程的期中PJ。我使用Resnet32来跑cifar-10分类问题。
为了让其他人能够复现我的结果，我将训练好的模型放在了logs/checkpoint.pth处。


## Requirements
整个项目使用Python 3.6 和 PyTorch 1.7.1构建。可使用下面的命令安装所有依赖包:
```
pip install -r requirements.txt
```

## How to train
直接使用
```
python main.py
```
第一次运行时，会连网下载数据并放到data文件夹下，后面运行时不会再次下载。
默认的日志文件和tensorboard文件都在logs文件夹下面。若要使用tensorboard查看结果，可用
```
tensorboard --logdir=/logs/tensorboard
```

我用一张12-GB的Titan X GPU训练200个epoch，用时1.2小时。
