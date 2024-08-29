## 在Jetson Xavier NX中部署Snowboy语音唤醒模型
### 实训背景
在智能机器人的开发中，语音唤醒功能扮演着至关重要的基本角色。它不仅可以通过自然的人机交互提升用户体验，还能在无人值守的情况下实现设备的自动唤醒，从而显著减少功耗，延长待机时间，满足低功耗应用场景的需求。

语音唤醒技术的核心在于设备能够在低功耗模式下持续监测周围环境中的特定声音信号（如唤醒词），并在检测到唤醒词时快速从待机状态切换到工作模式。Jetson Xavier NX作为一款高性能的嵌入式AI计算平台，能够在保持低功耗的同时，执行复杂的语音处理任务，特别适合在边缘设备上实现实时语音唤醒功能。

Snowboy是一款轻量级的、适用于嵌入式设备的语音唤醒引擎，支持自定义唤醒词，并能够在资源受限的设备上高效运行。将Snowboy语音唤醒模型部署在Jetson Xavier NX上，可以充分利用其计算能力和节能特性，实现智能机器人在待机状态下的语音唤醒功能，提升人机交互性能。

### 实训内容

在本次实训中，我们将深入了解如何在Jetson Xavier NX平台上部署开源的Snowboy语音唤醒模型，创建自己的唤醒词，编写符合自己需求的唤醒脚本，掌握在嵌入式平台上实现语音唤醒功能的全过程。

### 实训目的及要求

#### 实训目的

1. 成功部署Snowboy语音唤醒模型 
> 我们将学习如何从GitHub下载Snowboy项目，并在Jetson Xavier NX平台上编译和运行。
2. 实现语音唤醒后的动作执行
> 在成功部署模型后，我们需要学习如何编写自定义脚本，实现语音唤醒后的自动化任务执行。无论是启动机器人、执行特定指令，还是触发其他系统操作，我们都将通过实际编程和测试，掌握如何将语音识别与其他功能模块结合，实现智能化控制
> 

#### 实训要求
1. 能熟练掌握在Jetson Xavier NX平台上配置Python开发环境，并成功安装Snowboy运行所需的所有依赖项。这包括使用APT和PIP安装相关库、配置音频设备、以及解决可能出现的依赖冲突问题。
2. 能够根据Jetson Xavier NX的硬件架构，正确修改并编译Snowboy项目的源代码，确保其能够在该平台上运行。能够通过运行提供的示例代码，验证项目的正确性，并确保基本语音唤醒功能的正常工作 
3. 能编写符合个性化需求的自定义唤醒脚本。参考提供的示例脚本，需根据实际应用场景，设计和实现唤醒后执行特定任务的逻辑，并确保在Jetson Xavier NX平台上能够稳定运行。

### 实验软硬件环境

> 硬件：Jetson Xavier NX开发板，USB麦克风
> 软件：
> - python 3.7
> - pyaudio
> - swig
> - libatlas-base
> - portaudio19

### 实训操作指南

1. 下载github项目
```shell
git clone https://github.com/Kitt-AI/snowboy.git
```
2. 安装依赖
先运行如下指令
```shell
sudo apt install libatlas-base-dev portaudio19-dev swig
```
安装完以上依赖之后，进入对应的虚拟环境，然后运行
```shell
pip install pyaudio
```
3. 编译Snowboy
> - 查看当前架构
> 使用`lscpu`即可查看当前架构，Jetson Xavier NX架构为aarch64
> - 修改编译文件中使用的架构
> 进入路径`snowboy/swig/Python3`，使用`gedit Makefile`打开文件，修改文件中51、53、55行，都修改为如下内容
> `SNOWBOYDETECTLIBFILE = $(TOPDIR)/lib/aarch64-ubuntu1604/libsnowboy-detect.a` 
> - 编译Snowboy，在当前路径下运行`make`
> 

4. 运行提供的案例
- 将编译好的库移动到样例文件夹
```shell
ysc@ysc-desktop:~/Desktop/snowboy/swig/Python3$ ls
Makefile          _snowboydetect.so       snowboy-detect-swig.i
snowboydetect.py  snowboy-detect-swig.cc  snowboy-detect-swig.o
ysc@ysc-desktop:~/Desktop/snowboy/swig/Python3$ cp snowboydetect.py _snowboydetect.so ../../examples/Python3/
```

- 修改库调用
>进入到样例文件夹下`cd ../../examples/Python3/`
>打开`snowboydetect.py`文件，将`from . import snowboydetect`修改为`import snowboydetect`
>运行demo
>
>```shell
>ysc@ysc-desktop:~/Desktop/snowboy/examples/Python3$ python demo.py resources/models/snowboy.umdl 
>```
>运行后，对着USB麦克风喊出`snowboy`，听到叮的声音，表示运行成功
>
5. 创建自定义唤醒词
在网站中训练唤醒词模型`https://snowboy.hahack.com/`，并保存到模型文件夹中

6. 将必要的文件移植到项目中
    必要的文件是`snowboydetect.py` `_snowboydetect.so` `snowboydecoder.py` `demo.py`
    还有一个文件夹`resources`里存放了模型，如果有自己的模型就不用复制这个

7. 编写符合自己需求的唤醒脚本
在demo.py的基础上进行修改，唤醒后会执行回调函数`interrupt_callback`，在该回调函数中可以编写自己需要的逻辑
### 参考资料
[CSDN](https://blog.csdn.net/qq_38844263/article/details/127143725)

[GitHub](https://github.com/Jadeble/sizutrain)