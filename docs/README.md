
<div align=center>
<img width="500" src="img/cover.png" alt="封面"/>
</div>

[本項目](https://tangshusen.me/Dive-into-DL-PyTorch)將[《動手學深度學習》](http://zh.d2l.ai/) 原書中MXNet代碼實現改為PyTorch實現。原書作者：阿斯頓·張、李沐、扎卡里 C. 立頓、亞歷山大 J. 斯莫拉以及其他社區貢獻者，GitHub地址：https://github.com/d2l-ai/d2l-zh

此書的[中](https://zh.d2l.ai/)[英](https://d2l.ai/)版本存在一些不同，針對此書英文版的PyTorch重構可參考[這個項目](https://github.com/dsgiitr/d2l-pytorch)。
There are some differences between the [Chinese](https://zh.d2l.ai/) and [English](https://d2l.ai/) versions of this book. For the PyTorch modifying of the English version, you can refer to [this repo](https://github.com/dsgiitr/d2l-pytorch).


## 簡介
本倉庫主要包含code和docs兩個文件夾（外加一些數據存放在data中）。其中code文件夾就是每章相關jupyter notebook代碼（基於PyTorch）；docs文件夾就是markdown格式的《動手學深度學習》書中的相關內容，然後利用[docsify](https://docsify.js.org/#/zh-cn/)將網頁文檔部署到GitHub Pages上，由於原書使用的是MXNet框架，所以docs內容可能與原書略有不同，但是整體內容是一樣的。歡迎對本項目做出貢獻或提出issue。

## 面向人群
本項目面向對深度學習感興趣，尤其是想使用PyTorch進行深度學習的童鞋。本項目並不要求你有任何深度學習或者機器學習的背景知識，你只需瞭解基礎的數學和編程，如基礎的線性代數、微分和概率，以及基礎的Python編程。

## 食用方法 
### 方法一
本倉庫包含一些latex公式，但github的markdown原生是不支持公式顯示的，而docs文件夾已經利用[docsify](https://docsify.js.org/#/zh-cn/)被部署到了GitHub Pages上，所以查看文檔最簡便的方法就是直接訪問[本項目網頁版](https://tangshusen.me/Dive-into-DL-PyTorch)。當然如果你還想跑一下運行相關代碼的話還是得把本項目clone下來，然後運行code文件夾下相關代碼。

### 方法二
你還可以在本地訪問文檔，先安裝`docsify-cli`工具:
``` shell
npm i docsify-cli -g
```
然後將本項目clone到本地:
``` shell
git clone https://github.com/ShusenTang/Dive-into-DL-PyTorch.git
cd Dive-into-DL-PyTorch
```
然後運行一個本地服務器，這樣就可以很方便的在`http://localhost:3000`實時訪問文檔網頁渲染效果。
``` shell
docsify serve docs
```

### 方法三
如果你不想安裝`docsify-cli`工具，甚至你的電腦上都沒有安裝`Node.js`，而出於某些原因你又想在本地瀏覽文檔，那麼你可以在`docker`容器中運行網頁服務。

首先將本項目clone到本地:
``` shell
git clone https://github.com/ShusenTang/Dive-into-DL-PyTorch.git
cd Dive-into-DL-PyTorch
```
之後使用如下命令創建一個名稱為「d2dl」的`docker`鏡像：
``` shell
docker build -t d2dl .
```
鏡像創建好後，運行如下命令創建一個新的容器：
``` shell
docker run -dp 3000:3000 d2dl
```
最後在瀏覽器中打開這個地址`http://localhost:3000/#/`，就能愉快地訪問文檔了。適合那些不想在電腦上裝太多工具的小夥伴。


## 目錄
* [簡介]()
* [閱讀指南](read_guide.md)
* [1. 深度學習簡介](chapter01_DL-intro/deep-learning-intro.md)
* 2\. 預備知識
   * [2.1 環境配置](chapter02_prerequisite/2.1_install.md)
   * [2.2 數據操作](chapter02_prerequisite/2.2_tensor.md)
   * [2.3 自動求梯度](chapter02_prerequisite/2.3_autograd.md)
* 3\. 深度學習基礎
   * [3.1 線性迴歸](chapter03_DL-basics/3.1_linear-regression.md)
   * [3.2 線性迴歸的從零開始實現](chapter03_DL-basics/3.2_linear-regression-scratch.md)
   * [3.3 線性迴歸的簡潔實現](chapter03_DL-basics/3.3_linear-regression-pytorch.md)
   * [3.4 softmax迴歸](chapter03_DL-basics/3.4_softmax-regression.md)
   * [3.5 圖像分類數據集（Fashion-MNIST）](chapter03_DL-basics/3.5_fashion-mnist.md)
   * [3.6 softmax迴歸的從零開始實現](chapter03_DL-basics/3.6_softmax-regression-scratch.md)
   * [3.7 softmax迴歸的簡潔實現](chapter03_DL-basics/3.7_softmax-regression-pytorch.md)
   * [3.8 多層感知機](chapter03_DL-basics/3.8_mlp.md)
   * [3.9 多層感知機的從零開始實現](chapter03_DL-basics/3.9_mlp-scratch.md)
   * [3.10 多層感知機的簡潔實現](chapter03_DL-basics/3.10_mlp-pytorch.md)
   * [3.11 模型選擇、欠擬合和過擬合](chapter03_DL-basics/3.11_underfit-overfit.md)
   * [3.12 權重衰減](chapter03_DL-basics/3.12_weight-decay.md)
   * [3.13 丟棄法](chapter03_DL-basics/3.13_dropout.md)
   * [3.14 正向傳播、反向傳播和計算圖](chapter03_DL-basics/3.14_backprop.md)
   * [3.15 數值穩定性和模型初始化](chapter03_DL-basics/3.15_numerical-stability-and-init.md)
   * [3.16 實戰Kaggle比賽：房價預測](chapter03_DL-basics/3.16_kaggle-house-price.md)
* 4\. 深度學習計算
   * [4.1 模型構造](chapter04_DL_computation/4.1_model-construction.md)
   * [4.2 模型參數的訪問、初始化和共享](chapter04_DL_computation/4.2_parameters.md)
   * [4.3 模型參數的延後初始化](chapter04_DL_computation/4.3_deferred-init.md)
   * [4.4 自定義層](chapter04_DL_computation/4.4_custom-layer.md)
   * [4.5 讀取和存儲](chapter04_DL_computation/4.5_read-write.md)
   * [4.6 GPU計算](chapter04_DL_computation/4.6_use-gpu.md)
* 5\. 卷積神經網絡
   * [5.1 二維卷積層](chapter05_CNN/5.1_conv-layer.md)
   * [5.2 填充和步幅](chapter05_CNN/5.2_padding-and-strides.md)
   * [5.3 多輸入通道和多輸出通道](chapter05_CNN/5.3_channels.md)
   * [5.4 池化層](chapter05_CNN/5.4_pooling.md)
   * [5.5 卷積神經網絡（LeNet）](chapter05_CNN/5.5_lenet.md)
   * [5.6 深度卷積神經網絡（AlexNet）](chapter05_CNN/5.6_alexnet.md)
   * [5.7 使用重複元素的網絡（VGG）](chapter05_CNN/5.7_vgg.md)
   * [5.8 網絡中的網絡（NiN）](chapter05_CNN/5.8_nin.md)
   * [5.9 含並行連結的網絡（GoogLeNet）](chapter05_CNN/5.9_googlenet.md)
   * [5.10 批量歸一化](chapter05_CNN/5.10_batch-norm.md)
   * [5.11 殘差網絡（ResNet）](chapter05_CNN/5.11_resnet.md)
   * [5.12 稠密連接網絡（DenseNet）](chapter05_CNN/5.12_densenet.md)
* 6\. 循環神經網絡
   * [6.1 語言模型](chapter06_RNN/6.1_lang-model.md)
   * [6.2 循環神經網絡](chapter06_RNN/6.2_rnn.md)
   * [6.3 語言模型數據集（周杰倫專輯歌詞）](chapter06_RNN/6.3_lang-model-dataset.md)
   * [6.4 循環神經網絡的從零開始實現](chapter06_RNN/6.4_rnn-scratch.md)
   * [6.5 循環神經網絡的簡潔實現](chapter06_RNN/6.5_rnn-pytorch.md)
   * [6.6 通過時間反向傳播](chapter06_RNN/6.6_bptt.md)
   * [6.7 門控循環單元（GRU）](chapter06_RNN/6.7_gru.md)
   * [6.8 長短期記憶（LSTM）](chapter06_RNN/6.8_lstm.md)
   * [6.9 深度循環神經網絡](chapter06_RNN/6.9_deep-rnn.md)
   * [6.10 雙向循環神經網絡](chapter06_RNN/6.10_bi-rnn.md)
* 7\. 優化算法
   * [7.1 優化與深度學習](chapter07_optimization/7.1_optimization-intro.md)
   * [7.2 梯度下降和隨機梯度下降](chapter07_optimization/7.2_gd-sgd.md)
   * [7.3 小批量隨機梯度下降](chapter07_optimization/7.3_minibatch-sgd.md)
   * [7.4 動量法](chapter07_optimization/7.4_momentum.md)
   * [7.5 AdaGrad算法](chapter07_optimization/7.5_adagrad.md)
   * [7.6 RMSProp算法](chapter07_optimization/7.6_rmsprop.md)
   * [7.7 AdaDelta算法](chapter07_optimization/7.7_adadelta.md)
   * [7.8 Adam算法](chapter07_optimization/7.8_adam.md)
* 8\. 計算性能
   * [8.1 命令式和符號式混合編程](chapter08_computational-performance/8.1_hybridize.md)
   * [8.2 異步計算](chapter08_computational-performance/8.2_async-computation.md)
   * [8.3 自動並行計算](chapter08_computational-performance/8.3_auto-parallelism.md)
   * [8.4 多GPU計算](chapter08_computational-performance/8.4_multiple-gpus.md)
* 9\. 計算機視覺
   * [9.1 圖像增廣](chapter09_computer-vision/9.1_image-augmentation.md)
   * [9.2 微調](chapter09_computer-vision/9.2_fine-tuning.md)
   * [9.3 目標檢測和邊界框](chapter09_computer-vision/9.3_bounding-box.md)
   * [9.4 錨框](chapter09_computer-vision/9.4_anchor.md)
   * [9.5 多尺度目標檢測](chapter09_computer-vision/9.5_multiscale-object-detection.md)
   * [9.6 目標檢測數據集（皮卡丘）](chapter09_computer-vision/9.6_object-detection-dataset.md)
   - [ ] 9.7 單發多框檢測（SSD）
   * [9.8 區域卷積神經網絡（R-CNN）系列](chapter09_computer-vision/9.8_rcnn.md)
   * [9.9 語義分割和數據集](chapter09_computer-vision/9.9_semantic-segmentation-and-dataset.md)
   - [ ] 9.10 全卷積網絡（FCN）
   * [9.11 樣式遷移](chapter09_computer-vision/9.11_neural-style.md)
   - [ ] 9.12 實戰Kaggle比賽：圖像分類（CIFAR-10）
   - [ ] 9.13 實戰Kaggle比賽：狗的品種識別（ImageNet Dogs）
* 10\. 自然語言處理
   * [10.1 詞嵌入（word2vec）](chapter10_natural-language-processing/10.1_word2vec.md)
   * [10.2 近似訓練](chapter10_natural-language-processing/10.2_approx-training.md)
   * [10.3 word2vec的實現](chapter10_natural-language-processing/10.3_word2vec-pytorch.md)
   * [10.4 子詞嵌入（fastText）](chapter10_natural-language-processing/10.4_fasttext.md)
   * [10.5 全局向量的詞嵌入（GloVe）](chapter10_natural-language-processing/10.5_glove.md)
   * [10.6 求近義詞和類比詞](chapter10_natural-language-processing/10.6_similarity-analogy.md)
   * [10.7 文本情感分類：使用循環神經網絡](chapter10_natural-language-processing/10.7_sentiment-analysis-rnn.md)
   * [10.8 文本情感分類：使用卷積神經網絡（textCNN）](chapter10_natural-language-processing/10.8_sentiment-analysis-cnn.md)
   * [10.9 編碼器—解碼器（seq2seq）](chapter10_natural-language-processing/10.9_seq2seq.md)
   * [10.10 束搜索](chapter10_natural-language-processing/10.10_beam-search.md)
   * [10.11 注意力機制](chapter10_natural-language-processing/10.11_attention.md)
   * [10.12 機器翻譯](chapter10_natural-language-processing/10.12_machine-translation.md)



持續更新中......




## 原書地址
中文版：[動手學深度學習](https://zh.d2l.ai/) | [Github倉庫](https://github.com/d2l-ai/d2l-zh)       
English Version: [Dive into Deep Learning](https://d2l.ai/) | [Github Repo](https://github.com/d2l-ai/d2l-en)


## 引用
如果您在研究中使用了這個項目請引用原書:
```
@book{zhang2019dive,
    title={Dive into Deep Learning},
    author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
    note={\url{http://www.d2l.ai}},
    year={2020}
}
```
