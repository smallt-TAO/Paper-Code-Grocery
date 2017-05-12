# Review-for-Paper

# ML论文翻译
http://www.jianshu.com/nb/8413272

# metric learning 的总结
http://blog.csdn.net/langb2014/article/details/53036216

# TensorFlow安装指南（Centos 7&Windows）及DCGAN demo测试
http://blog.csdn.net/yexiaogu1104/article/details/69055802

# Tensorflow一些常用基本概念与函数（4）
http://blog.csdn.net/lenbow/article/details/52218551

# gensim的LDA
http://nbviewer.jupyter.org/gist/boskaiolo/cc3e1341f59bfbd02726

# cython的window技术
https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

# 目标检测方面的技术发展。
https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html

# 你应该知道的9篇深度学习论文（CNNs 理解）
http://blog.csdn.net/feimengjuan/article/details/69666981

# cs231n学习笔记-CNN-目标检测、定位、分割
http://blog.csdn.net/myarrow/article/details/51878004

# GAN论文汇总，包含code
https://github.com/zhangqianhui/AdversarialNetsPapers



本周工作：

   图像方面：
      1.GAN方向：
         a）利用GAN处理图像的纹理的论文，《Precomputed real-time texture synthesis with markovian generative adversarial networks》。该模型利用及其复杂的对抗信息，三次使用VGG网络生成图像的纹理信息。
         b）同样的一篇利用GAN处理图像信息《Semantic Segmentation using Adversarial Networks》。与上文的策略相识。
      2.目标识别：
         a）faster-rcnn。原理，代码，修改参数。
         b）SSD。原理，代码。对特性任务效果差，抛弃。
         c）mask-rcnn。原理，代码，更换训练数据，训练中。
         d）来至google最新的技术：attention-ocr。利用CNN，LSTM，CTC，mask设计的识别街道上的商家名称。熟悉代码，跑通demo。准备修改代码，适应新的需求。
   NLP：
       1.SC-lstm：
          a）一篇emnlp的best paper《SemanticallyConditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems》。对lstm添加DA的reading gate实现更好的人机对话。
          b）实现代码，并在github上提交代码提高原始代码的鲁棒性。简单来讲就是使用更高版本的tensorflow。
          c）对原始的结果有改进的想法。TODO。
       2.word2vec：
          a）有好的想法，调试代码中。计划使用C++调试，为了最终模型训练速度。
