---
title: 表情识别
data: 2024-4-20
tags:
    - Python
    - Tensorflow
categories:
    - 学习
    - Python
top_img: 'linear-gradient(20deg, #0062be, #925696, #cc426e, #fb0347)'
cover: https://s2.loli.net/2024/02/14/yJ4w1SYfNDQhK8b.jpg

copyright_author: dadazhang
copyright_author_href: https://dadazhangn.com
copyright_url: https://dadazhangn.com
copyright_info: 此文章版权归公共所有，欢迎转载。
---






# 项目：人工智能-表情识别 (DLBDSEAIS02)

该项目利用 TensorFlow 作为主要框架，专注于使用卷积神经网络 (CNN) 从面部表情进行情绪检测。 TensorFlow 的部署功能（包括 TensorFlow Serving 和 TensorFlow Lite）可确保无缝过渡到生产。该工作流程涉及预处理管道，将 mma-facial-express 数据集的图像转换为 48x48 尺寸。 CNN 架构包括卷积层、最大池化、ReLU 激活、密集层和具有 Softmax 激活的全连接层。模型测试涉及准确率、召回率和精确率等指标，以及用于多分类的 F1“分数”。部署是通过 FastAPI 和 Docker 完成的，支持本地或基于云的执行。

## 目的
为构建人工智能工具，利用图像中的面部表情来分析对广告的情绪反应。开发一个系统来对图像数据集中的情绪（例如快乐、愤怒、恐惧）进行分类。目标是通过面部表情识别至少三种情绪状态。

# 开发规划 - UML 模式
## 统一建模语言图
图（概念阶段修订）- 卷积神经网络的设计，用于分类情绪并使用 API 提供服务
![Design of Convolutional Neural Network to classify emotions and serve using API](https://s2.loli.net/2024/04/20/ciCzGqohpR1lEK9.jpg)

创建的 Keras 序列模型是一个卷积神经网络 (CNN)，专为图像分类而设计，更具体地说，是对面部中存在的 7 种不同情绪进行分类。

该架构从具有 32 个滤波器的卷积层 (Conv2D_1) 开始，然后是批量归一化。随后，Conv2D_2 采用 64 个滤波器，具有 ReLU 激活和“相同”填充，并伴有批量归一化。 MaxPooling2D_1减少空间维度，Dropout_1防止过拟合。该模型还包括额外的几对卷积层和批量归一化层（Conv2D_3、BatchNormalization_3、Conv2D_4、BatchNormalization_4），每个层后面都有 MaxPooling 和 Dropout。该模式继续为 Conv2D_5、BatchNormalization_5、Conv2D_6、BatchNormalization_6、MaxPooling2D_3 和 Dropout_3。 Flattening 层为 Dense 层准备数据，从 Dense_1（2048 个单元，ReLU 激活）和 BatchNormalization_7 开始，然后是 Dropout_4。最终的 Dense_2 层利用 softmax 激活函数生成 7 个类别的输出概率。

该架构结合了卷积层和全连接层，通过批量归一化和 dropout 进行增强，以增强泛化并防止过度拟合。

## 数据集概述
提供的图像数据集名为 MMAFEDB，分为三个主要文件夹：test、train 和 valid。每个文件夹都包含七种不同情绪的子文件夹：愤怒、厌恶、恐惧、快乐、中性、悲伤和惊讶。数据集统计显示，测试集包含 17,356 张图像，其中 13,767 张灰度图像，3,589 张 RGB 图像。训练集由 92,968 张图像组成，其中 64,259 张灰度图像，28,709 张 RGB 图像。验证集包含 17,356 张图像，其中 13,767 张灰度图像和 3,589 张 RGB 图像。数据集源自 Kaggle ( https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression )。各组情绪的分布各不相同，其中中性情绪是最普遍的情绪。还可以在 1_dataset_exploration.ipynb 笔记本中找到以下结果，其中收集了有关数据集的见解，并且还对低于特定大小阈值的低质量图像进行了简要清理。

训练文件夹：
- 中性：29,384 张图像
- 愤怒：6,566 张图片
- 恐惧：4,859 张图片
- 快乐：28,592 张图片
- 厌恶：3,231 张图片
- 悲伤：12,223 张图片
- 惊喜：8,113 张图片

测试文件夹：
- 中性：5,858 张图像
- 愤怒：1,041 张图片
- 恐惧：691 张图片
- 快乐：5,459 张图片
- 厌恶：655 张图片
- 悲伤：2,177 张图片
- 惊喜：1,475 张图片

验证文件夹:
- 中性：5,839 张图片
- 愤怒：1,017 张图片
- 恐惧：659 张图片
- 快乐：5,475 张图片
- 厌恶：656 张图片
- 悲伤：2,236 张图片
- 惊喜：1,474 张图片

此细分提供了 MMAFEDB 数据集中每个文件夹的不同情绪图像分布的详细视图。

# 模型开发
## 模型训练
在 2_model_training_iterative.ipynb 中，使用清理后的数据集（不平衡情绪）训练四个不同的 Keras Sequential 模型。这些模型分别命名为 model_1_concept、model_2_best、model_3_best_nodropout 和 model_4_best_nodropout_nobatchn，是根据卷积神经网络 (CNN) 的初始概念和文献建议通过实验和调整而开发的。

该代码首先导入必要的模块，并为训练集和测试集创建 ImageDataGenerators 和 flow_from_directory 函数。模型架构在 JSON 文件中指定，代码读取这些文件以迭代构建每个 Keras 模型。这样可以轻松迭代不同的模型，并手动调整优化器和颜色模式（RGB 或灰度）。每个模型的训练历史记录都会被保存，从而实现性能和学习的跟踪和可视化。此外，模型架构保存在“diagram”文件夹中，检查点文件保存在改进的epoch上，最终模型保存为h5文件。

保存的文件格式的示例包括 `modelname` _ `optimizer` _ `colormode` _ `batchsize` _ `balanced vs standard` _ `history, final or cpt` . `filename_ext`:
- 历史记录：model_1_concept_adam_grayscale_32_augment_history.json
- 最终模型：model_2_best_rmsprop_rgb_512_augment_final.h5
- 检查点：model_3_best_nodropout_rmsprop_rgb_512_augment_cpt.h5

出于实用性的考虑，仅将性能最佳的模型保存在专用的“models/best_model”文件夹中，包括架构（JSON 文件）和权重（h5 文件）。

## 模型指标和超参数
在 3_training_evaluation.ipynb 中，对在清理后的数据集上训练的情绪识别模型进行了深入评估。分析首先导入模块并定义一个函数来识别每个模型的训练历史记录中的最高指标，例如准确度、损失、val_accuracy 和 val_loss。选择具有最高 val_accuracy 的模型（标识为“model_2_best_sgd_rgb_128”）进行进一步审查。实现了排名函数，创建一个根据 val_accuracy 对模型进行排名的数组。打印了前 10 个最佳模型和 5 个最差模型，提供了模型相对性能的快速概览。

随后，函数使用各种优化器和批量大小绘制不同的模型，比较 RGB 和灰度颜色模式的结果。此探索旨在考虑数据集的主要灰度图像，辨别使用任一颜色模式的任何明显效果。

为了详细了解最佳模型的性能，使用模型训练历史记录创建数据帧，捕获跨时期的指标（准确性、损失、val_accuracy、val_loss）。直方图说明了性能指标的分布，强调了最高的累积箱。 val_accuracy 与 epoch 的热图提供了有关模型学习速度的见解，颜色越亮表示学习速度越快。

此外，利用数据帧来查找按模型分组的最大值和最小值，从而提供训练历史记录的详细快照。最大值按 val_accuracy 降序排列，显示在数据框中。在所有模型中，确定了最低和最高值，从而提供对整体性能范围的深入了解。此外，还计算并可视化平均值、中位数和箱线图，以提供度量分布的概述。

然后，分析将重点转移到性能最佳的模型上，探究其训练和验证的准确性、损失和混淆矩阵。混淆矩阵突出了对角线上的正确预测，并暴露了预测某些情绪的挑战。分类报告提供详细的指标，揭示每个情感标签的 f1 分数、精确度、召回率和支持度。

接下来是微观和宏观平均 ROC 曲线分析，说明真阳性率和假阳性率与相应 AUC 值之间的权衡。最后，使用 batch_size = 1 的测试集对最佳模型进行评估，从而深入了解训练和验证的准确性。这种多方面的评估提供了对经过训练的情绪识别模型的全面理解，揭示了它们的优势、挑战和整体性能特征。

![Model 1 Image](https://s2.loli.net/2024/04/20/5IefN4L6FDp9UHB.png)

迭代训练 4 个不同模型的结果在 /models 中:
- **model_1_concept** - 概念阶段最初描述的模型
- **model_2_best** - 在尝试不同架构时发现的模型
- **model_3_best_nodropout** - 丢弃 model_2 中 dropout 的结果
- **model_4_best_nodroupout_nobatchn** - 从 model_2 中丢弃 dropout 和批量归一化的结果

使用三个通道（RGB）进行图像输入，得到以下结果：
![RGB Model Training Results](https://s2.loli.net/2024/04/20/WsOkwRQz3YAPciU.jpg)

使用一个通道（灰度）进行图像输入，得到以下结果：
![GRAYSCALE Model Training Results](https://s2.loli.net/2024/04/20/mvp3s715AJubOBD.jpg)

可以看出，尽管大部分训练数据是灰度图像，但 RGB 训练结果稍好一些。该模型似乎从 RGB 数据中学习了仅通过一个通道无法获得的方面。

## 平衡数据集
我们尝试使用均衡数量的情感图像，以便每个类别都有相同的数量。从原始数据集中可以清楚地看出，某些情绪包含许多图像，而另一些情绪只占很小的比例。因此，我们探索了两种不同的平衡尝试，以了解增强的影响。首先，使用10.000 个图像的低阈值，以便图像数量较少的文件夹最终不会出现其原始图像的许多倍。最初，它被认为是更好的选择，因为太多的增强来达到其他类别的原始计数预计会产生不太好的结果。然而，平衡 2 数据集（所有图像类别都经过增强，每种情感达到30,000 张图像）表现更好。即使包含很少图像的情感，它仍然表现得更好。

可以在找到`4_balanced_emotions.ipynb` 和 `5_balanced_emotions_2.ipynb`. 首先，旧的增强被清理以返回到原始数据集。然后，对于未达到所需量的每种情绪，执行增强，然后文件夹达到所需量，例如 10,000 个图像。接下来，修剪超出所需数量的文件夹（首先删除尺寸最小的图像），以便在训练开始时所有文件夹都具有相同数量的图像。

这两幅图说明了增强是如何进行的。

**Balanced-1数据集 (每类 10.000 个图像)**
![Balanced-1 Dataset](https://s2.loli.net/2024/04/20/hC53aDe9TqJmozs.png)
**Balanced-1 数据集(每类 30.000 个图像) - - 未更改，因为全部满足 30.000 个要求。**
![Balanced-2 Dataset](https://smms.app/image/xXD8VHOMy2uw1rc)

模型经过训练，评估数据包含如下。


## 最佳的表现模型
根据最终的最大训练验证精度（val_accuracy）选择最佳模型。

**验证顶部和底部 5 个模型的准确性**
![best_worst_models_val_accuracy](https://s2.loli.net/2024/04/20/mTraQiGABK4klJp.png)

**用于测试和验证的最高准确度和最低损失的箱线图**
![boxenplot_highest_acc_lowest_loss](https://s2.loli.net/2024/04/20/eH472zFMPuURoaW.png)

**显示准确度的直方图分布，并突出显示最高条形（平均值）**
![histplot_val_accuracy_train_validate](https://s2.loli.net/2024/04/20/QoRkdDCEa3xJnmS.png)

- al_accuracy最高的模型文件为:  **../models/training/history/model_2_best_sgd_rgb_128_augment_history.json**
- 所有模型最大达到 val_accuracy 的平均值为: **0.5652485278745493**
- 最高的val_accuracy（model_2_best_sgd_rgb_128）是: **0.5928241014480591**

**不平衡数据集的训练历史**
![metrics model_2_best_sgd_rgb_128](https://s2.loli.net/2024/04/20/4Lx5BdVgzownPcW.png)
最佳模型的性能比平均 val_accuracy 训练验证好约 2.8%。

然而，如果我们在评估时使用批量大小 1，我们会发现最好的模型表现更好。
     评估模型的最终结果(batchsize = 1):
     训练准确度 = 83.28%
     验证准确度 = 63.52%

**Balanced-1 数据集的训练历史**
![metrics model_2_best_sgd_rgb_128 balanced](https://s2.loli.net/2024/04/20/biVfj6sD2kBOKuJ.png)

**Balanced-2 数据集的训练历史**
![metrics model_2_best_sgd_rgb_128 balanced 2](https://s2.loli.net/2024/04/20/4evM96YsPZlotBT.png)

## 分类图
- A -  最佳模型不平衡
- B - 最佳模型 Balanced-1 数据集
- C - 最佳模型 Balanced-2 数据集

### ROC曲线
![ROC Curve for the best model unbalanced & balanced emotions](https://s2.loli.net/2024/04/20/OBAdhZg3zPLKEHc.png)

### 混淆矩阵
![Confusion Matrixes for the best model unbalanced & balanced emotions](https://s2.loli.net/2024/04/20/rDj8kAc23nehPRS.png)

### 分类报告
![Classification Report for the best model unbalanced & balanced emotions](https://s2.loli.net/2024/04/20/hJ5VzDqln7wCNc1.png)

# Docker API 前端
### 参阅 docker-api中的更多信息
重新训练的模型使用 Balanced-2 数据集的情感图像进行预测。
- 用户从他们的设备中选择一个文件
- 用户点击预测
- 显示预测情绪和类别概率。
![docker-api front-end](https://s2.loli.net/2024/04/20/cGKxsQ8ieOv3NUD.png)

# 开始
## 使用 Jupyter Notebook 的依赖项 - 注意，如果不存在 CUDA GPU，它将默认使用 CPU（速度慢）
创建一个新的虚拟 python 环境。

`python3 -m venv venv`

激活环境

`source venv/bin/activate`

安装依赖项

`pip3 install -r requirements.txt`

## 使用预构建 docker 镜像的 API 使用
### 拉取最新的构建镜像
`docker pull dadazhang55/emotion_tensorflow:latest`
### 运行容器
`docker run --name emotion_prediction_container -p 8000:8000 emotion_prediction_fastapi:latest`
### 在本地主机上打开 API
`http://127.0.0.1:8000`

# Reflection
最初，使用 CUDA 驱动程序很麻烦，而且要反复试验才能让我的 NVIDIA GPU 正常工作，这是我面临的最大挑战。在使用卷积神经网络开发情感分析的过程中，我了解了模型架构（卷积层/最大池/dropout/批量归一化）、优化器和损失函数。

此外，如何使用不同的指标来衡量模型性能，以捕获训练历史并评估所使用的所有不同参数的影响。我的目标是创建一个以清晰且针对特定目的的方式组织的项目结构，并将权重和捕获的模型指标保存在指定的文件夹和文件中。通过这个过程，我更好地了解了如何以迭代方式保存和训练模型，以及如何在后期训练中恢复模型并与架构或仅训练的权重一起保存，以及如何使用这些文件。

我们发现，以 RGB 颜色训练时，包含大部分灰度图像和次要部分 RGB 图像的图像数据集似乎仍然表现更好。通过阅读不同的文献和在线材料，我了解了 dropout 和批量归一化如何显着提高泛化能力和性能。

使用不同的可视化对我训练的模型进行评估；例如，混淆矩阵、分类报告（f1 分数、召回率、精确度、准确度）让我很好地了解了模型的表现、意味着什么并确定了潜在的改进领域。通过图像增强平衡数据集让我学会了如何平衡图像类，但是它并没有显着改善我的不平衡数据集，这肯定是我在未来项目中更多探索和优化的领域。

最后，使用 FastAPI 和 Docker 构建一个简单的 api， 让我从用户的角度思考，调用 API 或使用前端网站与我训练的模型进行交互。


# 结论
该项目基于评估整个测试集，使用不平衡模型对情感图像进行了平均 63.52% 的分类。(https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression/code). 

