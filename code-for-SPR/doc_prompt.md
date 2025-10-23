## prompt1
为了发表sci论文获取实验数据，我需要设计一个实验，创建一个新的python工程目录，参考code-bert-LMF项目代码，训练train具体步骤：
1.  设计prompt ”The sentence of [X] means [mask], and also means [mask]",其中X为实际句子占位符，将prompt 输入到模型pretrain_models/bert-base-uncased进行编码，得到句子X正编码表示h1*和编码表示h2*， h1*,h2*可理解为句子X语义的不同方面, 可解决为一束白光经过三菱镜后分解成2束不同颜色的光类似. 然后h1*，h2*去掉prompt模版本身的噪声，得到h1和h2.

2. 然后采用自正则化（Self-Projection Regularization, SPR）的投影-预测框架（Projection-Prediction Consistency），分别设计h1 和h2对应的投影网络project1-network，project2-network，得到h1-project, h2-project，类似于h1、h2两束光分别经过另外两个三菱镜即这里的project1-network，project2-network。

3. 分别设计(h1, h2)和（h1-project, h2-project）两对Prediction 网络框架，分别为Prediction-network， Prediction-network-project，将(h1, h2)输入到Prediction-network预测网络，(h1-project, h2-project）输入到Prediction-network-project网络，得到投影前后两个预测值：h_prediction, h_predication_project

4. 训练目标loss=||norm(h_prediction) - norm(h_predication_project)||

验证eval步骤：
采用上面训练步骤3中的h_prediction为预测值

## prompt2
采用下面自监督训练方法优化句子语义，性能会怎么样，创新性怎么样，适合发表sci 哪个级别期刊。 训练train具体步骤：

设计prompt ”The sentence of [X] means [mask], and also means [mask]",其中X为实际句子占位符，将prompt 输入到模型pretrain_models/bert-base-uncased进行编码，得到句子X正编码表示h1和编码表示h2， h1*,h2可理解为句子X语义的不同方面, 可解决为一束白光经过三菱镜后分解成2束不同颜色的光类似. 然后h1，h2*去掉prompt模版本身的噪声，得到h1和h2.

然后采用自正则化（Self-Projection Regularization, SPR）的投影-预测框架（Projection-Prediction Consistency），分别设计h1 和h2对应的投影网络project1-network，project2-network，得到h1-project, h2-project，类似于h1、h2两束光分别经过另外两个三菱镜即这里的project1-network，project2-network。

分别设计(h1, h2)和（h1-project, h2-project）两对Prediction 网络框架，分别为Prediction-network， Prediction-network-project，将(h1, h2)输入到Prediction-network预测网络，(h1-project, h2-project）输入到Prediction-network-project网络，得到投影前后两个预测值：h_prediction, h_predication_project

训练目标loss=||norm(h_prediction) - norm(h_predication_project)||

## prompt3
现在有什么研究可以从一个句子中提取不同的语义细节，比如互补语义（如主题 vs 细节、主观 vs 客观），并提供原理说明
例子：prompt 两 mask 原理化（为什么两 mask 捕获互补视图）并提供稳定的、可复现的增益