## 思路记录

因为在自动驾驶小车运行时，各种传感器数据会一起发送，为了检测某一类数据（比如GNSS）是不是收到spoofing攻击，所以考虑了使用attention的思想。

但这里的attention有区别于transformer中的attention -- 计算不同的token之间的correlation，而是为了看输入的数据（因为是多维的）哪些特征以及哪些内联特征是需要重点关注的。通过sigmoid函数可以使得权重属于0-1的范围，从而把注意力放在比较重要的特征上。

对于这个输入数据，因为各个数据特征之间是有一定关联程度（或多或少，比如GNSS变化的同时一般会存在imu的变化。。。）。但是其实，如果spoofing攻击的是非常独立的数据特征，比如时间之类的，可能会被漏检。

整体思路：
编码器把输入数据做了一次“抽象总结”，得到z。
FeatureAttention：类似于一个“智能打分器”，它先看看每个“压缩后特征”在当前样本里是什么状态，就决定每个特征应有多少权重。它一开始是盲目的（随机打分），但随着训练，慢慢学会 e.g. 第 1 维特征很有用，就给它打高分（接近 1）；第 3 维特征没什么用，就给它打低分（接近 0）。
解码器拿着已经打分过的隐藏向量去“复原”成原始输入。打分高的维度会被放大、保留，打分低的就被抑制，解码器就重点“关注”保留下来的那些重要成分去重建——这样它能更快学会怎么把正常数据还原出来。

异常分数就是对每个样本的原始输入向量x和网络重建输出 $\hat{x}$ 之间的均方误差进行计算