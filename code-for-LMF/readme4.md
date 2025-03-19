# 方案1
1)The sentence *sent* means [mask1]
2)The sentence *sent* does not mean [mask2]
3)it can be summarized_as as [mask2] from [mask1] and [mask2]
多阶段推理, 分别从1和2中得到句子表示mask1和mask2， 然后把mask1和mask2输入到3阶段。
代码该如何实现

https://yuanbao.tencent.com/bot/app/share/chat/x2rqW3ifIhHg