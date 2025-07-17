

## bpe分词器，把字符串转化为整数token序列(str -> list[int])

Unicode编码:把一个字符转化为数字，直接用这个将字符串转化为token为啥不行？
因为词表太大，部分数据很罕见稀疏。

因此把数据转化为字节序列，这样用固定的词表大小(256)就能表示所有序列了。

BPE分词器的训练和使用方法：
1. 预分词:理论上可以直接开始相邻pair的和并，但这样开销会非常大，每次合并需要遍历整个数据集，还会导致相近的词比如(people.)和(people!)被弄成不同的token，实际上识别好people和.分别作为token比较好。这里注意特殊字符不需要进行合并，需要把它记录到词表中，并且不用于接下来的训练。于是用一个正则表达式对整个输入字符串做预分词，效果如下：

``` py
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
print(re.findall(PAT, "some text that i'll pre-tokenize"))
# 输出: ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

2. 训练分词:得到上述很多个预分词的序列，仅仅在每个序列内进行合并，每次合并出现频率最高的pair,词表大小+1，直到词表vocab大小达到要求的大小。并且记录需要合并的(token1, token2)序列为merges表。

3. 使用训练得到的tokenizer进行编码和解码，依旧是进行预分词，然后严格按照训练顺序得到的merges表进行合并，细节比较多大概过程如下
``` py
# 1准备
pretokens : list[bytes] # 预分词得到多个字节序列，这里注意不要抛弃特殊字符
vocab : dict[int : bytes] # token_id -> token字节序列
r_vocab : dict[bytes : int] # token字节序列 -> token_id
merges : list[(bytes, bytes)] # 需要合并的两个字节序列
# 2用词表初始化pretokens,先单字节访问转化成初始的序列，这个过程反向查词表
pretokens : list[list[int]]
# 3根据merges和词表可以得到每轮哪两个token_id需要合并为新的token_id是啥，假设每次merges里取出的merge0, merge1
merge0 -> id0 # btyes -> int 根据词表r_vocab
merge1 -> id1 
r_vocab[merge0+merge1] -> new_id # 这一步很巧妙！
# 4在遍历了一个pretoken[i] 后即可得到一个新的list[int]序列，直接替换原序列
# 5最后展开为list[int]即可

```

profile:主要是一个运行时间和压缩比的问题，还有些细节
1. 原来多少字节 -> 我能压缩到多少token
2. 建议用uint16 保存结果，最后结果的每个数都不可能超过词表大小，这个大小足以保存
3. 时间消耗我的理解是可以用一些数据结构来加速，这个最后得到的list[int]可以保存为npy文件反复使用，vocab和merges也是。
