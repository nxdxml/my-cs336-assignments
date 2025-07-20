import os
from tests.common import FIXTURES_PATH
from typing import BinaryIO, Iterable, Iterator
# 多线程相关
from multiprocessing import Process, Queue
import regex as re
# 导入自动初始化的dict
from collections import defaultdict

# from cs336_basics.pretokenization_example import find_chunk_boundaries

# test
from tests.common import gpt2_bytes_to_unicode
# from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
import heapq

VOCAB_PATH = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\tests\\fixtures\\gpt2_vocab.json"
MERGES_PATH = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\tests\\fixtures\\gpt2_merges.txt"


import time

import json
def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding='utf-8') as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path, encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return BEPTokenizer(vocab, merges, special_tokens)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    如果总文件大小太小固定只有一个分块
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenizer(
        chunk : str,
        special_tokens: list[str],
        ) -> list[bytes]:
    """_summary_

    Args:
        chunk (str): 输入字符
        special_tokens (list[str]): 特殊字符

    Returns:
        list[bytes]: 预处理后的字符串
    """    """
    pretokenizer
    """
    # 1. "xxx|special_tokens[i]|yyy" ->  ["xxx","special_tokens[i]","yyy"]
    if not special_tokens : 
        pretoken_split_special_list = [chunk]
    else :
        sorted_special_token = sorted(special_tokens, key=len, reverse=True)
        # 输入["<|endoftext|>", "<|pad|>"]
        # 输出['<\\|endoftext\\|>', '<\\|pad\\|>']
        # 转义避免识别到一些特殊字符
        pattern = "|".join(map(re.escape, sorted_special_token))
        # 正则小知识，不用括号抱住会把pattern丢弃
        # re.split("abc", "123abc456")
        # ['123', '456']
        pretoken_split_special_list = re.split(f"({pattern})", chunk)
    
    # 2 pretokenization
    GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []

    for part in pretoken_split_special_list:
        if part in special_tokens:
            # 这里训练的时候不能加进去
            # continue
            # 很坑这里要弄成utf-8
            tokens.append(part.encode('utf-8'))
        else :
            # 找到所有匹配串
            str_token = re.findall(GPT2_TOKENIZER_REGEX, part)
            # 全部加入tokens中
            tokens.extend(c.encode('utf-8') for c in str_token)
    
    # 直接返回
    return tokens

def worker(
        chunk : str,
        special_tokens: list[str],
        q : Queue    
        ) -> None:
    """
    子进程执行的函数，处理任务并将结果放入队列。
    得到多个pre_tokenization的序列
    需要把speccial_token 单独处理
    """
    # 1. "xxx|special_tokens[i]|yyy" ->  ["xxx","special_tokens[i]","yyy"]
    if not special_tokens : 
        pretoken_split_special_list = [chunk]
    else :
        sorted_special_token = sorted(special_tokens, key=len, reverse=True)
        # 输入["<|endoftext|>", "<|pad|>"]
        # 输出['<\\|endoftext\\|>', '<\\|pad\\|>']
        # 转义避免识别到一些特殊字符
        pattern = "|".join(map(re.escape, sorted_special_token))
        # 正则小知识，不用括号抱住会把pattern丢弃
        # re.split("abc", "123abc456")
        # ['123', '456']
        pretoken_split_special_list = re.split(f"({pattern})", chunk)
    
    # 2 pretokenization
    GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []

    for part in pretoken_split_special_list:
        if part in special_tokens:
            # 这里训练的时候不能加进去
            continue
            tokens.append(part)
        else :
            # 找到所有匹配串
            str_token = re.findall(GPT2_TOKENIZER_REGEX, part)
            # 全部加入tokens中
            tokens.extend(c.encode('utf-8') for c in str_token)
    
    # 3 加入队列
    q.put(tokens)

"""
这里的bytes不是指返回长度是一个字节而是表示是2进制

"""
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1初始化词表 单字节+特殊字符

    start_time=time.time()

    vocab = {}
    vocab = {x:bytes([x]) for x in range(0,256)} 
    # .encoder()确保是bytes
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode()
    # print(f'vocav = {vocab}')
    merges = []

    # text = "Hello, 世界!"
    # print(text)            # 字符串
    # print(text.encode())   # 字节串 b'Hello, \xe4\xb8\x96\xe7\x95\x8c!'

    # 2 分块处理
    chunked_file_list = []
    num_processes = 8
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

        # print(f'len(boundaries) = {len(boundaries)}')
        # print(boundaries)

        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            # windows 换行符错误
            chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')
            # 加入分块
            chunked_file_list.append(chunk)
    
    # print(f'len(chunked_file_list) = {len(chunked_file_list)}')

    # 3 分块后每块多线程pre_tokenization
    # 3.1 创建进程间通信队列
    q = Queue()
    # 3.2 创建并且启动多进程
    processes = []
    for chunk in chunked_file_list:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    # 3.3 队列收集所有子进程结果
    results = [q.get() for _ in processes]

    # 3.4 等待子进程结束
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"所有子进程完成预分词, 已用时间: {elapsed:.2f}s")
    
    # print(len(results))
    # print(results[0][0:100])

    # 3.5 聚合结果
    tokens = []
    for part in results:
        for token in part:
            tokens.append(token)

    # print(tokens)
    # print(type(tokens), type(tokens[0]), tokens[0])
    
    # print(len(tokens))
    # print(tokens[0:500])
    # 小规模测试
    # tokens = tokens[0:500]
    # 4 merge
    counts = defaultdict(int) # 每个pair出现次数
    index_map = defaultdict(set)  # pretoken 位置映射
    
    # 4.1 初始化
    # for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
    #     pair = (a, b)
    #     counts[pair] += 1
    #     index_map[pair].add(i)
    
    for i, token in enumerate(tokens):
        for index1, index2 in zip(token, token[1:]):
            # print(f'{index1},{index2}')
            counts[index1, index2] += 1
            index_map[index1, index2].add(i)


    # print(index_map)
    # return vocab, merges
    # 4.2 merge


    
    base_idx = 256 + len(special_tokens)
    num_merge = vocab_size - base_idx
    for i in range(num_merge):
        # 关键时间消耗
        if i % 1 == 0 or i == num_merge - 1:
            elapsed = time.time() - start_time
            print(f"Merge步骤 {i+1}/{num_merge}, 已用时间: {elapsed:.2f}s")

        # len(token) 长度固定
        # 
        # 找出出现最多次的idx对
        # print(type(tokens), type(tokens[0]), tokens[0], len(tokens))
        # <class 'list'> <class 'bytes'> b'iron' 27758
        # ...
        # <class 'list'> <class 'bytes'> b'iron' 27758
        # <class 'list'> <class 'list'> [105, 114, 274] 27758
        # <class 'list'> <class 'list'> [105, 114, 274] 27758
        # <class 'list'> <class 'list'> [105, 114, 274] 27758
        most_common = max(
            counts.items(),
            key=lambda x:(
                x[1], # 次数优先
                vocab[x[0][0]].decode('utf-8', errors="ignore"), # token1字典序
                vocab[x[0][1]].decode('utf-8', errors="ignore"), # token2字典序
            )
        )[0]
        idx_a , idx_b = most_common
        new_token_id = base_idx + i
        new_token_bytes = vocab[idx_a] + vocab[idx_b]
        # 更新词表merge
        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[idx_a], vocab[idx_b]))

        # merge() 这部分研究一下数据结构
        # 包含最大的token id的set
        index_set = index_map[most_common]

        # print(type(tokens))

        for x in index_set:
            # 这个idx对在token中出现的位置
            pretoken = tokens[x]
            # print(tokens[x], len(tokens[x]), most_common)
            # b' Key' 4
            # [32, 75, 494, 111] 4
            # [289, 393, 114, 650, 472] 5
            # [99, 274, 319, 114, 650, 296] 6
            new_token = []


            # Replace max_pair with new_index in each pretoken

            # print(type(tokens[0]))
            # print(len(pretoken))
            
            # 当前list如果发现目标对则替换，否则用原来的
            # 在新的list中记录哪些是新merge的

            pos_list = [] # 被替换的pair在new_token中的位置
            pos = 0
            j = 0
            while j < len(pretoken):
                if (j < len(pretoken)-1) and ((pretoken[j], pretoken[j+1]) == most_common):
                    # print(pretoken, pretoken[j], pretoken[j+1], ((pretoken[j], pretoken[j+1]) == most_common))
                    new_token.append(new_token_id)
                    pos_list.append(pos)
                    j += 2
                else:
                    new_token.append(pretoken[j])
                    j += 1
                pos += 1

            # 更新数据结构
            # [10, 11, 12, 13]
            # [10, 300, 13]
            # 减掉旧 pair (11, 12)
            # 减掉旧 pair (10, 11) 和 (12, 13)
            # 增加新 pair (10, 300) 和 (300, 13)
            for pos in pos_list:
                counts[most_common] -= 1  # 减少当前被合并的pair计数

                # 处理左侧邻居
                if pos > 0:
                    left = new_token[pos - 1]
                    if left == new_token_id:
                        counts[(most_common[1], most_common[0])] -= 1
                    else:
                        counts[(left, most_common[0])] -= 1
                    
                    counts[(left, new_token[pos])] += 1
                    index_map[(left, new_token[pos])].add(x)

                # 处理右侧邻居
                if pos < len(new_token) - 1:
                    right = new_token[pos + 1]
                    if right == new_token_id:
                        counts[(most_common[1], most_common[0])] -= 1
                    else:
                        counts[(most_common[1], right)] -= 1
                    
                    counts[(new_token[pos], right)] += 1
                    index_map[(new_token[pos], right)].add(x)

            tokens[x] = new_token


    return vocab, merges


def train_bpe_try(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1初始化词表 单字节+特殊字符

    start_time=time.time()

    vocab = {}
    vocab = {x:bytes([x]) for x in range(0,256)} 
    # .encoder()确保是bytes
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode()
    # print(f'vocav = {vocab}')
    merges = []

    # text = "Hello, 世界!"
    # print(text)            # 字符串
    # print(text.encode())   # 字节串 b'Hello, \xe4\xb8\x96\xe7\x95\x8c!'

    # 2 分块处理
    chunked_file_list = []
    num_processes = 8
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

        # print(f'len(boundaries) = {len(boundaries)}')
        # print(boundaries)

        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            # windows 换行符错误
            chunk = chunk.replace('\r\n', '\n').replace('\r', '\n')
            # 加入分块
            chunked_file_list.append(chunk)
    
    # print(f'len(chunked_file_list) = {len(chunked_file_list)}')

    # 3 分块后每块多线程pre_tokenization
    # 3.1 创建进程间通信队列
    q = Queue()
    # 3.2 创建并且启动多进程
    processes = []
    for chunk in chunked_file_list:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    # 3.3 队列收集所有子进程结果
    results = [q.get() for _ in processes]

    # 3.4 等待子进程结束
    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"所有子进程完成预分词, 已用时间: {elapsed:.2f}s")
    
    # print(len(results))
    # print(results[0][0:100])

    # 3.5 聚合结果
    tokens = []
    for part in results:
        for token in part:
            tokens.append(token)

    # print(tokens)
    # print(type(tokens), type(tokens[0]), tokens[0])
    
    # print(len(tokens))
    # print(tokens[0:500])
    # 小规模测试
    # tokens = tokens[0:500]
    # 4 merge
    counts = defaultdict(int) # 每个pair出现次数
    index_map = defaultdict(set)  # pretoken 位置映射
    
    # 4.1 初始化
    # for i, (a, b) in enumerate(zip(tokens, tokens[1:])):
    #     pair = (a, b)
    #     counts[pair] += 1
    #     index_map[pair].add(i)
    
    for i, token in enumerate(tokens):
        for index1, index2 in zip(token, token[1:]):
            # print(f'{index1},{index2}')
            counts[index1, index2] += 1
            index_map[index1, index2].add(i)


    # print(index_map)
    # return vocab, merges
    # 4.2 merge

    most_common = None
    most_common_freq = -1
    
    base_idx = 256 + len(special_tokens)
    num_merge = vocab_size - base_idx
    for i in range(num_merge):
        # 关键时间消耗
        if i % 1 == 0 or i == num_merge - 1:
            elapsed = time.time() - start_time
            print(f"Merge步骤 {i+1}/{num_merge}, 已用时间: {elapsed:.2f}s")

        # len(token) 长度固定
        # 
        # 找出出现最多次的idx对
        # print(type(tokens), type(tokens[0]), tokens[0], len(tokens))
        # <class 'list'> <class 'bytes'> b'iron' 27758
        # ...
        # <class 'list'> <class 'bytes'> b'iron' 27758
        # <class 'list'> <class 'list'> [105, 114, 274] 27758
        # <class 'list'> <class 'list'> [105, 114, 274] 27758
        # <class 'list'> <class 'list'> [105, 114, 274] 27758
        if most_common is None or counts[most_common] <= 0:
            most_common, most_common_freq = max(
                counts.items(),
                key=lambda x:(
                    x[1], # 次数优先
                    vocab[x[0][0]].decode('utf-8', errors="ignore"), # token1字典序
                    vocab[x[0][1]].decode('utf-8', errors="ignore"), # token2字典序
                )
            )
        idx_a , idx_b = most_common
        new_token_id = base_idx + i
        new_token_bytes = vocab[idx_a] + vocab[idx_b]
        # 更新词表merge
        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[idx_a], vocab[idx_b]))

        # merge() 这部分研究一下数据结构
        # 包含最大的token id的set
        index_set = index_map[most_common]

        # print(type(tokens))

        for x in index_set:
            # 这个idx对在token中出现的位置
            pretoken = tokens[x]
            # print(tokens[x], len(tokens[x]), most_common)
            # b' Key' 4
            # [32, 75, 494, 111] 4
            # [289, 393, 114, 650, 472] 5
            # [99, 274, 319, 114, 650, 296] 6
            new_token = []


            # Replace max_pair with new_index in each pretoken

            # print(type(tokens[0]))
            # print(len(pretoken))
            
            # 当前list如果发现目标对则替换，否则用原来的
            # 在新的list中记录哪些是新merge的

            pos_list = [] # 被替换的pair在new_token中的位置
            pos = 0
            j = 0
            while j < len(pretoken):
                if (j < len(pretoken)-1) and ((pretoken[j], pretoken[j+1]) == most_common):
                    # print(pretoken, pretoken[j], pretoken[j+1], ((pretoken[j], pretoken[j+1]) == most_common))
                    new_token.append(new_token_id)
                    pos_list.append(pos)
                    j += 2
                else:
                    new_token.append(pretoken[j])
                    j += 1
                pos += 1

            # 更新数据结构
            # [10, 11, 12, 13]
            # [10, 300, 13]
            # 减掉旧 pair (11, 12)
            # 减掉旧 pair (10, 11) 和 (12, 13)
            # 增加新 pair (10, 300) 和 (300, 13)
            for pos in pos_list:
                counts[most_common] -= 1  # 减少当前被合并的pair计数

                # 处理左侧邻居
                if pos > 0:
                    left = new_token[pos - 1]
                    if left == new_token_id:
                        counts[(most_common[1], most_common[0])] -= 1
                    else:
                        counts[(left, most_common[0])] -= 1
                    
                    counts[(left, new_token[pos])] += 1
                    index_map[(left, new_token[pos])].add(x)

                # 处理右侧邻居
                if pos < len(new_token) - 1:
                    right = new_token[pos + 1]
                    if right == new_token_id:
                        counts[(most_common[1], most_common[0])] -= 1
                    else:
                        counts[(most_common[1], right)] -= 1
                    
                    counts[(new_token[pos], right)] += 1
                    index_map[(new_token[pos], right)].add(x)

            tokens[x] = new_token


    return vocab, merges






class BEPTokenizer:
    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None :
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.r_vocab =  {v:k for k,v in self.vocab.items()}
        self.b_special_tokens = [x.encode('utf-8') for x in self.special_tokens]
        # print(self.b_special_tokens)
    # 过于朴素非常慢
    # def encode(self, text : str) -> list[int]:

    #     pre_tokens = pretokenizer(text, self.special_tokens) # list[bytes]
    #     tokens = [] # list[list[int]]
    #     for i, pre_token in enumerate(pre_tokens):
    #         new_token = []
    #         # print(i, pre_token)
    #         if pre_token in self.b_special_tokens:
    #             new_token.append(self.r_vocab[pre_token])
    #             # print(i, pre_token)
    #         else :
    #             for x in pre_token:
    #                 # 这里之前写错了直接写了字符的ASCII码进去。。
    #                 new_token.append(self.r_vocab[bytes([x])])
    #         tokens.append(new_token)

    #     # print(tokens)
    #     # self.merge_ranks = { (a + b): i for i, (a, b) in enumerate(self.merges) }


    #     # 不知为啥这里正向应用一遍merge就行了，很奇怪，可能和训练严格按照顺序有关？
    #     for i, token in enumerate(tokens):
    #         for merge in self.merges:
    #             new_token = []
    #             new_id = self.r_vocab[merge[0] + merge[1]]
    #             j = 0
    #             while j < len(token):
    #                 # 匹配
    #                 if j < len(token) - 1 and (self.vocab[token[j]],self.vocab[token[j + 1]]) == merge:
    #                     new_token.append(new_id)
    #                     j += 2
    #                 else :
    #                     new_token.append(token[j])
    #                     j += 1
    #             token = new_token
    #         tokens[i] = token
    #     # 展开
    #     flat_tokens = []
    #     for token in tokens:
    #         for x in token:
    #             flat_tokens.append(x)
    #     return flat_tokens
 
    def encode(self, text: str) -> list[int]:
        pre_tokens = pretokenizer(text, self.special_tokens)  # list[bytes]
        tokens = []
        for pre_token in pre_tokens:
            if pre_token in self.b_special_tokens:
                tokens.append([self.r_vocab[pre_token]])
            else:
                tokens.append([self.r_vocab[bytes([b])] for b in pre_token])

        self.merge_ranks = {(a + b): i for i, (a, b) in enumerate(self.merges)}

        for i, token in enumerate(tokens):
            while True:
                pairs = [(token[j], token[j + 1]) for j in range(len(token) - 1)]
                # 找所有可合并的pair，pair是字节串，token是id，需要转成bytes再查
                merge_candidates = []
                for j, (id1, id2) in enumerate(pairs):
                    bytes_pair = self.vocab[id1] + self.vocab[id2]
                    if bytes_pair in self.merge_ranks:
                        rank = self.merge_ranks[bytes_pair]
                        merge_candidates.append((j, rank, bytes_pair))
                if not merge_candidates:
                    break
                # 找rank最小的pair
                j, _, bytes_pair = min(merge_candidates, key=lambda x: x[1])
                new_token_id = self.r_vocab[bytes_pair]
                # 合并
                token = token[:j] + [new_token_id] + token[j+2:]
            tokens[i] = token

        # 展开
        flat_tokens = [tid for token in tokens for tid in token]
        return flat_tokens
 
    # TODO 循环调用encode，学一下
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-eﬀicient tokenization of large files 
        that we cannot directly load into memory.
        """
        for line in iterable:
            for idx in self.encode(line):
                yield idx

    
    def decode(self, ids : list[int]) -> str :
        tokens = bytes()
        for id in ids:
            tokens += self.vocab[id]
        # 这里不加error有问题
        return tokens.decode(encoding='utf-8', errors='replace')
        



def main():
    
    input_path = FIXTURES_PATH / "corpus.en"
    # input_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\data\\TinyStoriesV2-GPT4-valid.txt"

    # print(input_path)

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    bpe_tokenizer = BEPTokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    test_str = "u don't have to be scared of the loud dog <|endoftext|>"
    test_str = "s"
    output = bpe_tokenizer.encode(test_str)
    print(f'编码输出{output}')
    d_str = bpe_tokenizer.decode(output)
    print(f'解码后字符串{d_str}')
    assert test_str == d_str


    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    output = tokenizer.encode(test_str)
    print(f'编码输出{output}')
    d_str = tokenizer.decode(output)
    print(f'解码后字符串{d_str}')

    # reference_vocab_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\tests\\fixtures\\train-bpe-reference-vocab.json"
    # reference_merges_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\tests\\fixtures\\train-bpe-reference-merges.txt"

    # # Compare the learned merges to the expected output merges
    # gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    # print(gpt2_byte_decoder)
    # with open(reference_merges_path, encoding='utf-8') as f:
    #     gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
    #     reference_merges = [
    #         (
    #             bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
    #             bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
    #         )
    #         for merge_token_1, merge_token_2 in gpt2_reference_merges
    #     ]
    # print(reference_merges)
    # print(merges)
    # print(vocab)

if __name__ == "__main__":
    main()