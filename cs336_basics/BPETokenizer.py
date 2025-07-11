import os
from tests.common import FIXTURES_PATH
from typing import BinaryIO
# 多线程相关
from multiprocessing import Process, Queue
import regex as re
# 导入自动初始化的dict
from collections import defaultdict

# from cs336_basics.pretokenization_example import find_chunk_boundaries

# test
from tests.common import gpt2_bytes_to_unicode

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

    # print(len(results))
    # print(results[0][0:100])

    # 3.5 聚合结果
    tokens = []
    for part in results:
        for token in part:
            tokens.append(token)
        
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
            counts[index1, index2] += 1
            index_map[index1, index2].add(i)


    # print(index_map)
    # return vocab, merges
    # 4.2 merge
    base_idx = 256 + len(special_tokens)
    num_merge = vocab_size - base_idx
    for i in range(num_merge):
        most_common = max(
            counts.items(),
            key=lambda x:(
                x[1],
                vocab[x[0][0]].decode('utf-8', errors="ignore"),
                vocab[x[0][1]].decode('utf-8', errors="ignore"),
            )
        )[0]
        idx_a , idx_b = most_common
        new_token_id = base_idx + i
        new_token_bytes = vocab[idx_a] + vocab[idx_b]
        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[idx_a], vocab[idx_b]))

        # merge() TODO 这部分研究一下数据结构
        index_set = index_map[most_common]

        # print(type(tokens))

        for x in index_set:
            pretoken = tokens[x]
            new_pretoken = []

            pos_list = []   # Store positions of max_pair for each new pretoken after merge
            pos = 0
            j = 0

            # Replace max_pair with new_index in each pretoken
            while j < len(pretoken):
                if (j < len(pretoken)-1) and ((pretoken[j], pretoken[j+1]) == most_common):
                    new_pretoken.append(new_token_id)
                    pos_list.append(pos)
                    j += 2
                else:
                    new_pretoken.append(pretoken[j])
                    j += 1
                pos += 1

            # Update counts and index_dict
            for pos in pos_list:
                counts[most_common] -= 1

                if pos > 0:
                    if new_pretoken[pos-1] == new_token_id:
                        counts[(most_common[1], most_common[0])] -= 1    
                    else:
                        counts[(new_pretoken[pos-1], most_common[0])] -= 1

                    counts[(new_pretoken[pos-1], new_pretoken[pos])] += 1
                    index_map[(new_pretoken[pos-1], new_pretoken[pos])].add(x)

                if pos < len(new_pretoken)-1:
                    if new_pretoken[pos+1] == new_token_id:
                        counts[(most_common[1], most_common[0])] -= 1     
                    else:
                        counts[(most_common[1], new_pretoken[pos+1])] -= 1

                    counts[(new_pretoken[pos], new_pretoken[pos+1])] += 1
                    index_map[(new_pretoken[pos], new_pretoken[pos+1])].add(x)

            tokens[x] = new_pretoken


    return vocab, merges




def main():
    
    input_path = FIXTURES_PATH / "corpus.en"
    # input_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\data\\TinyStoriesV2-GPT4-valid.txt"

    # print(input_path)

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
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
    print(vocab)

if __name__ == "__main__":
    main()