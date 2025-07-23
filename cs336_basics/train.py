"""
TODO
加入模型从断掉的step开始训练
动态调整学习率
梯度截断

"""

from tqdm import tqdm
import time
from tests.adapters import *
import os
import numpy as np


from cs336_basics.inference import generate_text
from cs336_basics.utils import save_bpe, load_bpe, detailed_parameter_stats

import logging
# 导入输出到一个log中
def setup_logger(log_file_path: str):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 全局最低日志等级

    # 清理已有处理器，防止重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(message)s')  # 简洁格式
    console_handler.setFormatter(console_formatter)

    # 文件处理器
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def main():
    # 1训练配置参数
    train_epochs = 1000 # 训练步数
    input_path = "/home/dl/projects/my-cs336-assignments/data/TinyStoriesV2-GPT4-valid-mini.txt"
    # input_path = "/home/dl/projects/my-cs336-assignments/data/TinyStoriesV2-GPT4-train.txt"
    base_name = os.path.basename(input_path)  # "TinyStoriesV2-GPT4-valid-mini.txt"
    name_without_ext = os.path.splitext(base_name)[0]  # "TinyStoriesV2-GPT4-valid-mini"
    
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    batch_size = 4
    context_length = 64 # 序列长度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 768 # 嵌入维度
    num_heads = 12
    num_layers = 12 # transformer block 个数
    d_ff = 4 * d_model 
    rope_theta = 10000
    checkpoint_path = "/home/dl/projects/my-cs336-assignments/cs336_basics/checkpoint"
    use_bpe = False # 是否需要使用bpe_train
    use_tokenizer = False # 是否已经训练好了词表和merges

    dataset_path = f"/home/dl/projects/my-cs336-assignments/data/dataset_{name_without_ext}_{vocab_size}.npy"

    # print(f"device = {device}")
    logger = setup_logger(f"{checkpoint_path}/training.log") # 记录训练日志

    # 2训练BPE分词器
    if use_bpe or use_tokenizer:
        if use_bpe:
            logger.info("train_bpe开始")
            start = time.time()
            vocab, merges = run_train_bpe(
                input_path=input_path,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
            save_bpe(vocab=vocab, merges=merges, save_dir=checkpoint_path, data_name=name_without_ext)
            logger.info("train_bpe结束")
            tokenizer = get_tokenizer(vocab=vocab,
                                    merges=merges,
                                    special_tokens=special_tokens)
            end = time.time()
            logger.info(f"耗时：{end - start:.4f} 秒")
        else :
            vocab, merges = load_bpe(load_dir=checkpoint_path, data_name=name_without_ext)
            tokenizer = get_tokenizer(vocab=vocab,
                        merges=merges,
                        special_tokens=special_tokens)
        logger.info("tokenizer初始化结束")
        dataset = []
        with open(input_path, encoding='utf-8') as f:
            for i, _id in tqdm(enumerate(tokenizer.encode_iterable(f)), desc="bpe_tokenizing"):
                dataset.append(_id)
        np.save(dataset_path, np.array(dataset, dtype=np.uint16))
    else :
        dataset = np.load(dataset_path)
        max_token = dataset.max()
        logger.info(f"Loaded dataset with max token id = {max_token}")
        assert max_token < vocab_size, f"Error: dataset max token id {max_token} >= vocab_size {vocab_size}!"
    # 已处理 9369000 行 vocav=500

    # dataset = []
    # with open(input_path, encoding='utf-8') as f:
    #     for _id in tokenizer.encode_iterable(f):
    #         dataset.append(_id)
    logger.info("分词器工作结束")
    # 3模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device, # 加入device
    ).to(device)
    # 统计参数量
    detailed_parameter_stats(model=model)
    # 4优化器
    optimizer = AdamW(model.parameters())
    # 检查参数所在设备
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.device}")

    # 5展开训练
    # 内存分析
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    model.train()
    for step in range(train_epochs):
        # 清楚所有可学习参数梯度
        x, targets = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        # print("get_batch_done")
        optimizer.zero_grad()

        x = x.to(device)
        targets = targets.to(device)

        x = model(x)


        # cross_entropy输入维度问题
        B, T, V = x.shape
        x = x.view(B * T, V)
        targets = targets.view(B * T)

        loss = run_cross_entropy(inputs=x, targets=targets)
        loss.backward()
        optimizer.step()

        # 保存
        if step % 100 == 0:
            logger.info(f'正在进行第{step}轮训练')
            logger.info(f'loss = {loss.item()}')
            # run_save_checkpoint(model=model,
            #                 optimizer=optimizer,
            #                 iteration=step,
            #                 out=os.path.join(checkpoint_path, f"checkpoint_step{step}.pt"))
        

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    logger.info("训练完成")
    # 临时写一个推理代码
    if use_bpe:
        prompt_text = "Long Long ago"
        prompt_tokens = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long)
        generated_tokens = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            context_length=context_length,
        )
        generated_text = tokenizer.decode(generated_tokens.tolist())
        logger.info(generated_text)

if __name__ == "__main__":
    main()