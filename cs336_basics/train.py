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

def main():
    # 1训练配置参数
    train_epochs = 100 # 训练步数
    input_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\data\\TinyStoriesV2-GPT4-valid-mini.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    batch_size = 16
    context_length = 64 # 序列长度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 128 # 嵌入维度
    num_heads = 4
    num_layers = 4 # transformer block 个数
    d_ff = 4 * d_model 
    rope_theta = 10000
    checkpoint_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\cs336_basics\\checkpoint"
    use_bpe = True # 是否需要使用bpe
    dataset_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\data\\dataset.npy"

    # print(f"device = {device}")

    # 2训练BPE分词器
    if use_bpe:
        print("train_bpe开始")
        start = time.time()
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        print("train_bpe结束")
        tokenizer = get_tokenizer(vocab=vocab,
                                merges=merges,
                                special_tokens=special_tokens)
        end = time.time()
        print(f"耗时：{end - start:.4f} 秒")
        print("tokenizer初始化结束")
        # 90s
        dataset = []
        with open(input_path, encoding='utf-8') as f:
            for i, _id in enumerate(tokenizer.encode_iterable(f)):
                dataset.append(_id)
                if i % 10000 == 0:
                    print(f"已处理 {i} 行")
        np.save(dataset_path, np.array(dataset, dtype=np.uint16))
    else :
        dataset = np.load(dataset_path)
        max_token = dataset.max()
        print(f"Loaded dataset with max token id = {max_token}")
        assert max_token < vocab_size, f"Error: dataset max token id {max_token} >= vocab_size {vocab_size}!"
    # 已处理 9369000 行 vocav=500

    # dataset = []
    # with open(input_path, encoding='utf-8') as f:
    #     for _id in tokenizer.encode_iterable(f):
    #         dataset.append(_id)
    print("分词器工作结束")
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
    # 4优化器
    optimizer = AdamW(model.parameters())
    # 检查参数所在设备
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.device}")

    # 5展开训练
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
            print(f'正在进行第{step}轮训练')
            print(f'loss = {loss.item()}')
            # run_save_checkpoint(model=model,
            #                 optimizer=optimizer,
            #                 iteration=step,
            #                 out=os.path.join(checkpoint_path, f"checkpoint_step{step}.pt"))
        


    print("训练完成")
    # 临时写一个推理代码
    if use_bpe:
        prompt_text = "Once upon a time"
        prompt_tokens = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long)
        generated_tokens = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            context_length=context_length,
        )
        generated_text = tokenizer.decode(generated_tokens.tolist())
        print(generated_text)

if __name__ == "__main__":
    main()