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
def main():
    # 1训练配置参数
    train_epochs = 5 # 训练步数
    input_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\data\\TinyStoriesV2-GPT4-valid-mini.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    batch_size = 4
    context_length = 32 # 序列长度
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 64 # 嵌入维度
    num_heads = 2
    num_layers = 2 # transformer block 个数
    d_ff = 4 * d_model
    rope_theta = 10000
    checkpoint_path = "C:\\Users\\asus\\Desktop\\cuda\\source\\assignment1-basics\\cs336_basics\\checkpoint"

    # 2训练BPE分词器
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
    ).to(device)
    # 4优化器
    optimizer = AdamW(model.parameters())


    # 5展开训练
    model.train()
    for step in range(train_epochs):
        print(f'正在进行第{step}轮训练')
        # 清楚所有可学习参数梯度
        x, targets = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        print("get_batch_done")
        optimizer.zero_grad()

        x = model(x)

        # cross_entropy输入维度问题
        B, T, V = x.shape
        x = x.view(B * T, V)
        targets = targets.view(B * T)

        loss = run_cross_entropy(inputs=x, targets=targets)
        print(f'loss = {loss.item()}')
        loss.backward()
        optimizer.step()

        # 保存
        run_save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=step,
                            out=os.path.join(checkpoint_path, f"checkpoint_step{step}.pt"))
        


    print("训练完成")


if __name__ == "__main__":
    main()