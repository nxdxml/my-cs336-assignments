import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()  # 推理时不计算梯度，节省内存和计算
def generate_text(
    model,
    tokenizer,
    prompt_tokens,          # 输入的初始token序列，tensor格式，形状为 (t,)
    context_length=64,
    max_new_tokens=50,      # 最大生成token数
    temperature=1.0,        # softmax温度缩放参数
    top_p=0.9,              # top-p采样阈值
    device="cuda"           # 运行设备
):
    model.eval()            # 切换模型到推理模式
    generated = prompt_tokens.clone().to(device)  # 把prompt放到设备上

    end_token_id = tokenizer.encode("<|endoftext|>")[0]  # 获取结束token的id，假设tokenizer有encoder字典

    for _ in range(max_new_tokens):
        # 保证输入长度不超过模型上下文窗口大小
        if generated.size(0) > context_length:
            input_ids = generated[-context_length:]  # 只取最新context_length个token作为输入
        else:
            input_ids = generated

        input_ids = input_ids.unsqueeze(0)  # 批量维度，变成 (1, seq_len)

        # 前向计算模型输出，得到 logits，形状为 (1, seq_len, vocab_size)
        logits = model(input_ids)

        # 取序列最后一个位置的 logits，形状 (vocab_size,)
        next_token_logits = logits[0, -1, :]

        # 温度缩放
        next_token_logits = next_token_logits / temperature

        # 计算softmax概率
        probs = F.softmax(next_token_logits, dim=-1)

        # top-p核采样实现
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        # 找到累积概率超过top_p的最小集合
        cutoff_idx = torch.searchsorted(cumulative_probs, top_p).item()

        # 过滤掉不在top-p集合里的token
        filtered_indices = sorted_indices[:cutoff_idx+1]
        filtered_probs = sorted_probs[:cutoff_idx+1]
        filtered_probs = filtered_probs / filtered_probs.sum()  # 重新归一化概率

        # 从过滤后的分布采样下一个token
        next_token = np.random.choice(filtered_indices.cpu().numpy(), p=filtered_probs.cpu().numpy())

        # 把采样出的token追加到序列
        generated = torch.cat([generated, torch.tensor([next_token], device=device)])

        # 如果采样到了结束token，停止生成
        if next_token == end_token_id:
            break

    return generated.cpu().numpy()  # 返回numpy数组形式的完整token序列

