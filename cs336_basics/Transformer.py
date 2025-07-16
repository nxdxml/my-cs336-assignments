from cs336_basics.RMSnorm import RMSnorm
from cs336_basics.Attention import MultiheadSelfAttention
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.Embedding import Embedding
from cs336_basics.Linear import Linear
from cs336_basics.Attention import softmax
import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self,
                    d_model: int,
                    num_heads: int,
                    d_ff: int,
                    max_seq_len: int,
                    theta: float,
                    device: torch.device | str = "cpu", # 加入device
                 ):
        """

        Args:
            d_model (int): Dimensionality of the feed-forward inner layer.
            num_heads (int): _description_
            d_ff (int): Dimensionality of the feed-forward inner layer.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            theta (float): theta (float): RoPE parameter.
            weights (dict):             
                State dict of our reference implementation.
                The keys of this dictionary are:
                - `attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (d_model, d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is (d_model, d_model).
                - `ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
        """        
        super().__init__()

        self.rms_norm1 = RMSnorm(d_model=d_model, device=device)

        self.multi_head_attn = MultiheadSelfAttention(
            d_model=d_model,
            num_head=num_heads,
            apply_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device, # 加入device
        )

        self.rms_norm2 = RMSnorm(d_model=d_model, device=device)

        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device)

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # print(f"transformer {x.device}")
        y = x + self.multi_head_attn(self.rms_norm1(x))
        output = y + self.swiglu(self.rms_norm2(y))

        return output
    


class TransformerLM(nn.Module):
    def __init__(self,
                    vocab_size: int,
                    context_length: int,
                    d_model: int,
                    num_layers: int,
                    num_heads: int,
                    d_ff: int,
                    rope_theta: float,
                    device: torch.device | str = "cpu", # 加入device
                 ):
        """Given the weights of a Transformer language model and input indices,return the output of running a forward pass on the input indices.
        This function should use RoPE.

        Args:
            vocab_size (int): The number of unique items in the output vocabulary to be predicted.
            context_length (int): The maximum number of tokens to process at once.
            d_model (int): The dimensionality of the model embeddings and sublayer outputs.
            num_layers (int): The number of Transformer layers to use.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
            rope_theta (float): The RoPE $\Theta$ parameter.
            weights (dict[str, Tensor]): 
                State dict of our reference implementation. {num_layers} refers to an
                integer between `0` and `num_layers - 1` (the layer index).
                The keys of this dictionary are:
                - `token_embeddings.weight`
                    Token embedding matrix. Shape is (vocab_size, d_model).
                - `layers.{num_layers}.attn.q_proj.weight`
                    The query projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.k_proj.weight`
                    The key projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_k),
                    so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.v_proj.weight`
                    The value projections for all `num_heads` attention heads.
                    Shape is (num_heads * (d_model / num_heads), d_model).
                    The rows are ordered by matrices of shape (num_heads, d_v),
                    so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
                - `layers.{num_layers}.attn.output_proj.weight`
                    Weight of the multi-head self-attention output projection
                    Shape is ((d_model / num_heads) * num_heads, d_model).
                - `layers.{num_layers}.ln1.weight`
                    Weights of affine transform for the first RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `layers.{num_layers}.ffn.w1.weight`
                    Weight of the first linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `layers.{num_layers}.ffn.w2.weight`
                    Weight of the second linear transformation in the FFN.
                    Shape is (d_ff, d_model).
                - `layers.{num_layers}.ffn.w3.weight`
                    Weight of the third linear transformation in the FFN.
                    Shape is (d_model, d_ff).
                - `layers.{num_layers}.ln2.weight`
                    Weights of affine transform for the second RMSNorm
                    applied in the transformer block.
                    Shape is (d_model,).
                - `ln_final.weight`
                    Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                    Shape is (d_model, ).
                - `lm_head.weight`
                    Weights of the language model output embedding.
                    Shape is (vocab_size, d_model).
            in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
                `sequence_length` is at most `context_length`.

        Returns:
            Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
            next-word distribution for each token.
        """
        super().__init__()

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model,device=device)


        self.layers = nn.ModuleList(
            Transformer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device, # 加入device
            )
            for _ in range(num_layers)
        )
        
        self.rms_norm = RMSnorm(d_model=d_model, device=device)

        self.ln_out = Linear(d_model, vocab_size, device=device)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """        
        # print(f"transformerlm {x.device}") # transformerlm cuda:0
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.rms_norm(x)
        x = self.ln_out(x)
        # 这里加softmax并不合理，会影响数值稳定性甚至损害训练效果。
        # x = softmax(x, -1)
        return x