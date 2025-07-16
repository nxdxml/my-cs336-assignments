"""
Problem (embedding): Implement the embedding module (1 point)
Deliverable: Implement the Embedding class that inherits from torch.nn.Module and performs an
embedding lookup. Your implementation should follow the interface of PyTorch's built-in
nn.Embedding module. We recommend the following interface:
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) Construct
an embedding module. This function should accept the following parameters:
num_embeddings: int Size of the vocabulary
embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
def forward(self, token_ids: torch.Tensor) -> torch.Tensor Lookup the embedding vectors
for the given token IDs.
Make sure to:
• subclass nn.Module
• call the superclass constructor
• initialize your embedding matrix as a nn.Parameter
• store the embedding matrix with the d_model being the final dimension
• of course, don't use nn.Embedding or nn.functional.embedding
Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to
initialize the weights.
To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, run
uv run pytest -k test_embedding.

"""
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings : int, 
                 embedding_dim : int, 
                 device : torch.device | None = None,
                 dtype : torch.dtype | None = None,
                 ):
        """初始化

        Args:
            num_embeddings (int): 词表大小
            embedding_dim (int): 嵌入的维度
            device (torch.device | None, optional): 设备
            dtype (torch.dtype | None, optional): 数据类型
        """        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)

    def forward(self, token_id : torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_id (torch.Tensor): 词表 (batch_size, sequence_length)

        Returns:
            torch.Tensor: 查询结果 (batch_size, sequence_length, vocab_size)
        """        
        # TODO 向量化操作
        # batch_size, sequence_length = token_id.shape
        # output = torch.empty(batch_size, sequence_length, self.embedding_dim)
        # for i, seq in enumerate(token_id):
        #     for j, id in enumerate(seq):
        #         output[i][j] = self.W[id]
        output = self.W[token_id]
        return output
