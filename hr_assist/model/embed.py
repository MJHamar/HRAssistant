"""
Module defining different embedding models
"""
from typing import Literal, Dict, Any
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import Module

class PreTrainedEmbedder(Module):
    """
    Wrapper around an AutoModel
    """
    def __init__(self,
                 model_name_or_path: str,
                 embedding_method: Literal['cls_token', 'average_pool', 'mean_pool'] = 'cls_token'
                 ):
        super(PreTrainedEmbedder, self).__init__()
        # TODO: make a dataloader + tokenizer
        # self._tokenizer = PreTrainedEmbedder.make_tokenizer(model_name_or_path)
        self._model = AutoModel.from_pretrained(model_name_or_path)
        self.embedding_method = embedding_method

    # average_pool and mean_pool are nearly the same, mean_pool is adapted from the official code of Jina V2.
    def average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """taken from https://github.com/jasonyux/ConFit-v2"""
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def mean_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        Taken from https://github.com/jasonyux/ConFit-v2, which
        adapted from https://huggingface.co/jinaai/jina-embeddings-v2-base-zh
        """
        token_embeddings = last_hidden_states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9).to(token_embeddings.dtype)

    def _get_encoder_embedding(self, batched_encoding: dict) -> torch.Tensor:
        """
        Taken from https://github.com/jasonyux/ConFit-v2
        given a pretrained encoder model (e.g. bert), how to get the embedding vector
        """
        embedding_vec = self._model(
            **batched_encoding.to(self._model.device)
        ).last_hidden_state  # B * seq_length * word_embedding_size

        if self.embedding_method == "cls_token":
            embedding_vec = embedding_vec[:, 0, :]  # B * word_embedding_size
        elif self.embedding_method == "average_pool":
            embedding_vec = self.average_pool(
                embedding_vec, attention_mask=batched_encoding["attention_mask"]
            )  # B * word_embedding_size
        elif self.embedding_method == "mean_pool":
            embedding_vec = self.mean_pool(
                embedding_vec, attention_mask=batched_encoding["attention_mask"]
            )  # B * word_embedding_size
        else:
            raise NotImplementedError
        return embedding_vec

    def forward(self, batched_data: Dict[str, Any]):
        """
        batched_data is a dictionary of string keys mapping to tokenized inputs.
        Tokenized inputs (like in Yu et. al 2025) should be formatted as <context_taxon_token> : <tokenized text>
        """
        return self._get_encoder_embedding(batched_data)

embedder = None
def init_embedder(model_name_or_path: str, embedding_method: Literal['cls_token', 'average_pool', 'mean_pool'] = 'cls_token', embedder_cls=PreTrainedEmbedder) -> PreTrainedEmbedder:
    global embedder
    if embedder is None:
        embedder = embedder_cls(model_name_or_path=model_name_or_path, embedding_method=embedding_method)
    return embedder
