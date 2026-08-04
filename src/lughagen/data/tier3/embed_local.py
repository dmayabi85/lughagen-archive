"""
embed_local.py — Drop-in replacement for sentence_transformers.SentenceTransformer

Uses transformers directly with mean-pooling instead of going through
sentence_transformers, which on this environment pulls in Trainer → peft
→ awq → apex.amp and explodes.

API-compatible enough for our Tier 3 needs:
    embed = LocalSentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vecs = embed.encode(texts, batch_size=64, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)

Returns float32 numpy arrays. Same shapes as sentence_transformers.
"""

from typing import List, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("tier3.embed_local")


class LocalSentenceTransformer:
    """Drop-in replacement for SentenceTransformer (encode-only)."""

    def __init__(self, model_id: str, device: str = None, max_seq_length: int = 128):
        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_seq_length = max_seq_length

        logger.info("Loading %s (device=%s)", model_id, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        logger.info("Embedding model loaded (hidden_dim=%d)",
                    self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        **kwargs,  # swallow any extra sentence_transformers kwargs
    ):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)

            # Mean pooling over tokens, weighted by attention mask
            attn = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * attn).sum(dim=1)
            counts = attn.sum(dim=1).clamp(min=1e-9)
            mean = summed / counts

            if normalize_embeddings:
                mean = F.normalize(mean, p=2, dim=1)

            all_embs.append(mean.cpu())

            if show_progress_bar and (i // batch_size) % 10 == 0:
                logger.info("  encoded %d/%d", min(i + batch_size, len(texts)), len(texts))

        result = torch.cat(all_embs, dim=0)
        if convert_to_numpy:
            result = result.numpy().astype(np.float32)
        if single:
            result = result[0]
        return result


# Re-export with the standard name so a simple `from embed_local import SentenceTransformer`
# works as a drop-in replacement
SentenceTransformer = LocalSentenceTransformer
