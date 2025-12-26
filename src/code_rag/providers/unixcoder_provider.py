"""UniXcoder embedding provider for code-specific embeddings.

UniXcoder is a unified cross-modal pre-trained model for programming languages.
It provides 768-dimensional embeddings optimized for code understanding.

Requires optional dependencies:
    pip install torch transformers

Usage:
    from code_rag.providers.unixcoder_provider import UniXcoderEmbeddingProvider
    provider = UniXcoderEmbeddingProvider()
    embedding = await provider.embed("def hello(): pass")
"""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

from code_rag.providers.base import BaseEmbeddingProvider, ProviderConfig

logger = logging.getLogger(__name__)

# Check for required dependencies
_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import transformers
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def has_unixcoder_dependencies() -> bool:
    """Check if UniXcoder dependencies are available."""
    return _TORCH_AVAILABLE and _TRANSFORMERS_AVAILABLE


# Only define the model if dependencies are available
if has_unixcoder_dependencies():
    import numpy as np
    import torch
    import torch.nn as nn
    from numpy.typing import NDArray
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

    class UniXcoder(nn.Module):
        """UniXcoder model for code embeddings.

        Microsoft's UniXcoder: A unified cross-modal pre-trained model for
        programming languages. Produces 768-dimensional embeddings.
        """

        def __init__(self, model_name: str = "microsoft/unixcoder-base") -> None:
            """Initialize UniXcoder.

            Args:
                model_name: Huggingface model card name.
            """
            super().__init__()
            self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.config: RobertaConfig = RobertaConfig.from_pretrained(model_name)
            self.config.is_decoder = True
            self.model: RobertaModel = RobertaModel.from_pretrained(
                model_name, config=self.config
            )

            self.register_buffer(
                "bias",
                torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024),
            )
            self.lm_head: nn.Linear = nn.Linear(
                self.config.hidden_size, self.config.vocab_size, bias=False
            )
            self.lm_head.weight = self.model.embeddings.word_embeddings.weight
            self.lsm: nn.LogSoftmax = nn.LogSoftmax(dim=-1)

            self.tokenizer.add_tokens(["<mask0>"], special_tokens=True)

        def tokenize(
            self,
            inputs: list[str],
            mode: str = "<encoder-only>",
            max_length: int = 512,
            padding: bool = False,
        ) -> list[list[int]]:
            """Convert strings to token IDs.

            Args:
                inputs: List of input strings.
                max_length: Maximum sequence length.
                padding: Whether to pad to max_length.
                mode: Encoding mode (<encoder-only>, <decoder-only>, <encoder-decoder>).

            Returns:
                List of token ID lists.
            """
            assert mode in ["<encoder-only>", "<decoder-only>", "<encoder-decoder>"]
            assert max_length < 1024

            tokens_ids = []
            for x in inputs:
                tokens = self.tokenizer.tokenize(x)
                if mode == "<encoder-only>":
                    tokens = tokens[: max_length - 4]
                    tokens = (
                        [self.tokenizer.cls_token, mode, self.tokenizer.sep_token]
                        + tokens
                        + [self.tokenizer.sep_token]
                    )
                elif mode == "<decoder-only>":
                    tokens = tokens[-(max_length - 3) :]
                    tokens = [self.tokenizer.cls_token, mode, self.tokenizer.sep_token] + tokens
                else:
                    tokens = tokens[: max_length - 5]
                    tokens = (
                        [self.tokenizer.cls_token, mode, self.tokenizer.sep_token]
                        + tokens
                        + [self.tokenizer.sep_token]
                    )

                tokens_id: list[int] = self.tokenizer.convert_tokens_to_ids(tokens)
                if padding:
                    pad_id = self.config.pad_token_id
                    assert pad_id is not None
                    tokens_id = tokens_id + [pad_id] * (max_length - len(tokens_id))
                tokens_ids.append(tokens_id)
            return tokens_ids

        def forward(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Get token and sentence embeddings.

            Args:
                source_ids: Token ID tensor.

            Returns:
                Tuple of (token_embeddings, sentence_embeddings).
            """
            pad_id = self.config.pad_token_id
            assert pad_id is not None
            mask = source_ids.ne(pad_id)
            token_embeddings = self.model(
                source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2)
            )[0]
            sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(
                -1
            ).unsqueeze(-1)
            return token_embeddings, sentence_embeddings

    @functools.lru_cache(maxsize=1)
    def get_unixcoder_model() -> UniXcoder:
        """Get or create UniXcoder model singleton.

        Uses LRU cache for singleton pattern with lazy initialization.

        Returns:
            UniXcoder model instance.
        """
        logger.info("Loading UniXcoder model (microsoft/unixcoder-base)...")
        model = UniXcoder("microsoft/unixcoder-base")
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("UniXcoder loaded on CUDA")
        else:
            logger.info("UniXcoder loaded on CPU")
        return model

    def embed_code_sync(code: str, max_length: int = 512) -> list[float]:
        """Synchronously embed code using UniXcoder.

        Args:
            code: Source code to embed.
            max_length: Maximum token length.

        Returns:
            768-dimensional embedding.
        """
        model = get_unixcoder_model()
        device = next(model.parameters()).device
        tokens = model.tokenize([code], max_length=max_length)
        tokens_tensor = torch.tensor(tokens).to(device)
        with torch.no_grad():
            _, sentence_embeddings = model(tokens_tensor)
            embedding: NDArray[np.float32] = sentence_embeddings.cpu().numpy()
        return embedding[0].tolist()

    def embed_batch_sync(codes: list[str], max_length: int = 512) -> list[list[float]]:
        """Synchronously embed multiple code snippets.

        Args:
            codes: List of source code strings.
            max_length: Maximum token length.

        Returns:
            List of 768-dimensional embeddings.
        """
        if not codes:
            return []

        model = get_unixcoder_model()
        device = next(model.parameters()).device
        tokens = model.tokenize(codes, max_length=max_length)
        tokens_tensor = torch.tensor(tokens).to(device)
        with torch.no_grad():
            _, sentence_embeddings = model(tokens_tensor)
            embeddings: NDArray[np.float32] = sentence_embeddings.cpu().numpy()
        return [emb.tolist() for emb in embeddings]


class UniXcoderEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using Microsoft's UniXcoder model.

    UniXcoder is optimized for code understanding and produces 768-dimensional
    embeddings. This is better suited for code search than general-purpose
    embeddings like OpenAI's text-embedding models.

    Requires: torch, transformers
    Install: pip install torch transformers
    """

    # Embedding dimension for UniXcoder
    EMBEDDING_DIM = 768

    def __init__(
        self,
        config: ProviderConfig | None = None,
        max_length: int = 512,
    ):
        """Initialize UniXcoder embedding provider.

        Args:
            config: Provider configuration (optional for UniXcoder).
            max_length: Maximum token length for inputs.

        Raises:
            RuntimeError: If torch or transformers are not installed.
        """
        if not has_unixcoder_dependencies():
            raise RuntimeError(
                "UniXcoder requires 'torch' and 'transformers' packages. "
                "Install with: pip install torch transformers"
            )

        # Use default config if not provided
        if config is None:
            config = ProviderConfig(
                provider="unixcoder",
                model="microsoft/unixcoder-base",
            )

        super().__init__(config)
        self.max_length = max_length

        # Thread pool for running sync torch code
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Pre-load the model
        logger.info("Initializing UniXcoder embedding provider...")

    async def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using UniXcoder.

        Args:
            texts: List of code strings to embed.

        Returns:
            List of 768-dimensional embeddings.
        """
        loop = asyncio.get_event_loop()

        # Run synchronous torch code in thread pool
        embeddings = await loop.run_in_executor(
            self._executor,
            embed_batch_sync,
            texts,
            self.max_length,
        )

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.EMBEDDING_DIM

    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
