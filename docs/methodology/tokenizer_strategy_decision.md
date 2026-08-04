# Tokenizer Strategy Decision — Gemma 2 27B FFT

## Decision

Resize Gemma 2 27B from its native 256K SentencePiece vocabulary to the strategy_c
8K vocabulary used by LughaGen. This produces methodology parity for direct
raw-quality comparison against LughaGen FFT.

## Why not keep Gemma native tokenizer

Keeping Gemma 2 27B at 256K vocab would produce results not comparable to LughaGen:

- Different token counts per input mean different training dynamics
- Embedding params dominate at 256K vocab (256K × 4608 = 1.18B params for embeddings alone)
- East African text tokenizes inefficiently under a globally-optimized vocabulary
- Any perplexity comparison is on different token distributions

The point of running this experiment is head-to-head comparison. If methodology
diverges from LughaGen, we cannot answer the research question cleanly.

## What the resize entails

The same class of engineering work Derek did for Llama-3.1-8B, applied to Gemma 2 27B.

Steps required:

1. Load Gemma 2 27B state_dict from downloaded weights
2. Identify embedding tensor keys (embed_tokens, lm_head or shared depending on tie_word_embeddings)
3. Compute remap: for each token in strategy_c 8K vocab, find corresponding position in
   Gemma native tokenizer (or use initialization heuristic for tokens without direct mapping)
4. Extract 8K rows from Gemma embedding matrix into new tensor
5. Resize model config to vocab_size=8000
6. Save remapped weights + config as new checkpoint at
   /workspace/gemma2_27b_prep/models/gemma-2-27b-strategy_c/

## Gemma-specific complications

Gemma 2 differs from Llama-3.1 architecturally in ways that may affect the remap:

- tie_word_embeddings: Gemma 2 defaults to True (same as Llama). Expect the same
  PEFT save bug pattern Derek fixed for Llama. Manual patch of lm_head after
  training may be needed if using PEFT wrapping.

- Logit soft-capping: Gemma 2 applies tanh-based logit soft-capping (final_logit_softcapping)
  which Llama does not. This lives in the model forward pass, not the tokenizer, but
  affects perplexity calculation — softcapping compresses the loss landscape. Any
  comparison to LughaGen needs to note this architectural difference.

- Sliding window attention: Gemma 2 alternates between global and sliding window
  attention layers. Does not affect the tokenizer resize but may affect training
  dynamics differently from Llama's uniform attention.

- Embedding scaling: Gemma 2 multiplies embeddings by sqrt(hidden_size) at input.
  If the strategy_c remap changes hidden_size, this constant needs adjustment.
  Gemma 2 27B hidden_size is 4608, not 4096 like Llama, so scaling constants differ.

## Estimated engineering effort

3-5 days of Derek's time to:

- Write remap script adapted from Llama version
- Handle tie_word_embeddings save/load correctly
- Smoke test on 100 steps to verify pipeline
- Debug any Gemma-specific issues that surface

Contingency: if remap turns out to be substantially harder for Gemma than expected,
fallback is to keep Gemma native tokenizer and document the comparison caveats
explicitly in Chapter 6.

## What we learn from this experiment

Under matched methodology (same corpus, same tokenizer strategy, same training procedure):

- Does 27B > 8B raw quality justify 5x compute cost?
- Does Gemma 2 base pretraining exposure to African languages (per AfroBench
  strongest open baseline) transfer to native language modeling?
- Where does LughaGen sit on the Pareto frontier of (quality, compute)?

## Fallback if resize fails

If tokenizer resize proves infeasible within reasonable engineering time (>1 week),
fall back to keeping Gemma native 256K tokenizer and:

- Document the tokenization difference in Chapter 6
- Report both raw perplexity and normalized-by-tokens perplexity
- Acknowledge comparison is approximate, not exact

Better to have approximate comparison than no comparison.
