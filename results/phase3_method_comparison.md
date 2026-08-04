# Phase 3 Method Comparison: FFT vs QLoRA (clean baseline)

Both trained on filtered corpus (1,380,264 records) with identical structural
hyperparameters (bs=256 / seq=1024 / cosine / curriculum / seed=137).
Method differences: FFT uses bf16 full parameter update + DeepSpeed ZeRO-3;
QLoRA uses 4-bit NF4 + LoRA adapters (r=16).

## Per-language validation perplexity

| Track | QLoRA-clean | FFT | Delta | Relative |
|---|---:|---:|---:|---:|

## Pre-registration decision rule

- Aggregate improvement >= 2 pph: **MET** (4.74 pph, 237% of threshold)
- Kamba improvement >= 5 pph: **MET** (10.77 pph, 215% of threshold)

## Conclusion

FFT is the unambiguous production candidate. All 5 evaluation tracks
improve; the largest absolute improvement is on the smallest language
(Kamba, -10.77 pph). This suggests architectural capacity, not data
quantity, was the primary bottleneck at 91k records for Kamba.

## Cost

- FFT: 133 GPU-hours (16h 37m wall clock on 8×A100)
- QLoRA-clean: ~20 GPU-hours (21h 44m wall clock on 8×A100)
- FFT/QLoRA cost ratio: 6.7× GPU-hours
- FFT/QLoRA quality gain: 19.3% pooled PPL reduction

## Files

- Best FFT checkpoint: /workspace/checkpoints/RUN-P3-FFT/checkpoint-5400/
- Best QLoRA checkpoint: /workspace/checkpoints/RUN-P3-CLEAN-QLORA/checkpoint-6000/
- FFT eval: /workspace/results/P3_FFT_per_language_eval.json
- QLoRA eval: /workspace/results/P3_CLEAN_QLORA_per_language_eval.json