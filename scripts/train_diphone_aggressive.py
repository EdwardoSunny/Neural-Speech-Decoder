"""
AGGRESSIVE Multi-Scale Diphone Training Script

Push for maximum performance with research-backed hyperparameters.
Higher risk of instability but potentially better final CER.

Use this if the optimized version is too conservative for your target of 0.1 CER.
"""

modelName = 'multiscale_diphone_aggressive_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# ===========================================================================
# AGGRESSIVE LEARNING RATE: Based on transformer research
# ===========================================================================
# Research shows transformers work well with LR 1e-3 to 5e-3
# With proper warmup, even 3e-3 can be stable
args['lrStart'] = 0.0015  # 3x original (between 2x and 4x)
args['lrEnd'] = 0.00005

args['nUnits'] = 1024
args['nBatch'] = 150000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True

# ===========================================================================
# REDUCED WEIGHT DECAY: Allow more capacity
# ===========================================================================
args['l2_decay'] = 1e-5
args['weight_decay'] = 5e-4  # Reduced from 1e-3

# ============================================================================
# MULTI-SCALE CTC + DIPHONE MARGINALIZATION
# ============================================================================
args['model_type'] = 'multiscale_ctc'
args['latent_dim'] = 1024
args['transformer_num_layers'] = 6
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048

# ===========================================================================
# MINIMAL DROPOUT: Research shows lower is often better
# ===========================================================================
args['transformer_dropout'] = 0.15  # Lower than 0.2 (from research)
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# ===========================================================================
# AGGRESSIVE TRAINING OPTIMIZATION
# ===========================================================================
args['optimizer'] = 'adamw'

# Much longer warmup for higher peak LR (20% of training)
# This is key to stability with aggressive LR
args['warmup_steps'] = 30000  # 3x original

# Standard gradient clipping for transformers
args['grad_clip_norm'] = 1.0

# Stronger label smoothing for better generalization
args['label_smoothing'] = 0.15  # Increased from 0.1

# More aggressive data augmentation
args['time_mask_param'] = 20  # Increased from 15

# ===========================================================================
# DIPHONE WITH MARGINALIZATION
# ===========================================================================
args['use_diphone_head'] = True
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True

# Scheduled alpha with more aggressive phoneme focus
args['diphone_alpha_schedule'] = 'scheduled'

# ===========================================================================
# MULTI-SCALE CTC WITH STRONGER AUXILIARY SUPERVISION
# ===========================================================================
args['use_multiscale_ctc'] = True

# Higher auxiliary weights for stronger multi-scale supervision
# Research shows 0.3-0.5 works well
args['multiscale_lambda_fast'] = 0.4  # Stronger fast pathway
args['multiscale_lambda_slow'] = 0.4  # Stronger slow pathway

print("=" * 80)
print("AGGRESSIVE TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model: {modelName}")
print(f"Output: {args['outputDir']}")
print()
print("AGGRESSIVE OPTIMIZATIONS (push for 0.10 CER):")
print()
print("ARCHITECTURE:")
print(f"  ✓ use_multiscale_ctc = {args['use_multiscale_ctc']}")
print(f"  ✓ multiscale_lambda_fast/slow = 0.4 (STRONG supervision)")
print()
print("LEARNING (AGGRESSIVE):")
print(f"  ✓ lrStart = {args['lrStart']} (3x original)")
print(f"  ✓ warmup_steps = {args['warmup_steps']} (3x original, 20% of training)")
print(f"  ✓ grad_clip_norm = {args['grad_clip_norm']} (standard)")
print()
print("REGULARIZATION (REDUCED):")
print(f"  ✓ transformer_dropout = {args['transformer_dropout']} (minimal)")
print(f"  ✓ weight_decay = {args['weight_decay']} (reduced)")
print(f"  ✓ label_smoothing = {args['label_smoothing']} (stronger)")
print(f"  ✓ time_mask_param = {args['time_mask_param']} (more augmentation)")
print()
print("STRATEGY:")
print("  1. Higher LR with long warmup (20% warmup is key to stability)")
print("  2. Less dropout/weight decay (allow model to use full capacity)")
print("  3. Stronger multi-scale supervision (λ=0.4 instead of 0.3)")
print("  4. More data augmentation (time masking)")
print("  5. Scheduled alpha (better phoneme focus)")
print()
print("EXPECTED RESULTS:")
print("  - BEST CASE: 0.08-0.12 CER (achieves your target!)")
print("  - LIKELY: 0.10-0.15 CER (strong improvement)")
print("  - WORST CASE: May diverge if LR still too high")
print()
print("WHEN TO USE:")
print("  - If optimized version is stable but CER > 0.15")
print("  - If you need to push closer to 0.10 target")
print()
print("MONITORING:")
print("  - Watch for divergence in first 1000 batches")
print("  - If CER increases after batch 2000, stop and use optimized version")
print("  - Long warmup should prevent early instability")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
