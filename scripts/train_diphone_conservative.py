"""
CONSERVATIVE Multi-Scale Diphone Training Script

Strategy: Enable ONLY the critical fix (use_multiscale_ctc) while keeping
all other hyperparameters stable and proven to work.

Previous aggressive fixes caused training instability (CER increased from 0.28 to 0.33).
This version takes a more measured approach.
"""

modelName = 'multiscale_diphone_conservative_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# ===========================================================================
# CONSERVATIVE APPROACH: Keep proven hyperparameters from stable run
# ===========================================================================
args['lrStart'] = 0.0005  # Keep original (was stable)
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
args['l2_decay'] = 1e-5

# ============================================================================
# MULTI-SCALE CTC + DIPHONE MARGINALIZATION
# ============================================================================
args['model_type'] = 'multiscale_ctc'

# Multi-scale encoder architecture
args['latent_dim'] = 1024
args['transformer_num_layers'] = 6
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.3  # Keep original
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training optimization - KEEP ORIGINAL STABLE VALUES
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 10000
args['grad_clip_norm'] = 0.5  # Keep original (was stable)
args['label_smoothing'] = 0.1

# Data augmentation
args['time_mask_param'] = 15

# ===========================================================================
# DIPHONE WITH MARGINALIZATION (PROPER DCoND)
# ===========================================================================
args['use_diphone_head'] = True
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True
args['diphone_alpha_schedule'] = 'constant'  # Keep stable 50/50

# ===========================================================================
# THE ONLY CRITICAL FIX: ENABLE MULTI-SCALE CTC
# ===========================================================================
args['use_multiscale_ctc'] = True  # This is the key missing piece!

# Multi-scale auxiliary loss weights (proven defaults)
args['multiscale_lambda_fast'] = 0.3
args['multiscale_lambda_slow'] = 0.3

print("=" * 80)
print("CONSERVATIVE TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model: {modelName}")
print(f"Output: {args['outputDir']}")
print()
print("STRATEGY: Enable ONLY the critical missing feature")
print()
print("CHANGES FROM ORIGINAL:")
print(f"  1. ✓ use_multiscale_ctc = {args['use_multiscale_ctc']} (THE KEY FIX)")
print(f"  2. ✓ Explicit multiscale_lambda_fast/slow = 0.3")
print()
print("KEPT STABLE (NOT CHANGED):")
print(f"  - lrStart = {args['lrStart']} (original stable value)")
print(f"  - grad_clip_norm = {args['grad_clip_norm']} (original stable value)")
print(f"  - transformer_dropout = {args['transformer_dropout']} (original)")
print(f"  - diphone_alpha_schedule = '{args['diphone_alpha_schedule']}' (original)")
print()
print("WHY THIS APPROACH:")
print("  - Previous run was STABLE but missing multi-scale CTC")
print("  - Aggressive LR/clipping changes caused DIVERGENCE")
print("  - This adds ONLY the missing architectural feature")
print()
print("EXPECTED RESULTS:")
print("  - Stable training (no divergence)")
print("  - CER improvement from multi-scale supervision alone")
print("  - Target: 0.15-0.20 CER (conservative estimate)")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
