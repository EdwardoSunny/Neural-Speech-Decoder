"""
ULTRA-CONSERVATIVE Multi-Scale Diphone Training Script

Start with minimal multi-scale auxiliary loss weights to ensure stability.
If this works, we can gradually increase them.
"""

modelName = 'multiscale_diphone_ultra_conservative_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Keep ALL original hyperparameters exactly as they were
args['lrStart'] = 0.0005
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

# Model architecture
args['model_type'] = 'multiscale_ctc'
args['latent_dim'] = 1024
args['transformer_num_layers'] = 6
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.3
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training optimization - ORIGINAL VALUES
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 10000
args['grad_clip_norm'] = 0.5
args['label_smoothing'] = 0.1
args['time_mask_param'] = 15

# Diphone configuration - ORIGINAL
args['use_diphone_head'] = True
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True
args['diphone_alpha_schedule'] = 'constant'

# ===========================================================================
# CRITICAL FIX: Enable multi-scale CTC with MINIMAL auxiliary loss weights
# ===========================================================================
args['use_multiscale_ctc'] = True

# Start with VERY low auxiliary loss weights for stability
# (Default was 0.3 each, but that might be too much when first enabling)
args['multiscale_lambda_fast'] = 0.1  # Low weight for fast pathway
args['multiscale_lambda_slow'] = 0.1  # Low weight for slow pathway

print("=" * 80)
print("ULTRA-CONSERVATIVE TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model: {modelName}")
print()
print("SINGLE CHANGE FROM ORIGINAL:")
print(f"  ✓ use_multiscale_ctc = True")
print(f"  ✓ multiscale_lambda_fast = 0.1 (very conservative)")
print(f"  ✓ multiscale_lambda_slow = 0.1 (very conservative)")
print()
print("RATIONALE:")
print("  - Previous 'fixed' version diverged (CER: 0.28 → 0.33)")
print("  - Likely due to too many simultaneous changes")
print("  - This version changes ONLY the critical missing feature")
print("  - Using minimal auxiliary weights for maximum stability")
print()
print("EXPECTED RESULTS:")
print("  - Stable training (no divergence)")
print("  - Modest but real improvement from multi-scale supervision")
print("  - Target: 0.18-0.22 CER (conservative due to low aux weights)")
print()
print("IF THIS WORKS:")
print("  - Can gradually increase lambda weights in future runs")
print("  - Can then try other optimizations (LR, dropout, etc.)")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
