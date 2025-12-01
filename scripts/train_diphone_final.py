"""
FINAL OPTIMIZED Training Script

Fixed all configuration issues found in logic review:
1. Use CONSTANT alpha (not scheduled) - prevents loss instability
2. Multi-scale lambda 0.15 (not 0.05) - proven helpful, but not too strong
3. Tiny label smoothing 0.02 (not 0.0) - better generalization
4. Shorter warmup 20k (not 40k) - model learns fast early
5. Grad clip 1.0 (not 1.5) - more stable
"""

modelName = 'multiscale_diphone_final_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Aggressive but proven-stable LR
args['lrStart'] = 0.002
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
args['transformer_dropout'] = 0.1
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training optimization
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3

# ===========================================================================
# FIX #4: Shorter warmup (20k not 40k) - model learns fast early
# ===========================================================================
args['warmup_steps'] = 20000  # Reduced from 40k

# ===========================================================================
# FIX #5: More conservative grad clipping (1.0 not 1.5)
# ===========================================================================
args['grad_clip_norm'] = 1.0  # More stable

# ===========================================================================
# FIX #3: Tiny label smoothing (0.02 not 0.0) - better generalization
# ===========================================================================
args['label_smoothing'] = 0.02  # Tiny smoothing for generalization

args['time_mask_param'] = 20

# Diphone configuration
args['use_diphone_head'] = True
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True

# ===========================================================================
# FIX #1: CONSTANT alpha (not scheduled) - prevents loss instability
# ===========================================================================
args['diphone_alpha_schedule'] = 'constant'  # Stable 50/50 throughout

# ===========================================================================
# FIX #2: Balanced auxiliary weights (0.15 not 0.05)
# ===========================================================================
args['use_multiscale_ctc'] = True
args['multiscale_lambda_fast'] = 0.15  # Middle ground (proven helpful)
args['multiscale_lambda_slow'] = 0.15  # Not too weak, not too strong

print("=" * 80)
print("FINAL OPTIMIZED CONFIGURATION")
print("=" * 80)
print(f"Model: {modelName}")
print()
print("ALL LOGIC FIXES APPLIED:")
print(f"  1. ✓ diphone_alpha = 'constant' (not scheduled - STABLE)")
print(f"  2. ✓ multiscale_lambda = 0.15 (not 0.05 - BALANCED)")
print(f"  3. ✓ label_smoothing = 0.02 (not 0.0 - GENERALIZATION)")
print(f"  4. ✓ warmup_steps = 20k (not 40k - FASTER LEARNING)")
print(f"  5. ✓ grad_clip_norm = 1.0 (not 1.5 - MORE STABLE)")
print()
print("WHY THESE FIXES:")
print("  Issue 1: Scheduled alpha caused loss to increase (moving target)")
print("  Issue 2: Lambda 0.05 too weak (simple baseline proved it helps)")
print("  Issue 3: No smoothing → overconfident predictions")
print("  Issue 4: 40k warmup too long (model learns fast early)")
print("  Issue 5: 1.5 grad clip less stable than 1.0")
print()
print("EXPECTED: Best performance, stable training, breakthrough plateau")
print("TARGET: 0.15-0.22 CER (realistic with all fixes)")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
