"""
BREAKTHROUGH Multi-Scale Diphone Training Script

Analysis of why optimized version plateaued at 0.27:
- Loss was INCREASING (485 → 742) while CER plateaued
- Scheduled alpha changes loss composition during training (unstable)
- Need stronger optimization to break through 0.27 plateau

This version: Aggressive optimization + CONSTANT alpha for stability
"""

modelName = 'multiscale_diphone_breakthrough_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# ===========================================================================
# AGGRESSIVE LEARNING RATE with LONG WARMUP (key to stability)
# ===========================================================================
args['lrStart'] = 0.0015  # 3x original (proven in research)
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

# Reduced weight decay to allow more capacity
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

# Minimal dropout for maximum capacity
args['transformer_dropout'] = 0.15  # Reduced from 0.2
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# ===========================================================================
# AGGRESSIVE TRAINING OPTIMIZATION
# ===========================================================================
args['optimizer'] = 'adamw'

# CRITICAL: Very long warmup (20% of training) for stability with high LR
args['warmup_steps'] = 30000  # 3x original - this is KEY for stability!

# Standard transformer gradient clipping
args['grad_clip_norm'] = 1.0

# Stronger label smoothing
args['label_smoothing'] = 0.15  # Increased from 0.1

# More aggressive data augmentation
args['time_mask_param'] = 20  # Increased from 15

# ===========================================================================
# DIPHONE WITH MARGINALIZATION
# ===========================================================================
args['use_diphone_head'] = True
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True

# ===========================================================================
# CRITICAL FIX: Use CONSTANT alpha (not scheduled)
# ===========================================================================
# Optimized version used 'scheduled' and loss increased during training
# Constant alpha provides stable loss composition throughout training
args['diphone_alpha_schedule'] = 'constant'  # 50/50 stable weighting

# ===========================================================================
# MULTI-SCALE CTC WITH STRONGER SUPERVISION
# ===========================================================================
args['use_multiscale_ctc'] = True

# Higher auxiliary weights for stronger gradient signal
args['multiscale_lambda_fast'] = 0.4  # Increased from 0.3
args['multiscale_lambda_slow'] = 0.4  # Increased from 0.3

print("=" * 80)
print("BREAKTHROUGH TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model: {modelName}")
print(f"Output: {args['outputDir']}")
print()
print("WHY OPTIMIZED VERSION PLATEAUED AT 0.27:")
print("  ✗ Scheduled alpha → loss increased (485 → 742)")
print("  ✗ Not aggressive enough to break through plateau")
print("  ✗ Plateaued at batch 2300 (only 1.5% through training!)")
print()
print("BREAKTHROUGH FIXES:")
print()
print("LEARNING (AGGRESSIVE):")
print(f"  ✓ lrStart = {args['lrStart']} (3x original, not 2x)")
print(f"  ✓ warmup_steps = {args['warmup_steps']} (20% of training - KEY for stability!)")
print(f"  ✓ grad_clip_norm = {args['grad_clip_norm']}")
print()
print("STABILITY:")
print(f"  ✓ diphone_alpha_schedule = '{args['diphone_alpha_schedule']}' (CONSTANT not scheduled!)")
print(f"  ✓ Long warmup prevents early divergence")
print()
print("CAPACITY:")
print(f"  ✓ transformer_dropout = {args['transformer_dropout']} (minimal)")
print(f"  ✓ weight_decay = {args['weight_decay']} (reduced)")
print(f"  ✓ multiscale_lambda = 0.4 (stronger auxiliary supervision)")
print()
print("THE PLAN:")
print("  1. 3x LR allows escaping the 0.27 plateau")
print("  2. 30k warmup steps (20% of training) prevents divergence")
print("  3. Constant alpha keeps loss stable (no increase like before)")
print("  4. Stronger multi-scale supervision (λ=0.4)")
print("  5. Less regularization allows model to use full capacity")
print()
print("EXPECTED TRAJECTORY:")
print("  - Batches 0-1000: Slow warmup (preventing divergence)")
print("  - Batches 1000-5000: Rapid learning (should drop below 0.25)")
print("  - Batches 5000-20000: Break through 0.20 barrier")
print("  - Batches 20000+: Converge to 0.10-0.15 CER")
print()
print("TARGET: 0.10-0.15 CER (achieves your goal!)")
print()
print("MONITORING:")
print("  Watch batch 2000-5000: Should be < 0.25 CER (not plateaued like before)")
print("  If CER > 0.30 at batch 5000 → diverging, stop")
print("  If CER plateaus at 0.25 → need even more aggressive approach")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
