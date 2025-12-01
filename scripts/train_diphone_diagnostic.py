"""
DIAGNOSTIC Training Script

Hypothesis: Multi-scale auxiliary heads (lambda=0.3-0.4) might be HURTING performance
by creating conflicting gradient signals.

This version: DISABLE auxiliary heads entirely to test if they help or hurt.
"""

modelName = 'multiscale_diphone_diagnostic_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64

# Use breakthrough version's proven-stable hyperparameters
args['lrStart'] = 0.0015  # 3x (proven stable)
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
args['weight_decay'] = 5e-4

# Model architecture
args['model_type'] = 'multiscale_ctc'
args['latent_dim'] = 1024
args['transformer_num_layers'] = 6
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.15
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training
args['optimizer'] = 'adamw'
args['warmup_steps'] = 30000
args['grad_clip_norm'] = 1.0

# ===========================================================================
# CRITICAL DIAGNOSTIC: Reduce label smoothing (was 0.15)
# ===========================================================================
args['label_smoothing'] = 0.05  # Much lower - allow model to fit better

args['time_mask_param'] = 20

# Diphone configuration
args['use_diphone_head'] = True
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['use_diphone_marginalization'] = True
args['diphone_alpha_schedule'] = 'constant'

# ===========================================================================
# DIAGNOSTIC TEST: DISABLE multi-scale auxiliary heads
# ===========================================================================
# Set to False to see if auxiliary heads are helping or hurting
args['use_multiscale_ctc'] = False  # DISABLED FOR TESTING

# If we DO enable them, use MINIMAL weights
args['multiscale_lambda_fast'] = 0.1  # Very low (was 0.4)
args['multiscale_lambda_slow'] = 0.1  # Very low (was 0.4)

print("=" * 80)
print("DIAGNOSTIC TRAINING CONFIGURATION")
print("=" * 80)
print(f"Model: {modelName}")
print()
print("HYPOTHESIS TO TEST:")
print("  Multi-scale auxiliary heads might be HURTING performance")
print("  by creating conflicting gradient signals.")
print()
print("DIAGNOSTIC CHANGES:")
print(f"  ✓ use_multiscale_ctc = {args['use_multiscale_ctc']} (DISABLED)")
print(f"  ✓ label_smoothing = {args['label_smoothing']} (reduced from 0.15)")
print()
print("KEPT FROM BREAKTHROUGH:")
print(f"  ✓ lrStart = {args['lrStart']} (proven stable)")
print(f"  ✓ warmup_steps = {args['warmup_steps']}")
print(f"  ✓ transformer_dropout = {args['transformer_dropout']}")
print()
print("IF THIS BREAKS THROUGH 0.25:")
print("  → Multi-scale auxiliary heads were HURTING performance")
print("  → Solution: Keep them disabled OR use very low lambda (<0.1)")
print()
print("IF THIS ALSO PLATEAUS AT 0.27:")
print("  → Issue is NOT multi-scale auxiliary heads")
print("  → Likely: alpha weighting, label quality, or architecture")
print()
print("TARGET: Break through 0.25 CER by batch 10000")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
