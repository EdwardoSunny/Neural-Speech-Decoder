"""
RADICAL SIMPLIFICATION

All complex approaches plateau at 0.27. Try simplest possible architecture:
- NO multi-scale encoder (single scale only)
- NO diphone marginalization (direct phoneme prediction)
- NO auxiliary heads
- Just: Encoder → Phoneme CTC

If this works better, the problem was architecture complexity.
If this also plateaus at 0.27, the problem is data/labels.
"""

modelName = 'simple_baseline_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 32  # Reduced to avoid OOM

# Proven-stable hyperparameters
args['lrStart'] = 0.001
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

# ===========================================================================
# RADICAL SIMPLIFICATION: Use basic transformer (NO multi-scale!)
# ===========================================================================
args['model_type'] = 'transformer_ctc'  # NOT multiscale_ctc!

# Standard transformer settings
args['latent_dim'] = 512
args['transformer_num_layers'] = 6
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.1
args['use_conformer'] = True
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 15000
args['grad_clip_norm'] = 1.0
args['label_smoothing'] = 0.0  # No smoothing
args['time_mask_param'] = 20

# ===========================================================================
# NO DIPHONE - Direct phoneme prediction only
# ===========================================================================
# (Don't set use_diphone_head - defaults to False)

print("=" * 80)
print("RADICAL SIMPLIFICATION TEST")
print("=" * 80)
print(f"Model: {modelName}")
print()
print("HYPOTHESIS:")
print("  Multi-scale architecture / diphone marginalization is causing the plateau.")
print()
print("SIMPLIFICATIONS:")
print("  ✗ NO multi-scale encoder (single-scale conformer)")
print("  ✗ NO diphone marginalization")
print("  ✗ NO auxiliary heads")
print("  ✓ Just: Conformer → Phoneme CTC (simplest possible)")
print()
print("IF THIS BREAKS THROUGH 0.25:")
print("  → Multi-scale/diphone was the problem")
print("  → Use this simple architecture instead")
print()
print("IF THIS ALSO PLATEAUS AT 0.27:")
print("  → Problem is data quality (27% label noise)")
print("  → OR evaluation (need beam search decoding)")
print()
print("TARGET: Test if simple = better")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
