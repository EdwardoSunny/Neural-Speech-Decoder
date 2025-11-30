
modelName = 'multiscale_diphone_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.0005  # Conservative LR (fixed from divergence)
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
# MULTI-SCALE CTC + DIPHONE AUXILIARY HEAD
# ============================================================================
# Diphone head provides context-aware supervision to improve phoneme decoding
# Based on DCoND paper approach
# ============================================================================
args['model_type'] = 'multiscale_ctc'

# Multi-scale encoder architecture
args['latent_dim'] = 1024
args['transformer_num_layers'] = 6
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.3
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training optimization (FIXED for stability)
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 10000  # Longer warmup for stability
args['grad_clip_norm'] = 0.5  # Stronger clipping
args['label_smoothing'] = 0.1

# Data augmentation
args['time_mask_param'] = 15  # Less aggressive (was 20)

# ===========================================================================
# DIPHONE AUXILIARY HEAD CONFIGURATION (NEW!)
# ===========================================================================
args['use_diphone_head'] = True  # Enable diphone auxiliary head
args['diphone_vocab_path'] = '/home/edward/neural_seq_decoder/diphone_vocab.pkl'
args['diphone_alpha_schedule'] = 'constant'  # Options: 'constant' or 'scheduled'
# With 'constant': 50/50 weighting between phoneme and diphone loss
# With 'scheduled': Paper-style schedule (start with diphones, end with phonemes)

# Note: Diphone head adds ~1M parameters (1024 -> 1012 diphones)
# Total model: ~242M parameters (241M base + 1M diphone head)

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
