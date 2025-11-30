
modelName = 'multiscale_ctc_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64  # Can use larger batch since CTC is simpler than attention
args['lrStart'] = 0.0005  # REDUCED from 0.002 - multi-scale needs lower LR
args['lrEnd'] = 0.00005
args['nUnits'] = 1024
args['nBatch'] = 150000  # Train for 150k batches for sub-0.1 CER
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
# NOVEL MULTI-SCALE CTC ARCHITECTURE (FIXED)
# ============================================================================
# Key innovation: Multi-scale temporal pyramid encoder
# - Processes neural signals at 3 temporal resolutions (fast/medium/slow)
# - Fuses information across scales with cross-attention
# - Uses proven CTC decoding (avoids attention complexity)
# ============================================================================
args['model_type'] = 'multiscale_ctc'

# Multi-scale encoder architecture
args['latent_dim'] = 1024  # Match Conformer baseline capacity
args['transformer_num_layers'] = 6  # Encoder depth per pathway
args['transformer_n_heads'] = 8  # Multi-head attention
args['transformer_dim_ff'] = 2048  # Feedforward dimension
args['transformer_dropout'] = 0.3  # Proven regularization
args['conformer_conv_kernel'] = 31  # Conformer conv kernel
args['gaussian_smooth_width'] = 2.0  # Match GRU preprocessing

# Training optimization
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 10000  # INCREASED from 4000 - longer warmup for stability
args['grad_clip_norm'] = 0.5  # STRONGER clipping (was 1.0 in trainer)
args['label_smoothing'] = 0.1  # Better generalization

# Optional: Contrastive pre-training (can enable for further improvement)
args['use_contrastive_pretraining'] = False
args['contrastive_warmup_batches'] = 2000

# Data augmentation
args['time_mask_param'] = 20  # SpecAugment time masking

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
