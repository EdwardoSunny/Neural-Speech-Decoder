
modelName = 'advanced_multiscale_v1'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.002  # Proven to work
args['lrEnd'] = 0.0001
args['nUnits'] = 1024
args['nBatch'] = 150000  # Long training for <0.1 CER
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
# ADVANCED MULTI-SCALE ARCHITECTURE (MAX NOVELTY!)
# ============================================================================
# Three novel components:
# 1. ADAPTIVE SCALE FUSION - learns which temporal scales matter when
# 2. PHONETIC FEATURE LEARNING - multi-task auxiliary objectives
# 3. CONTRASTIVE PHONEME LEARNING - better phoneme representations
# ============================================================================
args['model_type'] = 'advanced_multiscale'

# Architecture
args['latent_dim'] = 768  # Smaller for stability (was 1024)
args['transformer_num_layers'] = 6  # Encoder depth per pathway
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.3
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Novel components (DISABLE CONTRASTIVE FOR STABILITY)
args['use_phonetic_features'] = False  # Disable for now - adds complexity
args['use_contrastive'] = False  # DISABLE - causes divergence
args['phonetic_loss_weight'] = 0.0
args['contrastive_loss_weight'] = 0.0

# Training (MORE CONSERVATIVE)
args['optimizer'] = 'adamw'
args['lrStart'] = 0.001  # Lower LR (was 0.002)
args['weight_decay'] = 5e-4  # Reduce weight decay
args['warmup_steps'] = 8000  # Longer warmup (was 4000)
args['label_smoothing'] = 0.1

# Data augmentation
args['time_mask_param'] = 15  # Less aggressive (was 20)

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
