#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')

modelName = 'multiscale_attention_test'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 48
args['lrStart'] = 0.001
args['lrEnd'] = 0.00005
args['nUnits'] = 1024
args['nBatch'] = 500  # Just test 500 batches
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

# Multi-scale attention architecture
args['model_type'] = 'multiscale_attention'
args['latent_dim'] = 768
args['transformer_num_layers'] = 6
args['decoder_num_layers'] = 4
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048
args['transformer_dropout'] = 0.2
args['conformer_conv_kernel'] = 31
args['gaussian_smooth_width'] = 2.0

# Training
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 6000
args['label_smoothing'] = 0.1
args['use_contrastive_pretraining'] = False
args['contrastive_warmup_batches'] = 2000
args['time_mask_param'] = 0

print("=" * 80)
print("TESTING MULTI-SCALE ATTENTION ARCHITECTURE")
print("=" * 80)
print(f"Model: {args['model_type']}")
print(f"Batches: {args['nBatch']}")
print(f"Batch size: {args['batchSize']}")
print(f"Latent dim: {args['latent_dim']}")
print("=" * 80)
print()

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)

print("\n" + "=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
