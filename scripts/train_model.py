
modelName = 'conformer_ctc'

args = {}
args['outputDir'] = '/home/edward/neural_seq_decoder/logs/speech_logs/' + modelName
args['datasetPath'] = '/home/edward/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.002  # Stable training
args['lrEnd'] = 0.0001  # Decay to very low LR for fine-tuning
args['nUnits'] = 1024
args['nBatch'] = 100000  # Train much longer for <0.1 CER target
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
args['model_type'] = 'transformer_ctc'

# Conformer architecture (combines conv + attention for best performance)
args['use_conformer'] = True  # Use Conformer blocks instead of standard transformer
args['conformer_conv_kernel'] = 31  # Conformer conv kernel (proven effective)

# Temporal processing (match GRU)
args['temporal_kernel'] = 32  # Match GRU kernel size
args['temporal_stride'] = 4  # Match GRU stride - reduces seq length 150 -> ~29
args['gaussian_smooth_width'] = 2.0  # Match GRU Gaussian smoothing

# Feature extraction (keep simple - no wavelets)
args['use_wavelets'] = False
args['use_pac_features'] = False

# Model architecture (large capacity - training is stable now)
args['frontend_dim'] = 1024  # Full capacity
args['latent_dim'] = 1024  # Match GRU + more
args['autoencoder_hidden_dim'] = 512
args['transformer_num_layers'] = 8  # Deep Conformer for sub-0.1 CER
args['transformer_n_heads'] = 8
args['transformer_dim_ff'] = 2048  # Large feedforward
args['transformer_dropout'] = 0.3  # Proven to work well

# Training optimization
args['optimizer'] = 'adamw'
args['weight_decay'] = 1e-3
args['warmup_steps'] = 4000
args['label_smoothing'] = 0.1  # Label smoothing for better generalization

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
