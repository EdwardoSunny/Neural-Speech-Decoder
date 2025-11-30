# Multi-Scale CTC Divergence Fix

## Problem Diagnosis

Both Advanced Multi-Scale and Multi-Scale CTC models diverged at **batch 3000-4000** with identical patterns:
- Batch 700: CER=0.389 (good learning)
- Batch 3000-4000: CER jumps to 0.75-1.0 (divergence)
- Loss explodes from ~2400 to ~6000+

## Root Cause

**NOT** the contrastive loss (as initially thought) - the simpler Multi-Scale CTC without contrastive loss also diverged at the same point.

**ACTUAL CAUSE**: Learning rate too high when warmup ends
- Warmup ends at batch 4000
- LR reaches peak of **0.002** - too aggressive for complex multi-scale architecture
- CrossScaleFusion has very deep gradient paths:
  - 3 parallel Conformer pathways (3-6 layers each)
  - 3 separate cross-attention modules
  - Interpolation/pooling operations
  - Deep fusion layer

## Fixes Implemented

### 1. Reduced Learning Rate (4x reduction)
```python
lrStart: 0.002 → 0.0005  # Primary fix
lrEnd: 0.0001 → 0.00005
```

### 2. Longer Warmup (2.5x increase)
```python
warmup_steps: 4000 → 10000  # More gradual LR increase
```

### 3. Stronger Gradient Clipping (2x stronger)
```python
grad_clip_norm: 1.0 → 0.5  # Prevent gradient explosion
```

### 4. Architecture Stability
Added final LayerNorm after CrossScaleFusion:
```python
self.fusion = nn.Sequential(
    nn.LayerNorm(d_model * 3),
    nn.Linear(d_model * 3, d_model),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.LayerNorm(d_model)  # NEW: Final norm for stability
)
```

## Expected Behavior

With these fixes, the model should:
1. Learn smoothly through warmup (0-10000 batches)
2. Continue stable learning after warmup ends
3. Not diverge around batch 10000
4. Achieve target <0.1 CER by batch 150000

## Training Configuration

```python
Model: MultiScaleCTCDecoder
Parameters: 241M
Batch size: 64
LR: 0.0005 → 0.00005 (cosine decay)
Warmup: 10000 steps
Gradient clipping: 0.5
Total batches: 150000
```

## Verification Steps

Monitor these metrics during training:
1. **Loss should decrease monotonically** (no sudden jumps)
2. **CER should decrease smoothly** (0.4 → 0.3 → 0.2 → 0.1)
3. **Learning rate** should reach 0.0005 at batch 10000, then decay
4. **No divergence** around batch 10000-15000

If divergence still occurs:
- Further reduce lrStart to 0.0003
- Increase warmup to 15000
- Reduce model capacity (latent_dim: 1024 → 768)
