# Chicken Counting with Pyramid Vision Transformer

Automatic chicken counting in poultry farms using deep learning. Based on the paper:

> **"Transforming Poultry Farming: A Pyramid Vision Transformer Approach for Accurate Chicken Counting"**  
> _Sensors 2024, 24, 2977_  
> DOI: [10.3390/s24092977](https://doi.org/10.3390/s24092977)

## About the Project

This project implements a complete inference system for counting chickens in poultry farm images using the architecture proposed in the paper:

-   **Backbone**: Pyramid Vision Transformer v2 B2 (PVT-v2-B2)
-   **PFA**: Pyramid Feature Aggregation to combine multi-scale features
-   **MDC**: Multi-Scale Dilated Convolution Head to generate density maps
-   **Output**: 2D density map where the sum = estimated number of chickens

## Model Architecture

```
Input Image (3×256×256)
         ↓
┌────────────────────────┐
│   PVT-v2-B2 Backbone   │
│  (Pyramid Features)     │
└────────────────────────┘
         ↓
    [f1, f2, f3, f4]
         ↓
┌────────────────────────┐
│  Pyramid Feature       │
│  Aggregation (PFA)     │
│  - Lateral connections │
│  - Top-down upsampling │
└────────────────────────┘
         ↓
   Aggregated Features
         ↓
┌────────────────────────┐
│  Multi-Scale Dilated   │
│  Conv Head (MDC)       │
│  - Dilation: 1, 2, 3   │
│  - Fusion + Regression │
└────────────────────────┘
         ↓
  Density Map (1×256×256)
         ↓
    Σ = Count
```

## Quick Start

TODO
