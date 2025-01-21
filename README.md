# Beyond Single-view Analysis: An Interaction-Aware Multi-Period Encoder Network for Multi-Person Motion Prediction

## Abstract
Recent advancements in multi-person motion prediction have significantly improved the modeling of interactions between individuals in multimedia environments by modeling interactions between individuals. Existing methods, predominantly relying on Transformer-based architectures, struggle with extracting robust temporal dependencies. The attention mechanisms in Transformers, while effective in modeling pairwise relationships, often dilute critical temporal signals when faced with sparse or irregular time points, where significant motion events might be scattered or intermittent. They failed to capture the complex temporal variations inherent in dynamic environments, where multiple patterns of movement overlap and evolve. To tackle these issues, we introduce an Interaction-Aware Multi-Period Encoder Network that rethinks multi-person motion prediction from the perspective of temporal variation. Specifically, the Intra-Interaction Multi-Period Encoder is developed to disentangle and model the multi-periodic features implicit in different individuals, which aims to reveal hidden temporal dynamics and provide a more accurate prediction of future human actions in multi-person settings. Additionally, we’ve developed an Inter-interaction Multi-Time Scale Encoder to delve into the interaction dynamics across different timescales, aiming for a more comprehensive understanding of interpersonal movements. Empirical results on challenging benchmarks demonstrate that our approach significantly outperforms existing methods.Our code will be uploaded gradually in the future.

## Overview


## Prepare Data



## Requirements
- python==3.6
- pytorch==1.10.0
- dct==0.1.6
- numpy==1.19.2


## Training
`python train.py`



