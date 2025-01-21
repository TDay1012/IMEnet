# Beyond Single-view Analysis: An Interaction-Aware Multi-Period Encoder Network for Multi-Person Motion Prediction

## Abstract
Recent advancements in multi-person motion prediction have significantly improved the modeling of interactions between individuals in multimedia environments by modeling interactions between individuals. Existing methods, predominantly relying on Transformer-based architectures, struggle with extracting robust temporal dependencies. The attention mechanisms in Transformers, while effective in modeling pairwise relationships, often dilute critical temporal signals when faced with sparse or irregular time points, where significant motion events might be scattered or intermittent. They failed to capture the complex temporal variations inherent in dynamic environments, where multiple patterns of movement overlap and evolve. To tackle these issues, we introduce an Interaction-Aware Multi-Period Encoder Network that rethinks multi-person motion prediction from the perspective of temporal variation. Specifically, the Intra-Interaction Multi-Period Encoder is developed to disentangle and model the multi-periodic features implicit in different individuals, which aims to reveal hidden temporal dynamics and provide a more accurate prediction of future human actions in multi-person settings. Additionally, we’ve developed an Inter-interaction Multi-Time Scale Encoder to delve into the interaction dynamics across different timescales, aiming for a more comprehensive understanding of interpersonal movements. Empirical results on challenging benchmarks demonstrate that our approach significantly outperforms existing methods.Our code will be uploaded gradually in the future.

## Overview


## Prepare Data
[Go to Google](https://www.google.com)
[The dataset can be downloaded from]((https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBa0gyTDdKNHlqNlhnU2RkandwM1F3UGUwaUZCP2U9UllKS1pt&id=973ECA78B22FF641%21167&cid=973ECA78B22FF641))
 
```
your_project_folder/
├── Data/
│   ├── mix
│   │   ├── mix1_6persons.npy
│   │   ├── mix2_6persons.npy
│   ├── mocap
│   │   ├── test_3_75_mocap.npy
│   │   ├── train_3_75_mocap.npy
│   ├── mupots3d
│   │   ├── mupots_2_150_mocap.npy
│   │   ├── mupots_2_150_mocap.npy
│   ├── ...
├── Model
│   ├── Conv_Block.py
│   ├── GCN.py
│   ├── Layers.py
│   ├── Models.py
│   ├── Modules.py
│   ├── SubLayers.py
│   ├── MPE.py
│   ├── ...
├── data_short
├── metrics.py
├── Train_IME.py
├── Test_IME.py
```


## Requirements
- python==3.6
- pytorch==1.10.0
- dct==0.1.6
- numpy==1.19.2


## Training
`python Train_IME.py`

## Evaluation
`python Test_IME.py`



