# Wind Speed Prediction with Cross-Dataset Data Augmentation

## Overview

Wind energy has emerged as a cornerstone in global efforts to transition to renewable energy, driven by its low environmental impact and significant generation potential. However, the inherent intermittency of wind, influenced by complex and dynamic atmospheric patterns, poses significant challenges for accurate wind speed prediction. Existing approaches, including statistical methods, Machine Learning, and Deep Learning, often struggle with limitations such as non-linearity, non-stationarity, computational demands, and the requirement for extensive, high-quality datasets. In response to these challenges, we propose a novel Neighborhood Preserving Cross-Dataset Data Augmentation framework for high-horizon wind speed prediction. The proposed method addresses data variability and dynamic behaviors through three key components: (i) The Uniform Manifold Approximation and Projection (UMAP) is employed as a nonlinear dimensionality reduction technique to encode local relationships in wind speed time-series data while preserving neighborhood structures, (ii) a localized cross-dataset data augmentation (DA) approach is introduced using UMAP-reduced spaces to enhance data diversity and mitigate variability across datasets, and (iii) Recurrent Neural Networks (RNNs) are trained on the augmented datasets to model temporal dependencies and nonlinear patterns effectively. Our framework was evaluated using datasets from diverse geographical locations, including the Argonne Weather Observatory (USA), Chengdu Airport (China), and Beijing Capital International Airport (China). Comparative tests using regression-based measures on RNN, GRU, and LSTM architectures showed that the proposed method was better at improving the accuracy and generalizability of predictions, leading to an average reduction in prediction error. Consequently, our study highlights the potential of integrating advanced dimensionality reduction, data augmentation, and Deep Learning techniques to address critical challenges in renewable energy forecasting.

## Key Contributions

- Introduces a UMAP-based cross-dataset data augmentation approach to enhance training diversity.
- Evaluates SRNN, GRU, and LSTM models on wind speed datasets from Argonne (USA), Beijing (China), and Chengdu (China).
- Demonstrates that data augmentation significantly improves predictive accuracy, particularly in high-variability datasets like Chengdu.
- Validates performance using MAPE, MAE, R², and Friedman rankings, confirming the benefits of augmentation.

## Repository Contents

- data/ - Processed wind speed datasets.
- notebooks/ - Jupyter notebooks for data preprocessing, training, and evaluation.
- results/ - Performance metrics, visualizations, and comparison tables.

## Citation

If you find this work useful, please cite:
Leon-Gomez, E. A., Álvarez-Meza, A., & Castellanos-Dominguez, G. (2025). Cross-Dataset Data Augmentation using UMAP for Deep Learning-based Wind Speed Prediction. Computers, 2025. [DOI Placeholder]

