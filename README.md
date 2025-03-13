# Wind Speed Prediction with Cross-Dataset Data Augmentation

## Overview

This repository contains the code and data for the paper "Cross-Dataset Data Augmentation using UMAP for Deep Learning-based Wind Speed Prediction." The study explores a novel Neighborhood-Preserving Cross-Dataset Data Augmentation (UMAP-CDDA) framework to improve wind speed forecasting using deep learning models.

## Key Contributions

- Introduces a UMAP-based cross-dataset data augmentation approach to enhance training diversity.
- Evaluates SRNN, GRU, and LSTM models on wind speed datasets from Argonne (USA), Beijing (China), and Chengdu (China).
- Demonstrates that data augmentation significantly improves predictive accuracy, particularly in high-variability datasets like Chengdu.
- Validates performance using MAPE, MAE, R², and Friedman rankings, confirming the benefits of augmentation.

## Repository Contents

- data/ - Processed wind speed datasets.
- models/ - Implementations of SRNN, GRU, and LSTM architectures.
- notebooks/ - Jupyter notebooks for data preprocessing, training, and evaluation.
- results/ - Performance metrics, visualizations, and comparison tables.

## Citation

If you find this work useful, please cite:
Leon-Gomez, E. A., Álvarez-Meza, A., & Castellanos-Dominguez, G. (2025). Cross-Dataset Data Augmentation using UMAP for Deep Learning-based Wind Speed Prediction. Computers, 2025. [DOI Placeholder]

## License

This project is licensed under the Creative Commons Attribution (CC BY 4.0) license.