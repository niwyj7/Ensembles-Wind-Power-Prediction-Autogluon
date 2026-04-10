# Ensembles-Wind-Power-Autogluon
This project provides a systematic pipeline for high-frequency wind power forecasting, built on the **AutoGluon** framework to enable rigorous model benchmarking and feature engineering.

### **Methodological Rationale**
While recent literature, such as Cambridge’s **WindDragon** project, has demonstrated the potential of automated deep learning architectures for wind energy, this pipeline prioritizes **tree-based models** (e.g., LightGBM, CatBoost) for several strategic reasons:

* **Signal-to-Noise Ratio (SNR) & Granularity:** In wind power prediction, the inherent noise in meteorological data often limits the effectiveness of high-complexity deep models. Tree-based ensembles offer superior robustness and better capture the non-linear relationships at the specific spatial granularities used here.
* **Computational Efficiency:** Prioritizing rapid inference and lower computational overhead is critical for real-time grid balancing. Tree-based architectures provide a significantly faster training-to-prediction cycle compared to deep CNNs or Transformers.
* **Data Scale Suitability:** Given the selective use of spatial location points, the dataset density does not necessitate the heavy parameterization of deep learning, making "shallow" but high-performance ensembles more effective for preventing overfitting.

### **Key Features**
* **Systematic Feature Ablation:** Evaluates the impact of regional aggregates, cyclical temporal encodings, and lagged weather features.
* **Automated Model Selection:** Leverages AutoGluon to optimize hyperparameters across GBM, CatBoost, and Torch-based neural networks.
* **Production-Ready:** Designed for 15-minute interval forecasting with a focus on interpretability and deployment speed.
