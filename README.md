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
<img width="1189" height="1489" alt="image" src="https://github.com/user-attachments/assets/9b6bf685-294e-46f4-b55f-5e68d911b2e8" />
<img width="1789" height="490" alt="image" src="https://github.com/user-attachments/assets/6492292f-f389-4537-b383-488d0d52af9c" />
<img width="646" height="528" alt="image" src="https://github.com/user-attachments/assets/bb0bb3fc-dc25-4843-ad63-a9e1651c8781" />
<img width="1189" height="1489" alt="image" src="https://github.com/user-attachments/assets/4fcd9cf4-394c-493b-916f-aedee7aa2bea" />
<img width="750" height="528" alt="image" src="https://github.com/user-attachments/assets/931afd1f-aa8b-43cf-ac0f-768e6ffc879e" />
<img width="1488" height="1489" alt="image" src="https://github.com/user-attachments/assets/24055b7e-7bab-4e46-ad56-03acec6aac7c" />






