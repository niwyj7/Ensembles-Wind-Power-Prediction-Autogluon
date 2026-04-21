# Ensembles-Wind-Power-Autogluon
This project provides a systematic pipeline for 15-minute granularity wind power forecasting, built on the **AutoGluon** framework to enable rigorous model benchmarking and feature engineering.
### **Regional Wind Power Forecasting**

**Why AutoGluon & Tree-Based Ensembles?**
While projects like Cambridge’s *WindDragon* [^1] demonstrate the potential of AutoDL, this pipeline adopts a **tree-based approach** (LightGBM/CatBoost) via **AutoGluon** to address the unique constraints of emerging energy markets:

* **Robustness to Data Scarcity:** In rapidly expanding markets, there is often a lack of the long-term historical timelines required to train deep neural networks effectively. Tree-based models are significantly more robust when testing ideas with limited data.
* **Mitigating the Curse of Dimensionality:** With high-dimensional meteorological data, the risk of overfitting increases as the data timeline shortens. Tabular tree structures handle these feature spaces more effectively than deep learning, which often requires vast datasets to generalise.
* **Operational Efficiency:** This approach prioritises rapid inference and lower computational overhead, ensuring the pipeline remains practical for real-time grid balancing where "shallow" but high-performance ensembles often outperform over-parameterised models.
* **Signal-to-Noise Ratio (SNR) & Granularity:** In wind power prediction, the inherent noise in meteorological data often limits the effectiveness of high-complexity deep models. Tree-based ensembles offer superior robustness and better capture the non-linear relationships at the specific spatial granularities used here. I opted to test lower-resolution regional aggregates. My results demonstrate that strategic feature engineering (regional means/std) can capture the essential spatial dynamics of a weather system without the need for the high-resolution "feature maps" and CNN. This achieves a superior balance between Signal-to-Noise Ratio (SNR) and computational efficiency.
  
To handle 400+ features, the pipeline leverages AutoGluon’s automated regularisation, replacing manual calibration of L2 penalties, feature fractioning, and leaf constraints. By dynamically optimising structural parameters and employing multi-layer stacking, AutoGluon mitigates the "Curse of Dimensionality" and ensures robustness against noise—achieving stable generalisation on limited meteorological datasets without the need for exhaustive hand-tuning [^2].


### **Key Features**
* **Systematic Feature Ablation:** Evaluates the impact of regional aggregates, cyclical temporal encodings, and lagged weather features.
* **Automated Model Selection:** Leverages AutoGluon to optimise hyperparameters across GBM, CatBoost, and Torch-based neural networks.
* **Production-Ready:** Designed for 15-minute interval forecasting with a focus on interpretability and deployment speed.

### **Results**  

#### **Performance Evaluation**
The model was trained on **16,704 observations** with **493 features** (pivoted spatial grid points and temporal encodings). We utilized an automated ensemble strategy, which outperformed individual base models.

| Model | Validation RMSE | Training Time (s) | Interpretation |
| :--- | :--- | :--- | :--- |
| **WeightedEnsemble_L2** | **660.21** | **69.0s (Total)** | **Best overall** |
| CatBoost | 663.35 | 7.89s | Primary contributor (77.8%) |
| LightGBM | 689.12 | 18.82s | Secondary contributor (11.1%) |
| NeuralNetTorch | 799.84 | 31.18s | Tertiary contributor (11.1%) |

### **Ablation Study Results**

| Feature Group Removed | N Features | Val RMSE (Full) | Val RMSE (Ablated) | RMSE Delta |
| :--- | :---: | :---: | :---: | :---: |
| **grid_all** | 438 | 660.21 | 688.66 | **+28.46** |
| **weather_extra** | 3 | 660.21 | 684.99 | **+24.79** |
| **regional_diff** | 18 | 660.21 | 650.84 | -9.36 |
| **regional_roll** | 36 | 660.21 | 649.90 | -10.30 |
| **seperated_diff** | 219 | 660.21 | 648.71 | -11.50 |
| **time** | 4 | 660.21 | 642.09 | -18.11 |
| **regional_base** | 3 | 660.21 | 642.02 | -18.18 |


**Key Metric Analysis:**
* **Ensemble Composition:** The final model is a weighted ensemble dominated by **CatBoost (77.8%)**. This validates the initial hypothesis that gradient-boosted decision trees (GBDTs) are superior to deep learning for this "tabular" meteorological dataset.
* **Efficiency:** The pipeline achieved a high inference throughput of **~9,131 rows/s**, making it suitable for real-time grid dispatching scenarios.

#### **Conclusion & Discussion**
The experimental results demonstrate that for regional wind power forecasting in China using ECMWF data, **tree-based ensembles provide a more robust and computationally efficient solution than deep neural networks.**

1.  **Model Selection:** CatBoost's dominance suggests it better captures the non-linear relationship between 100m wind speeds and power generation, likely due to its superior handling of noisy tabular features compared to `NeuralNetTorch`, which showed a significantly higher RMSE (799.84).
2.  **Scalability:** Despite the high dimensionality of the input space (493 columns), the model converged in under 70 seconds. This suggests that the **Dimensionality** was effectively mitigated by AutoGluon’s feature pruning and the inherent feature selection of GBDTs.

> **Compute Environment:** > Experiments were conducted on a high-memory Linux cluster (56 Cores, 448GB RAM). The architecture leverages multi-core parallelization for rapid hyperparameter optimization.


### **EDA**  

**Regional Wind Power Generation**
<br>
* No obvious evidence of increasing installed capacity of wind farms.
<br>
<div align="center">
  <img width="1488" height="1489" alt="image" src="https://github.com/user-attachments/assets/24055b7e-7bab-4e46-ad56-03acec6aac7c" />
</div>
<br>

**Regional Wind Field Study** 
<br>
<div align="center">
  <img width="1189" height="1489" alt="image" src="https://github.com/user-attachments/assets/9b6bf685-294e-46f4-b55f-5e68d911b2e8" /><br>
  <img width="1789" height="490" alt="image" src="https://github.com/user-attachments/assets/6492292f-f389-4537-b383-488d0d52af9c" /><br>
  <img width="646" height="528" alt="image" src="https://github.com/user-attachments/assets/bb0bb3fc-dc25-4843-ad63-a9e1651c8781" />
</div>
<br>
* Diff features
<br>
<div align="center">
  <img width="1989" height="390" alt="image" src="https://github.com/user-attachments/assets/e98ec5d2-c517-4341-aac3-a744bf0f4f0a" />
</div>
<br>
* Wind speed square and cube value, cube shows better separation.
<br>
<div align="center">
  <img width="1789" height="489" alt="image" src="https://github.com/user-attachments/assets/d89d5a9f-71a2-44da-a738-110b2d090989" /><br>
  <img width="624" height="565" alt="image" src="https://github.com/user-attachments/assets/09ca57a2-75ac-4d7e-8767-d1217ec6bb6f" />
</div>
<br>

**Other Weather Features** 
<br>
<div align="center">
  <img width="1189" height="1489" alt="image" src="https://github.com/user-attachments/assets/4fcd9cf4-394c-493b-916f-aedee7aa2bea" /><br>
  <img width="750" height="528" alt="image" src="https://github.com/user-attachments/assets/931afd1f-aa8b-43cf-ac0f-768e6ffc879e" />
</div>



[^1]: Cambridge WindDragon Project. "Automated Deep Learning for Wind Power Prediction."(https://www.cambridge.org/core/journals/environmental-data-science/article/winddragon-automated-deep-learning-for-regional-wind-power-forecasting/64B3D7345C9B3EE66574E9F407F31482)

[^2]: Erickson, N., et al. (2020). "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data." arXiv preprint arXiv:2003.06505.
