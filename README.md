# Ensembles-Wind-Power-Autogluon
This project provides a systematic pipeline for 15-minute granularity wind power forecasting, built on the **AutoGluon** framework to enable rigorous model benchmarking and feature engineering.
### **Regional Wind Power Forecasting**

**Why AutoGluon & Tree-Based Ensembles?**
While projects like Cambridge’s *WindDragon* demonstrate the potential of AutoDL, this pipeline adopts a **tree-based approach** (LightGBM/CatBoost) via **AutoGluon** to address the unique constraints of emerging energy markets:

* **Robustness to Data Scarcity:** In rapidly expanding markets, there is often a lack of the long-term historical timelines required to train deep neural networks effectively. Tree-based models are significantly more robust when testing ideas with limited data.
* **Mitigating the Curse of Dimensionality:** With high-dimensional meteorological data, the risk of overfitting increases as the data timeline shortens. Tabular tree structures handle these feature spaces more effectively than deep learning, which often requires vast datasets to generalise.
* **Operational Efficiency:** This approach prioritises rapid inference and lower computational overhead, ensuring the pipeline remains practical for real-time grid balancing where "shallow" but high-performance ensembles often outperform over-parameterised models.
* **Signal-to-Noise Ratio (SNR) & Granularity:** In wind power prediction, the inherent noise in meteorological data often limits the effectiveness of high-complexity deep models. Tree-based ensembles offer superior robustness and better capture the non-linear relationships at the specific spatial granularities used here.
  

### **Key Features**
* **Systematic Feature Ablation:** Evaluates the impact of regional aggregates, cyclical temporal encodings, and lagged weather features.
* **Automated Model Selection:** Leverages AutoGluon to optimise hyperparameters across GBM, CatBoost, and Torch-based neural networks.
* **Production-Ready:** Designed for 15-minute interval forecasting with a focus on interpretability and deployment speed.
  

### **EDA**  
* **Regional Wind Field Study** 
<img width="1189" height="1489" alt="image" src="https://github.com/user-attachments/assets/9b6bf685-294e-46f4-b55f-5e68d911b2e8" />
<img width="1789" height="490" alt="image" src="https://github.com/user-attachments/assets/6492292f-f389-4537-b383-488d0d52af9c" />
<img width="646" height="528" alt="image" src="https://github.com/user-attachments/assets/bb0bb3fc-dc25-4843-ad63-a9e1651c8781" />

* Diff features.
<img width="1989" height="390" alt="image" src="https://github.com/user-attachments/assets/e98ec5d2-c517-4341-aac3-a744bf0f4f0a" />

* Wind speed square and cube value, cube shows better separation.
<img width="1789" height="489" alt="image" src="https://github.com/user-attachments/assets/d89d5a9f-71a2-44da-a738-110b2d090989" />
<img width="624" height="565" alt="image" src="https://github.com/user-attachments/assets/09ca57a2-75ac-4d7e-8767-d1217ec6bb6f" />

* **Other Weather Features** 
<img width="1189" height="1489" alt="image" src="https://github.com/user-attachments/assets/4fcd9cf4-394c-493b-916f-aedee7aa2bea" />

* **Regional Wind Power Generation**
* No obvious evidence of increasing installed capacity of wind farms.
<img width="750" height="528" alt="image" src="https://github.com/user-attachments/assets/931afd1f-aa8b-43cf-ac0f-768e6ffc879e" />
<img width="1488" height="1489" alt="image" src="https://github.com/user-attachments/assets/24055b7e-7bab-4e46-ad56-03acec6aac7c" />






