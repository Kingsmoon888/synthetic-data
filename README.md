This repository provides the battery datasets, synthetic data, and framework for battery lifetime prediction research.  
The data and codes corresponding to two battery chemistries are included in two separated folders under the folder `main`:

- LFP (Lithium Iron Phosphate)
- NCM (Nickel Cobalt Manganese)

For each chemistry, the repository contains:
- `data_lebeled.csv` - Original label data
- `index.csv` - Dataset split information
- `syn_data.zip` - Generated datasets using multiple generation strategies
- `main.py` - main implementation code

The main function of the proposed method is included in the file main.py. For each chemistry, given the labeled real data and unlabeled synthetic data as well as the test data, it triggers the training of prediction model, and returns the test error measured using MAPE.
  
The source dataset comes from:  

Severson, K.A., Attia, P.M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M.H., Aykol, M., Herring, P.K., Fraggedakis, D., et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nat. Energy 4, 383–391  

Weng A, Mohtat P, Attia P M, et al. Predicting the impact of formation protocols on battery lifetime immediately after manufacturing[J]. Joule, 2021, 5(11): 2971-2992.
