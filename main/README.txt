For both LFP and NCM, the main function of the proposed method is included in the file main.py. For each chemistry, given the labeled real data and unlabeled synthetic data as well as the test data, it triggers the training of prediction model, and returns the test error measured using MAPE.

Requirements:
python==3.9
numpy==1.23.3
pandas==1.4.4
scikit-learn==1.1.2
scipy==1.9.1
