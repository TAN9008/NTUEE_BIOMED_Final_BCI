# Task.1 Classifier
Implement a classifier that separates EEG signals into "relax" and "focus" states.
## Perparation
Before execution, install the following packages: 
- Classifier_EEGNet.py
```bash
pip install numpy matplotlib seaborn scikit-learn scipy
```
- Classifier_SVM.py
```bash
pip install numpy matplotlib seaborn scipy pywavelets torch
# torch needs to be precheck the CUDA version
```
## Compliation
```bash
python file_name.py
```
## Reminder
Directory name is preset to be "bci_dataset_113-2". Each dataset is under directory named "S#", and two text files are in the dataset, 1.txt being relax state and 2.txt being concentration state.
