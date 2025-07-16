
# Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity

This project contains the implementation of the **FedHEAL** algorithm along with standard **FedAvg** for federated learning tasks on heterogeneous datasets. The implementation supports both the original version and a **novel variation** introduced in the extended version.

---

## Directory Structure and File Descriptions

| File/Folder Name                      | Description |
|--------------------------------------|-------------|
| `FedHEAL-extension`     | Includes novel contributions added to FedHEAL. |
| `fedheal-paper-implementation.ipynb` | Jupyter notebook showcasing results of original FedHEAL paper. |
| `fed-heal-with-novelty.ipynb`   | Jupyter notebook demonstrating outcomes using the novel approach. |

---

## Included Datasets

The project makes use of the following datasets:

- **Digits Dataset**: Includes MNIST, USPS, SVHN, and SYN numbers.
- **Office-Caltech Dataset**: Includes Caltech, Amazon, Webcam, and DSLR domains.

**download datasets separately and place them in the data0 directory**

---

## How to Run

You can run the experiments using either of the `.ipynb` notebooks or from the command line after extracting the respective zip folder.

### Reproducing Results in Notebook

1. Upload and extract the ZIP archive in your environment:
```python
import zipfile
zip_path = "FedHEAL-main.zip"  # or "FedHEAL-extension"
extract_path = "./"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

2. Set working directory:
```python
import os
os.chdir("FedHEAL-main")  # or "FedHEAL-extension"
```

3. Run any of the following commands within the notebook or terminal:

---

## Commands to Run Algorithms

#### **1. Standard FedAvg on Digits Dataset**
```bash
python main.py --device_id 0 --model 'fedavg' --dataset 'fl_digits' --mnist 5 --usps 5 --svhn 5 --syn 5 --communication_epoch 40 --local_epoch 10 --parti_num 20 --seed 42
```

#### **2. FedHEAL (Paper) on Digits Dataset**
```bash
python main.py --device_id 0 --model 'fedavgheal' --dataset 'fl_digits' --mnist 5 --usps 5 --svhn 5 --syn 5 --communication_epoch 100 --local_epoch 10 --parti_num 20 --seed 42
```

#### **3. Standard FedAvg on Office-Caltech Dataset**
```bash
python main.py --device_id 0 --model 'fedavg' --dataset 'fl_officecaltech' --caltech 5 --amazon 5 --webcam 5 --dslr 5 --communication_epoch 40 --local_epoch 10 --parti_num 20 --seed 42
```

#### **4. FedHEAL (Paper) on Office-Caltech Dataset**
```bash
python main.py --device_id 0 --model 'fedavgheal' --dataset 'fl_officecaltech' --caltech 5 --amazon 5 --webcam 5 --dslr 5 --communication_epoch 100 --local_epoch 10 --parti_num 20 --seed 42
```

---

## Arguments Explained

- `--device_id` : Index of the GPU or CPU to use.
- `--model` : Type of model to use (`fedavg` or `fedavgheal`).
- `--dataset` : Dataset group used for training.
- `--mnist`, `--usps`, etc. : Number of clients per dataset.
- `--communication_epoch` : Number of communication rounds.
- `--local_epoch` : Number of local training epochs per client.
- `--parti_num` : Total number of participating clients.
- `--seed` : Random seed for reproducibility.

---

## Dependencies

Install commonly used packages manually:
```bash
pip install torch torchvision numpy matplotlib gdown
```

---

## Novelty: UCB-Based Dynamic Client Selection Strategy

This project extends the original **FedHEAL** framework by integrating a novel **UCB-based client selection strategy** that dynamically chooses clients per communication round based on their contribution and utility.

### Key Features:
- **Dynamic K selection**: Adjusts the number of participating clients (`K`) each round based on the observed gain in model accuracy.
- **UCB Scoring**: Selects clients using a bandit-based exploration-exploitation approach, giving preference to high-reward clients while occasionally exploring new ones.
- **Efficient Training**: Reduces communication overhead and computational cost, enabling faster convergence.
- **Preserves Fairness**: Works in conjunction with FedHEAL’s fairness mechanisms (FPHL and FAEL) to maintain balanced performance across domains.

### Results Summary:

| Dataset         | Method         | Avg Accuracy ↑ | Std. Dev ↓ |
|----------------|----------------|----------------|------------|
| **Digits**      | FedAvg         | 71.08%         | 29.21      |
|                | FedHEAL        | 75.31%         | 24.07      |
|                | **FedHEAL+UCB**| **79.71%**     | **17.62**  |
| **Office-Caltech** | FedAvg      | 50.20%         | 15.64      |
|                | FedHEAL        | 54.86%         | 6.87       |
|                | **FedHEAL+UCB**| **61.78%**     | **7.57**   |

### Impact:
- **+4.4%** improvement on Digits dataset over baseline FedHEAL.
- **+6.9%** improvement on Office-Caltech dataset.
- **Lower STD** across domains indicates better fairness.
- Converges faster with fewer communication rounds.

### Graphical representation of the convergence across communication rounds

On Digits Dataset

![Digits_result](/Novelty_digits.png)

On Office Caltech Dataset

![Office_caltech_result](/Novelty_office_caltech.png)

---

## Disclaimer

This project is an **independent academic extension** of the original paper titled *"Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity"* (CVPR 2024). All credit for the foundational FedHEAL framework goes to the original authors. This work builds upon their contribution to explore novel strategies for improving training efficiency and fairness.
