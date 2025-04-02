
## LargeMvC-Net
LargeMvC-Net is a framework for large-scale multi-view clustering. This repository contains the code and datasets used for the experiments in our paper.

### Installation
1. **Clone the repository:**
- git clone https://anonymous.4open.science/r/LargeMvC-Net-6F01
2. **Set up the environment:**
- Use Conda to create an environment with the required dependencies:
- conda create -n largemvc python=3.8.19
- conda activate largemvc 
- pip install -r requirements.txt

### Datasets Preparation
- For all datasets, please obtain them from the following links: <https://drive.google.com/drive/folders/1DcaeDbFz6eNYpy1ohliaqCBJ2gZtG3PE?usp=sharing>;
- Download datasets from the provided links.
- Place the datasets in the `/data` directory:
 /data/
  └── Dataset1
  └── Dataset2
  ...

### Usage Instructions
1. **Training and Testing the Model:** To train the model on a specific dataset, run the following command:
- python run.py.
2. **Modifying Configurations:**
- Edit `run.py` to modify settings such as the dataset, network layers or learning rate.

### Results Reproduction
For example, **Flickr** dataset:
- python run.py.

### Notes
 - Ensure all dependencies are installed, as listed in the requirements.txt.   
 - The code is designed to run on **GPU**.   
 - For custom datasets, modify the dataset loader in /util/utils.py.

### Contact
For further questions or clarifications, please raise an issue in this repository.

