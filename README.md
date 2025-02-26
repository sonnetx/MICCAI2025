# BiasICL

# In-Context Learning and Demographic Biases of Vision Language Models

This repository contains code and data for the paper "In-Context Learning and Demographic Biases of Vision Language Models" (Add paper link here when available).  This work investigates the impact of demographic composition of demonstration prompts on the performance of Vision-Language Models (VLMs) in medical diagnosis using in-context learning (ICL).

## Abstract

Vision language models (VLMs) show promise in medical diagnosis, but their performance across demographic subgroups when using in-context learning (ICL) remains poorly understood. We examine how the demographic composition of demonstration prompts affects VLM performance in two medical imaging tasks: skin lesion malignancy prediction and pneumothorax detection from chest radiographs. Our analysis reveals that ICL influences model predictions through multiple mechanisms: (1) ICL allows VLMs to learn subgroup-specific disease base rates from prompts and (2) ICL leads VLMs to make predictions that perform differently across demographic groups, even after controlling for subgroup-specific disease base rates. Our empirical results inform best-practices for prompting current VLMs, while also suggesting next steps for improving our theoretical understanding of these models.

## Project Overview

This project explores the following key questions:

* How does the demographic makeup of the examples provided in the ICL prompt affect the VLM's diagnostic accuracy for different demographic groups?
* Does ICL simply allow the VLM to learn and apply subgroup-specific base rates of disease?
* Are there other, more complex interactions between the prompt demographics and the VLM's predictions?

We investigate these questions using two medical imaging datasets:

* **Skin Lesion Malignancy Prediction:** DDI (https://ddi-dataset.github.io/)
* **Pneumothorax Detection from Chest Radiographs:** CheXpert (https://stanfordmlgroup.github.io/competitions/chexpert/)


## Code

The code for this project is organized as follows:

* `data/`: Contains the datasets used in the experiments (or scripts to download them).  [Provide more details about data format and preprocessing.]
* `models/`: Contains the code for loading and using the VLMs. [Specify the VLMs used, e.g., CLIP, BioViL, etc.]
* `scripts/`: Contains the scripts for running the experiments, including:
    * `run_experiments.py`: Main script for running the experiments with different prompt configurations. [Explain how to run the script and the available command-line arguments.]
    * `evaluate.py`: Script for evaluating the model performance and calculating fairness metrics. [Specify the metrics used, e.g., AUC, sensitivity, specificity, demographic parity, etc.]
    * `create_prompts.py`: Script for generating different ICL prompts with varying demographic compositions. [Explain the prompt generation strategy.]
* `notebooks/`: Contains Jupyter notebooks for analysis and visualization of the results. [Optional: Include notebooks for exploring the data and results.]
* `requirements.txt`: List of required Python packages.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/[repository-name].git

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the datasets:** [Provide instructions on how to download or access the datasets.]

5. **Run the experiments:**
   ```bash
   python scripts/run_experiments.py [arguments]
   ```

## Results

The results of the experiments are stored in the `results/` directory.  [Describe the structure of the results directory and the files generated.]  Key findings are summarized in the paper.

```

