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
* `requirements.txt`: List of required Python packages.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/[repository-name].git

2. **Create a virtual environment (recommended):**

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the datasets:** [Provide instructions on how to download or access the datasets.]

5. **Run the experiments:**

## Results

The results of the experiments are stored in the `results/` directory.  [Describe the structure of the results directory and the files generated.]  Key findings are summarized in the paper.

```

