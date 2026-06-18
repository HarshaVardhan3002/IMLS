# ML Safety Presentation Hub

Welcome to the presentation hub for the **Introduction to ML Safety** CARLA exercises.
This document serves as a quick jump-link reference to instantly access the implementations of all coded exercises during the live presentation.

## Core Implementations

Below are the direct links to the Python implementations for each practical exercise. Clicking these links will open the files directly in the editor.

- **Exercise 3 (Fundamentals)**: [ex3_fundamentals.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/exercises/ex3_fundamentals.py)
  - *Topic*: Multi-model evaluation, precision/recall/F1, and confusion matrices.
- **Exercise 4 (Model Testing)**: [ex4_testing.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/exercises/ex4_testing.py)
  - *Topic*: K-projection coverage of the Operational Design Domain (ODD).
- **Exercise 5 (Calibration & Backdoor)**: [ex5_calibration_backdoor.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/exercises/ex5_calibration_backdoor.py)
  - *Topic*: Temperature scaling and targeted data poisoning (backdoors).
- **Exercise 6 (Explainability)**: [ex6_explainability.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/exercises/ex6_explainability.py)
  - *Topic*: Visualizing model attention and feature importance (Occlusion/CAM).
- **Exercise 8 (Adversarial ML)**: [ex8_adversarial_ml.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/exercises/ex8_adversarial_ml.py)
  - *Topic*: Fast Gradient Sign Method (FGSM) attacks and empirical robustness.
- **Exercise 9 (Anomaly Detection)**: [ex7_anomaly_detection.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/exercises/ex7_anomaly_detection.py)
  - *Topic*: Maximum Softmax Probability (MSP) and feature-based k-NN detection. *(Note: The script is named `ex7` internally but covers Sheet 9)*.

## Core Infrastructure

Should the lecturer ask about the underlying dataset or model configurations:

- **Dataset Loader**: [dataset.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/src/dataset.py)
- **Model Architecture**: [model.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/src/model.py)
- **Configuration & Paths**: [config.py](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/config.py)

## Solutions & Analysis

The theoretical answers and detailed empirical analysis for all sheets are centrally located in the README:

- **Full Solutions**: [README.md](file:///f:/old_desktop/Desktop%20(1)/Lecturers%20SoSe%202026/Introduction%20to%20ML%20Safety/repo_staging/README.md)
