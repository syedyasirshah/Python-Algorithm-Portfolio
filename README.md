# Python Algorithm Portfolio

This repository contains two Python code samples demonstrating algorithm implementation, problem-solving, and object-oriented programming.

## 1. Decision Tree Classifier

**File:** `decision_tree/decision_tree.py`

A Decision Tree Classifier implemented from scratch using Python and NumPy.

**Key Features:**
- Calculates entropy and information gain.
- Finds the best feature and threshold for splitting data.
- Builds a decision tree recursively.
- Supports configurable tree depth and stopping conditions.
- Predicts class labels for new data.

The classifier is demonstrated using the Breast Cancer Wisconsin dataset from scikit-learn.

## 2. PPE Compliance Analyzer

**File:** `ppe_compliance/ppe_safety_compliance.py`

A Python program that checks whether workers have the required Personal Protective Equipment (PPE) based on precomputed object detections.

**Key Features:**
- Filters detections using confidence scores.
- Associates PPE items with workers using bounding-box coordinates.
- Identifies detected and missing PPE items.
- Reports whether each worker is compliant.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

**Decision Tree Classifier:**

```bash
cd decision_tree
python train.py
```

**PPE Compliance Analyzer:**

From the repository root:

```bash
cd ppe_compliance
python ppe_safety_comliance.py
```

## Technologies Used

- Python
- NumPy
- scikit-learn (for the Decision Tree example)

## Repository Structure

```text
python-algorithm-portfolio/
├── README.md
├── requirements.txt
├── .gitignore
├── decision_tree/
│   ├── decision_tree.py
│   └── train.py
└── ppe_compliance/
    ├── ppe_safety_comliance.py
```
