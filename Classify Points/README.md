# Project Title

Practicing Project: Classify Points as "Class A" or "Class B" based on PyTorch

## Description

Given three features:
- **Feature 1 (x)**: A integer value between 0 and 100
- **Feature 2 (y)**: A integer value between 0 and 100
- **Feature 3 (z)**: A integer value between 0 and 50

The goal is to classify the data points into two classes:
- **Class A**
- **Class B**

### Labeling Criteria:
- If **Feature 3 (z)** > 30, the label is **Class B**.
- If **Feature 3 (z)** <= 30 and **Feature 1 (x)** > 30, the label is **Class B**.
- Otherwise, the label is **Class A**.
