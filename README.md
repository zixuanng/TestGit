# TestGit

TestGit is a machine learning project focused on building, evaluating, and visualizing classification models using scikit-learn and pandas. The repository demonstrates model training pipelines, hyperparameter tuning, and performance comparison across different algorithms.

## Features

- **Model Training Pipeline:** Integrates preprocessing with a decision tree classifier using scikit-learn's `Pipeline`.
- **Hyperparameter Tuning:** Uses `GridSearchCV` to optimize decision tree parameters.
- **Comprehensive Metrics:** Calculates and reports accuracy, precision, recall, and F1-score for train and test sets.
- **Feature Importance:** Extracts and displays the most influential features of the best-performing model.
- **Visualization:** Compares model performance (Logistic Regression, Decision Tree, Random Forest) with bar charts and prints summary tables using matplotlib and seaborn.

## Main Files

- **day-4&5.py:** Implements the training pipeline, grid search, model reporting, and feature importance extraction.
- **visualization.py:** Visualizes performance metrics and prints a comparison table for multiple classification models.

## Example Workflow

1. **Pipeline Construction:**
   - Preprocessing and model steps are combined for streamlined training and evaluation.
2. **Grid Search:**
   - DecisionTreeClassifier parameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) are tuned.
3. **Evaluation:**
   - Reports best parameters, scores, and generates detailed performance metrics.
4. **Feature Analysis:**
   - Displays top features based on model importance.
5. **Visualization:**
   - Outputs comparison charts and tables for multiple models.

## Dependencies

- Python (>=3.7)
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

Install dependencies with:

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## Usage

Run the pipeline and visualization scripts to train models and visualize their performance:

```bash
python day-4&5.py
python visualization.py
```

## Output

- **Model Report:** Best parameters and metrics for decision tree classifier.
- **Feature Importance:** Top 5 influential features.
- **Comparison Chart:** Bar charts and tables comparing accuracy, precision, recall, and F1-score for different models.

## License

This repository is open source for educational and demonstration purposes.
