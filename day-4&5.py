from day_3 import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create a pipeline with preprocessing and decision tree
tree_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Parameter grid for the decision tree classifier (note the 'classifier__' prefix)
param_grid = {
    'classifier__max_depth': [2, 3, 4, 5, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV with the pipeline
grid = GridSearchCV(tree_pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# Optional: Get the best estimator and make predictions
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Test accuracy with best model:", best_model.score(X_test, y_test))

# Generate comprehensive model report
print("\n📊 MODEL REPORT")
print("-" * 40)

# Get predictions for both train and test
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate classification metrics
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

# Format similar to your example but with classification metrics
print(f"Train Accuracy: {acc_train:.3f}")
print(f"Test Accuracy: {acc_test:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}\n")

# Feature importance extraction
print("Top 5 Influential Features:")
try:
    # Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()
    
    # Access the classifier from the pipeline
    classifier = best_model.named_steps['classifier']
    
    if hasattr(classifier, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Format the feature importance output nicely
        for i, row in feature_importance.head(5).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    else:
        print("Feature importance not available for this model type.")
except Exception as e:
    print(f"Could not extract feature importance: {e}")
    # Fallback: show basic feature names if available
    try:
        if 'feature_names' in locals():
            print("Available features:", feature_names[:5])
    except:
        pass

# Additional debug information
print("\n" + "="*50)
print("DEBUG INFO:")
print(f"Model type: {type(best_model.named_steps['classifier'])}")
print(f"Has feature_importances_: {hasattr(best_model.named_steps['classifier'], 'feature_importances_')}")
print(f"Preprocessor type: {type(preprocessor)}")
print(f"Has get_feature_names_out: {hasattr(preprocessor, 'get_feature_names_out')}")