# src/inspect_model.py
import joblib
import sys
from pathlib import Path

model_path = Path("models/linear_regression.joblib")
if not model_path.exists():
    print(f"Model not found at {model_path}. Update path to the joblib model file.")
    sys.exit(1)

model = joblib.load(model_path)
print("Loaded model:", type(model))

# If pipeline or transformer saved, try to extract feature names
if hasattr(model, "feature_names_in_"):
    print("Expected feature names (model.feature_names_in_):")
    print(list(model.feature_names_in_))
else:
    # try digging into pipeline
    try:
        # if pipeline: last step may be estimator, earlier step may be ColumnTransformer
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            # try to find column transformer step
            for name, step in model.named_steps.items():
                if hasattr(step, "get_feature_names_out"):
                    try:
                        print("Feature names from step", name)
                        print(step.get_feature_names_out())
                        break
                    except Exception:
                        pass
        print("If the model has no `feature_names_in_`, check your preprocessing code to see which features were used.")
    except Exception:
        print("Cannot determine feature names programmatically. Open your preprocessing/features file to see the feature list.")
