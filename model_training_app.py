import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import zipfile
from pathlib import Path
from itertools import combinations

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ===================
# Utility Functions
# ===================

def load_and_clean_data(df, target_column='pic50'):
    """
    Load and clean the dataset from a DataFrame, focusing on numeric columns
    and returning X, y for modeling.
    """
    df = df.copy()
    # Ensure columns are stripped of whitespace
    df.columns = df.columns.str.strip()
    
    # Automatically select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # If target column is not numeric, try converting
    if target_column in df.columns and target_column not in numeric_cols:
        df[target_column] = df[target_column].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '', regex=False)
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        numeric_cols.append(target_column)

    # Drop rows with NA in the numeric columns
    df = df.dropna(subset=numeric_cols)
    
    # Separate X and y
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        # If the target column is not present, return the entire dataframe as X, and None for y
        X = df
        y = None
    
    return X, y

def remove_highly_correlated_features(X, threshold=0.85):
    """
    Remove features that have a correlation higher than the specified threshold.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    return X_reduced

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for each feature.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    vif_df = pd.DataFrame({'Feature': X.columns, 'VIF': vif})
    return vif_df

def select_features_via_vif(X, threshold=5.0):
    """
    Iteratively remove features with VIF greater than threshold.
    """
    X_selected = X.copy()
    while True:
        vif_df = calculate_vif(X_selected)
        max_vif = vif_df['VIF'].max()
        if max_vif > threshold:
            # Drop the feature with the highest VIF
            feature_to_drop = vif_df.loc[vif_df['VIF'] == max_vif, 'Feature'].values[0]
            X_selected = X_selected.drop(columns=[feature_to_drop])
        else:
            break
    return X_selected

def create_pipeline(model):
    """
    Create a pipeline with scaling + model.
    """
    steps = [('scaler', StandardScaler()),
             ('model', model)]
    return Pipeline(steps)

def get_param_grid(model_name):
    """
    Hyperparameter grids for different models.
    """
    param_grids = {
        'LinearRegression': {},
        'Ridge': {'model__alpha': [0.1, 1.0, 10.0, 100.0]},
        'Lasso': {'model__alpha': [0.01, 0.1, 1.0, 10.0]}
    }
    return param_grids.get(model_name, {})

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate the model, return metrics.
    """
    model.fit(X_train, y_train)
   
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
   
    # Adjusted R²
    n = X_test.shape[0]
    p = X_test.shape[1]
    if n > p + 1:
        adj_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
    else:
        adj_r2_test = None
   
    # Cross-validation on combined dataset
    cv_scores = cross_val_score(model, pd.concat([X_train, X_test]),
                                pd.concat([y_train, y_test]), cv=5, scoring='r2')
    cv_r2 = np.mean(cv_scores)
   
    # VIF (after scaling)
    scaler = model.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    vif = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
    vif_df = pd.DataFrame({'Feature': X_train.columns, 'VIF': vif})
   
    # Equation (for linear models)
    if hasattr(model.named_steps['model'], 'coef_'):
        coefficients = model.named_steps['model'].coef_
        intercept = model.named_steps['model'].intercept_
        eq_terms = []
        for coef, feat in zip(coefficients, X_train.columns):
            eq_terms.append(f"{coef:.3f}*{feat}")
        equation = f"{intercept:.3f} + " + " + ".join(eq_terms)
    else:
        equation = "N/A"
   
    metrics = {
        'R2_train': r2_train,
        'R2_test': r2_test,
        'Adjusted_R2': adj_r2_test,
        'MSE': mse_test,
        'RMSE': rmse_test,
        'MAE': mae_test,
        'CV_R2': cv_r2,
        'VIF': vif_df,
        'Equation': equation
    }
   
    return metrics

def train_and_select_models(
    X, y,
    models,
    param_grids,
    # Predefined thresholds (minimal user input):
    corr_threshold=0.85,
    vif_threshold=5.0,
    adjusted_r2_threshold=0.5,
    cv_r2_threshold=0.6,
    min_features=3,
    max_features=5,
    max_models=5
):
    """
    Train multiple models with different feature combinations and select best ones.
    """
    best_models = []
    all_combinations = []

    # Generate all combinations of features between min_features and max_features
    for n in range(min_features, max_features + 1):
        all_combinations += list(combinations(X.columns, n))
    
    total_combos = len(all_combinations) * len(models)
    st.write(f"Trying {total_combos} total combinations...")

    progress_count = 0
    progress_bar = st.progress(progress_count / total_combos)

    for combo in all_combinations:
        # Subset the data to these features
        X_subset = X[list(combo)]
       
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42
        )
       
        # Calculate VIF on training set
        vif_df = calculate_vif(X_train)
        if vif_df['VIF'].max() > vif_threshold:
            # Skip this combination
            progress_count += len(models)
            progress_bar.progress(progress_count / total_combos)
            continue
       
        # For each model
        for model_name, model in models.items():
            # Create pipeline
            pipeline = create_pipeline(model)
            param_grid = param_grids.get(model_name, {})
            
            # Perform GridSearch if needed
            if param_grid:
                grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
                grid.fit(X_train, y_train)
                best_pipeline = grid.best_estimator_
                best_params = grid.best_params_
            else:
                # No hyperparameters, just fit
                pipeline.fit(X_train, y_train)
                best_pipeline = pipeline
                best_params = {}
           
            # Evaluate the best pipeline
            metrics = evaluate_model(best_pipeline, X_train, y_train, X_test, y_test)
            
            # Check if it meets our selection thresholds
            if metrics['Adjusted_R2'] is not None and metrics['Adjusted_R2'] >= adjusted_r2_threshold:
                if metrics['CV_R2'] >= cv_r2_threshold:
                    if (metrics['VIF']['VIF'] <= vif_threshold).all():
                        # Save it
                        best_models.append({
                            'Model': model_name,
                            'Features': combo,
                            'Best Params': best_params,
                            'Metrics': metrics,
                            'Best Estimator': best_pipeline,
                            'X_train': X_train,
                            'y_train': y_train,
                            'X_test': X_test,
                            'y_test': y_test
                        })
                        if len(best_models) >= max_models:
                            # If we've collected enough models, return early
                            return sorted(best_models, key=lambda x: x['Metrics']['CV_R2'], reverse=True)

            progress_count += 1
            progress_bar.progress(progress_count / total_combos)

    # Sort by CV_R2 descending
    best_models_sorted = sorted(best_models, key=lambda x: x['Metrics']['CV_R2'], reverse=True)
    return best_models_sorted


# ===================
# Streamlit Interface
# ===================

def main():
    st.title("Simple Linear Model Training Demo")
    st.write("Upload your dataset and specify the target column. We'll do the rest!")

    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    # Text input for target column
    target_column = st.text_input("Target Column", value="pic50")

    # Only run if a file is uploaded
    if uploaded_file and target_column:
        if st.button("Train Models"):
            # Read DataFrame from the uploaded Excel file
            df = pd.read_excel(uploaded_file)
            
            # Load and clean data
            X, y = load_and_clean_data(df, target_column=target_column)
            if y is None:
                st.error(f"Target column '{target_column}' not found or not numeric.")
                return
            
            # Remove highly correlated features globally
            X_reduced = remove_highly_correlated_features(X, threshold=0.85)
            
            # Remove features based on VIF
            X_selected = select_features_via_vif(X_reduced, threshold=5.0)

            # Define models
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(random_state=42),
                'Lasso': Lasso(random_state=42)
            }

            param_grids = {
                'LinearRegression': get_param_grid('LinearRegression'),
                'Ridge': get_param_grid('Ridge'),
                'Lasso': get_param_grid('Lasso')
            }

            # Train & select
            best_models = train_and_select_models(
                X_selected, y, 
                models=models, 
                param_grids=param_grids,
                # Using fixed thresholds for simplicity:
                corr_threshold=0.85,
                vif_threshold=5.0,
                adjusted_r2_threshold=0.5,
                cv_r2_threshold=0.6,
                min_features=3,
                max_features=5,
                max_models=5
            )

            if not best_models:
                st.warning("No models met the criteria!")
            else:
                st.success(f"Found {len(best_models)} models meeting the criteria.")
                
                # Display the top model info
                top_model = best_models[0]
                st.write("### Best Model Summary")
                st.write(f"**Model:** {top_model['Model']}")
                st.write(f"**Features:** {top_model['Features']}")
                st.write(f"**Best Params:** {top_model['Best Params']}")
                st.write(f"**CV R²:** {top_model['Metrics']['CV_R2']:.3f}")
                st.write(f"**Equation:** {top_model['Metrics']['Equation']}")
                
                # Let the user download all best models as a ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    # 1) Write a summary text
                    summary_text = io.StringIO()
                    for i, bm in enumerate(best_models):
                        summary_text.write(f"Model {i+1}\n")
                        summary_text.write(f"  Name: {bm['Model']}\n")
                        summary_text.write(f"  Features: {bm['Features']}\n")
                        summary_text.write(f"  Best Params: {bm['Best Params']}\n")
                        summary_text.write(f"  CV R²: {bm['Metrics']['CV_R2']:.4f}\n")
                        summary_text.write(f"  Equation: {bm['Metrics']['Equation']}\n\n")
                    zf.writestr("best_models_summary.txt", summary_text.getvalue())
                    
                    # 2) For each best model, save train/test CSV + joblib
                    for i, bm in enumerate(best_models):
                        X_train = bm['X_train']
                        y_train = bm['y_train']
                        train_df = pd.concat([X_train, y_train], axis=1)
                        zf.writestr(f"model_{i+1}_train.csv", train_df.to_csv(index=False))

                        X_test = bm['X_test']
                        y_test = bm['y_test']
                        test_df = pd.concat([X_test, y_test], axis=1)
                        zf.writestr(f"model_{i+1}_test.csv", test_df.to_csv(index=False))

                        model_bytes = io.BytesIO()
                        joblib.dump(bm['Best Estimator'], model_bytes)
                        zf.writestr(f"model_{i+1}.joblib", model_bytes.getvalue())
                
                st.download_button(
                    label="Download Best Models (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="best_models.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()
