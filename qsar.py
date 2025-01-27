import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import zipfile
import joblib

from itertools import combinations
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------------------------
# 1. Define utility functions
# ------------------------------------------------------------------------------

def load_and_clean_data(df, target_column='pic50', exclude_columns=None, numeric_columns=None):
    """
    Clean the dataset, excluding specified columns, converting columns to numeric, etc.
    Instead of loading from file_path, we directly accept a DataFrame (df).
    """
    try:
        # Strip any trailing or leading spaces from column names
        df.columns = df.columns.str.strip()

        # Exclude specified columns
        if exclude_columns:
            df = df.drop(columns=exclude_columns, errors='ignore')

        # Automatically select numeric columns if none are specified
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Ensure target column is included in numeric_columns
        if target_column not in numeric_columns:
            numeric_columns.append(target_column)

        # Clean numeric columns: replace commas with dots, remove spaces, convert to float
        data_cleaned = df.copy()
        for col in numeric_columns:
            data_cleaned[col] = (
                data_cleaned[col]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .str.replace(' ', '', regex=False)
            )
            data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')

        # Drop rows with NA in numeric columns including target
        data_cleaned = data_cleaned.dropna(subset=numeric_columns)

        # Separate features and target
        X = data_cleaned.drop(columns=[target_column])
        y = data_cleaned[target_column]

        return X, y

    except Exception as e:
        st.error(f"Error in load_and_clean_data: {e}")
        raise


def remove_highly_correlated_features(X, threshold=0.85):
    """
    Remove features that have a correlation higher than the specified threshold.
    """
    try:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_reduced = X.drop(columns=to_drop)
        return X_reduced

    except Exception as e:
        st.error(f"Error in remove_highly_correlated_features: {e}")
        raise


def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the dataframe.
    """
    try:
        X_scaled = StandardScaler().fit_transform(X)
        vif = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
        vif_df = pd.DataFrame({'Feature': X.columns, 'VIF': vif})
        return vif_df

    except Exception as e:
        st.error(f"Error in calculate_vif: {e}")
        raise


def select_features_via_vif(X, threshold=5.0):
    """
    Iteratively remove features with VIF greater than the threshold.
    """
    try:
        X_selected = X.copy()
        while True:
            vif_df = calculate_vif(X_selected)
            max_vif = vif_df['VIF'].max()
            if max_vif > threshold:
                feature_to_drop = vif_df.loc[vif_df['VIF'] == max_vif, 'Feature'].values[0]
                X_selected = X_selected.drop(columns=[feature_to_drop])
            else:
                break
        return X_selected

    except Exception as e:
        st.error(f"Error in select_features_via_vif: {e}")
        raise


def create_pipeline(model):
    """
    Create a machine learning pipeline with scaling and the specified model.
    """
    steps = [('scaler', StandardScaler()), ('model', model)]
    return Pipeline(steps)


def get_param_grid(model_name):
    """
    Define hyperparameter grids for different models.
    """
    param_grids = {
        'LinearRegression': {},
        'Ridge': {'model__alpha': [0.1, 1.0, 10.0, 100.0]},
        'Lasso': {'model__alpha': [0.01, 0.1, 1.0, 10.0]}
    }
    return param_grids.get(model_name, {})


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train the model and evaluate it using various metrics.
    """
    try:
        # Train the model
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
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

        # Cross-Validation
        cv_scores = cross_val_score(model,
                                    pd.concat([X_train, X_test]),
                                    pd.concat([y_train, y_test]),
                                    cv=5, scoring='r2', n_jobs=-1)
        cv_r2 = np.mean(cv_scores)

        # VIF Calculation (after scaling)
        scaler = model.named_steps['scaler']
        X_train_scaled = scaler.transform(X_train)
        vif = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]
        vif_df = pd.DataFrame({'Feature': X_train.columns, 'VIF': vif})

        # Extract equation for linear models
        if hasattr(model.named_steps['model'], 'coef_'):
            coefficients = model.named_steps['model'].coef_
            intercept = model.named_steps['model'].intercept_
            equation_terms = [f"{coef:.3f}*{feat}" for coef, feat in zip(coefficients, X_train.columns)]
            equation = " + ".join(equation_terms)
            equation = f"{intercept:.3f} + " + equation
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

    except Exception as e:
        st.error(f"Error in evaluate_model: {e}")
        return None


def train_and_select_models(
    X, y, models, param_grids,
    min_features=3, max_features=5, vif_threshold=5.0,
    adjusted_r2_threshold=0.5, cv_r2_threshold=0.6, max_models=10
):
    """
    Train multiple models with different feature combinations and select the best based on metrics.
    Now collects all models and applies fallback if no models meet criteria.
    """
    selected_models = []
    all_trained_models = []
    all_combinations = []

    # Generate all feature combinations
    for n in range(min_features, max_features + 1):
        all_combinations += list(combinations(X.columns, n))

    total_iterations = len(all_combinations) * len(models)
    progress_bar = st.progress(0)
    iteration_count = 0

    for combo in all_combinations:
        X_subset = X[list(combo)]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42
        )

        # Calculate initial VIF to skip highly collinear feature sets
        try:
            vif_initial = calculate_vif(X_train)
            if vif_initial['VIF'].max() > vif_threshold:
                iteration_count += len(models)
                progress_bar.progress(min(iteration_count / total_iterations, 1.0))
                continue  # Skip this feature combination
        except:
            iteration_count += len(models)
            progress_bar.progress(min(iteration_count / total_iterations, 1.0))
            continue

        # Train each model in 'models'
        for model_name, model in models.items():
            pipeline = create_pipeline(model)
            param_grid = param_grids.get(model_name, {})

            # If hyperparameters are to be tuned
            if param_grid:
                grid = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)
                best_pipeline = grid.best_estimator_
            else:
                best_pipeline = pipeline.fit(X_train, y_train)

            # Evaluate the model
            metrics = evaluate_model(best_pipeline, X_train, y_train, X_test, y_test)

            if metrics:
                model_info = {
                    'Model': model_name,
                    'Features': combo,
                    'Best Parameters': grid.best_params_ if param_grid else {},
                    'R2_train': metrics['R2_train'],
                    'R2_test': metrics['R2_test'],
                    'Adjusted_R2': metrics['Adjusted_R2'],
                    'MSE': metrics['MSE'],
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'CV_R2': metrics['CV_R2'],
                    'VIF': metrics['VIF'],
                    'Equation': metrics['Equation'],
                    'Best Estimator': best_pipeline,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test
                }

                # Check if model meets all selection criteria
                if (metrics['Adjusted_R2'] is not None and
                    metrics['Adjusted_R2'] >= adjusted_r2_threshold and
                    metrics['CV_R2'] >= cv_r2_threshold and
                    (metrics['VIF']['VIF'] <= vif_threshold).all()):
                    selected_models.append(model_info)

                # Always add to all_trained_models for fallback
                all_trained_models.append(model_info)

            iteration_count += 1
            progress_bar.progress(min(iteration_count / total_iterations, 1.0))

            if len(selected_models) >= max_models:
                break

        if len(selected_models) >= max_models:
            break

    # Sort selected_models based on CV_R2
    selected_models_sorted = sorted(selected_models, key=lambda x: x['CV_R2'], reverse=True)

    # If no models meet the criteria, select top models based on CV_R2
    if not selected_models_sorted:
        st.warning("No models met the specified criteria. Selecting the top-performing models instead.")
        all_trained_models_sorted = sorted(all_trained_models, key=lambda x: x['CV_R2'], reverse=True)
        selected_models_sorted = all_trained_models_sorted[:max_models]

    return selected_models_sorted


def save_model_details_to_zip(best_models):
    """
    Save the best models' details, train/test datasets, and the trained models
    into an in-memory zip file, returning a BytesIO object for download.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Prepare text details
        details_content = []
        for i, model in enumerate(best_models):
            details_content.append(f"Model {i+1} Details:\n")
            details_content.append(f"Model: {model['Model']}\n")
            details_content.append(f"Features: {model['Features']}\n")
            details_content.append(f"Best Parameters: {model['Best Parameters']}\n")
            details_content.append(f"Training R²: {model['R2_train']:.4f}\n")
            details_content.append(f"Test R²: {model['R2_test']:.4f}\n")
            details_content.append(f"Adjusted R²: {model['Adjusted_R2']:.4f}\n")
            details_content.append(f"MSE: {model['MSE']:.4f}\n")
            details_content.append(f"RMSE: {model['RMSE']:.4f}\n")
            details_content.append(f"MAE: {model['MAE']:.4f}\n")
            details_content.append(f"CV R²: {model['CV_R2']:.4f}\n")
            details_content.append("VIF:\n")
            details_content.append(model['VIF'].to_string(index=False))
            details_content.append("\n")
            details_content.append(f"Equation: {model['Equation']}\n")
            details_content.append("\n" + "-"*80 + "\n\n")

        # Write details file to the zip
        details_filename = "best_models_details.txt"
        zf.writestr(details_filename, "\n".join(details_content))

        # Save train/test datasets and models
        for i, model in enumerate(best_models):
            # CSV for train
            X_train, y_train = model['X_train'], model['y_train']
            train_data = pd.concat([X_train, y_train], axis=1)
            train_csv = train_data.to_csv(index=False)
            zf.writestr(f"data_model_{i+1}_train.csv", train_csv)

            # CSV for test
            X_test, y_test = model['X_test'], model['y_test']
            test_data = pd.concat([X_test, y_test], axis=1)
            test_csv = test_data.to_csv(index=False)
            zf.writestr(f"data_model_{i+1}_test.csv", test_csv)

            # Save the trained model to bytes, then add to zip
            model_buffer = BytesIO()
            joblib.dump(model['Best Estimator'], model_buffer)
            model_buffer.seek(0)
            zf.writestr(f"model_{i+1}.joblib", model_buffer.read())

    zip_buffer.seek(0)
    return zip_buffer

# ------------------------------------------------------------------------------
# 2. Build the Streamlit Interface
# ------------------------------------------------------------------------------

def main():
    st.title("Automated Linear Model Builder")
    st.write(
        """
        This application allows you to:
        1. Upload a dataset (CSV or XLSX).
        2. Define the target column.
        3. Adjust thresholds (Correlation, VIF, Adjusted R², etc.).
        4. Select the min and max number of features.
        5. Train and select the best models.
        6. Download results (models + train/test splits) in a single ZIP.
        """
    )

    # Sidebar for parameters
    st.sidebar.header("Parameters & Thresholds")
    corr_threshold = st.sidebar.number_input(
        "Correlation Threshold (remove features above this correlation):",
        value=0.85, min_value=0.0, max_value=1.0, step=0.01
    )
    vif_threshold = st.sidebar.number_input(
        "VIF Threshold:",
        value=5.0, min_value=1.0, max_value=100.0, step=1.0
    )
    adjusted_r2_threshold = st.sidebar.number_input(
        "Minimum Adjusted R² Threshold:",
        value=0.3, min_value=0.0, max_value=1.0, step=0.01
    )
    cv_r2_threshold = st.sidebar.number_input(
        "Minimum Cross-Validated R² Threshold:",
        value=0.5, min_value=0.0, max_value=1.0, step=0.01
    )
    min_features = st.sidebar.number_input(
        "Minimum Number of Features:",
        value=3, min_value=1, step=1
    )
    max_features = st.sidebar.number_input(
        "Maximum Number of Features:",
        value=5, min_value=1, step=1
    )
    max_models = st.sidebar.number_input(
        "Maximum Number of Models to Output:",
        value=10, min_value=1, step=1
    )

    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            # Read data accordingly
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format! Please upload a CSV or XLSX.")
                return

            st.write("Data Preview:")
            st.dataframe(df.head())

            all_columns = df.columns.tolist()
            target_column = st.selectbox("Select Target Column", all_columns, index=len(all_columns)-1)

            # Let user specify which columns to exclude
            st.write("If you have columns to exclude (e.g., ID or 'N°'), select them below.")
            exclude_columns = st.multiselect("Exclude Columns", options=all_columns, default=[])

            # Button to run the main process
            if st.button("Train Models"):
                with st.spinner("Processing... This may take a while depending on data size."):
                    # 1) Load and clean data
                    X, y = load_and_clean_data(df, target_column=target_column, exclude_columns=exclude_columns)

                    # 2) Remove highly correlated features
                    X_reduced = remove_highly_correlated_features(X, threshold=corr_threshold)

                    # 3) Further feature selection via VIF
                    X_selected = select_features_via_vif(X_reduced, threshold=vif_threshold)

                    # 4) Define models
                    models = {
                        'LinearRegression': LinearRegression(),
                        'Ridge': Ridge(random_state=42),
                        'Lasso': Lasso(random_state=42)
                    }

                    # 5) Define parameter grids
                    param_grids = {
                        'LinearRegression': get_param_grid('LinearRegression'),
                        'Ridge': get_param_grid('Ridge'),
                        'Lasso': get_param_grid('Lasso')
                    }

                    # 6) Train and select best models
                    best_models = train_and_select_models(
                        X_selected, y, models, param_grids,
                        min_features=min_features, max_features=max_features,
                        vif_threshold=vif_threshold,
                        adjusted_r2_threshold=adjusted_r2_threshold,
                        cv_r2_threshold=cv_r2_threshold,
                        max_models=max_models
                    )

                    if not best_models:
                        st.warning("No models were trained. Please check your data and parameters.")
                    else:
                        st.success(f"Found {len(best_models)} models.")

                        # Display info about the top model
                        top_model = best_models[0]
                        st.write("### Best Model Summary:")
                        st.write(f"**Model:** {top_model['Model']}")
                        st.write(f"**Features:** {top_model['Features']}")
                        st.write(f"**R² (Test):** {top_model['R2_test']:.4f}")
                        st.write(f"**Adjusted R² (Test):** {top_model['Adjusted_R2']:.4f}")
                        st.write(f"**CV R²:** {top_model['CV_R2']:.4f}")
                        st.write(f"**Equation:** {top_model['Equation']}")

                        # Prepare ZIP for download
                        zip_buffer = save_model_details_to_zip(best_models)

                        st.download_button(
                            label="Download Results (ZIP)",
                            data=zip_buffer,
                            file_name="model_results.zip",
                            mime="application/zip"
                        )

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.info("Please upload a CSV or XLSX file to begin.")


if __name__ == "__main__":
    main()
