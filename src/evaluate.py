import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap

from src.config import *
from src.train import prepare_features
from src.data_preprocessing import load_data, clean_data


def evaluate_model(model, X_test, y_test):
    '''comprehensive model evaluation'''
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred)/y_test)) * 100

    print(f'\nMean Absolute Error (MAE)L: ${mae:.2f}')
    print(f'\nRoot Mean Squared Error: ${rmse:.2f}')
    print(f'\nR2 score: {r2:.4f}')
    print(f'\nMean Absolute Percentage Error: {mape:.2f}')

    errors = y_test - y_pred

    print(f'\nMean Error: ${errors.mean():.2f}')
    print(f'\nStd Error: ${errors.std():.2f}')
    print(f'\nMin Error: ${errors.min():.2f}')
    print(f'\nMax Error: ${errors.max():.2f}')

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'errors': errors
    }


def plot_results(y_test, y_pred, save_path=None):
    '''visualize predictions vs actual'''
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # plot 1: actual vs predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Charges ($)')
    axes[0, 0].set_ylabel('Predicted Charges($)')
    axes[0, 0].set_title('Actual vs Predicted Charges')

    # plot 2: residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Charges ($)')
    axes[0, 1].set_ylabel('Residuals ($)')
    axes[0, 1].set_title('Residual Plot')

    # plot 3: residual distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')

    # plot 4: error by cost tier
    df_temp = pd.DataFrame({'actual' : y_test, 'predicted' : y_pred})
    df_temp['cost_tier'] = pd.cut(df_temp['actual'],
                                bins = [0, 5000, 15000, 30000, 10000],
                                labels = ['Low', 'Medium', 'High', 'Very High']
                            )

    tier_errrors = df_temp.groupby('cost_tier').apply(
        lambda x: mean_absolute_error(x['actual'], x['predicted'])
    )

    tier_errrors.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Cost Tier')
    axes[1, 1].set_ylabel('MAE ($)')
    axes[1, 1].set_title('Error by Cost Tier')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'plot saved to {save_path}')


    plt.show()



def explain_with_shap(model, X_sample, features_names):
    '''shap explainability'''
    print('\n Shap Model Explanation')

    # create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # summary plot
    shap.summary_plot(shap_values, X_sample, feature_names=features_names, show=False)
    plt.tight_layout()
    plt.savefig(MODEL_PATH.parent / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.show()

    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature' : features_names,
        'importance': shap_importance
    }).sort_values('importance', ascending=False)

    print('\nTop 5 Most Important Features:')
    print(importance_df.head(5).to_string(index=False))

    return importance_df


def main():
    # load data and model
    df = load_data()
    df = clean_data(df)
    X, y, featue_names = prepare_features(df)

    model = joblib.load(MODEL_PATH)

    # evaluate
    metrics = evaluate_model(model, X, y)

    # visualize
    y_pred = model.predict(X)
    plot_results(y, y_pred, save_path=MODEL_PATH.parent / 'evaluation_plot.png')

    # shap explanation
    X_sample = X.sample(100, random_state=RANDOM_SEED)
    importance_df = explain_with_shap(model, X_sample, features_names)

    print('\nEvaluation Complete')


if __name__ == '__main__':
    main()
