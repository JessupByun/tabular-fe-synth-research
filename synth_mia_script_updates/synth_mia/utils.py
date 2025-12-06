import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder  

from sklearn import metrics
import torch
from geomloss import SamplesLoss

def create_random_equal_dfs(df, df_size, num_dfs=4, seed=42):
    if len(df) < df_size * num_dfs:
        raise ValueError("Not enough rows in the original dataframe to create the specified number of dataframes of the given size.")

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return tuple(df_shuffled[i*df_size:(i+1)*df_size].reset_index(drop=True) for i in range(num_dfs))

def fit_transformer(X, categorical_encoding='one-hot'):
    """
    Fits a transformer on the reference dataset and stores column information.
    
    Parameters:
    X (pd.DataFrame): Reference dataset to fit the transformer
    categorical_encoding (str): Encoding method for categorical variables ('onehot' or 'ordinal')
    
    Returns:
    tuple: (fitted_transformer, feature_names)
    """
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numerical_transformer = StandardScaler()
    
    if categorical_encoding == 'one-hot':
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif categorical_encoding == 'ordinal':
        categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        raise ValueError("Unsupported categorical encoding. Choose 'one-hot' or 'ordinal'.")
    
    full_transformer = ColumnTransformer([
        ('numerical', numerical_transformer, numerical_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ], remainder='drop')
    
    full_transformer.fit(X)
    
    feature_names = numerical_cols[:]
    
    if categorical_encoding == 'one-hot' and categorical_cols:
        encoder = full_transformer.named_transformers_['categorical']
        for i, feature in enumerate(categorical_cols):
            categories = encoder.categories_[i]
            for cat in categories:
                feature_names.append(f"{feature}_{cat}")
    else:
        feature_names.extend(categorical_cols)
    
    return full_transformer, feature_names

def transform_data(X, transformer_info):
    """
    Transforms data using a pre-fitted transformer and ensures consistent columns.
    
    Parameters:
    X (pd.DataFrame): Dataset to transform
    transformer_info: tuple of (transformer, feature_names)
    
    Returns:
    np.ndarray: Transformed data with consistent columns
    """
    transformer, feature_names = transformer_info
    transformed_data = transformer.transform(X)
    
    if transformed_data.shape[1] != len(feature_names):
        padding = np.zeros((transformed_data.shape[0], len(feature_names) - transformed_data.shape[1]))
        transformed_data = np.hstack([transformed_data, padding])
    
    return transformed_data

def tabular_preprocess(mem, non_mem, synth, ref=None, fit_target='synth', categorical_encoding='one-hot'):
    if fit_target not in ['synth', 'ref']:
        raise ValueError("fit_target must be 'synth' or 'ref' to prevent data leakage.")
    
    # If ref is None and fit_target is 'ref', raise an error
    if ref is None and fit_target == 'ref':
        raise ValueError("Cannot use 'ref' as fit_target when ref is None.")
    
    fit_data = synth if fit_target == 'synth' else ref
    transformer = fit_transformer(fit_data, categorical_encoding)
    
    # Handle datasets based on whether ref is provided
    if ref is None:
        datasets = [mem, non_mem, synth]
        transformed = [transform_data(data, transformer) for data in datasets]
        return (*transformed, transformer)
    else:
        datasets = [mem, non_mem, synth, ref]
        transformed = [transform_data(data, transformer) for data in datasets]
        return (*transformed, transformer)
