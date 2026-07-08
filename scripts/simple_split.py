import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 


from skin_lesion_ai.utils.data_utils import (
    load_metadata_parquet,
    save_metadata_parquet,
)


def split_by_patient_id(df, patient_id_col='patient_id', test_size=0.2, random_state=42):
    """
    Split data by patient_id to avoid data leakage.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The full dataset
    patient_id_col : str
        Name of the patient ID column
    test_size : float
        Proportion of patients to include in test set
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_df, test_df : pandas DataFrames
    """
    # Usamos la funcion GroupShuffleSplit para hacer un spliter con el que haremos el split
    splitter = GroupShuffleSplit(test_size=test_size,
                                  n_splits=1,
                                  random_state = random_state)
    
    
    # Split patients en train y test sets
    split = splitter.split(df, groups=df['patient_id'])
    train_inds, test_inds = next(split)
    
    # Creamos los df segun los indexs obtenidos para cada split
    train_df = df.iloc[train_inds]
    test_df = df.iloc[test_inds]

    # Chekeamos la longitud de los sets
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Chekeamos el overlap de patients entre los splits
    df_overlap_train_test = pd.merge(train_df,
                                      test_df,
                                      how="inner",
                                      on=["patient_id","patient_id"])

    print(f"Train ∩ Test overlap: {len(df_overlap_train_test)} patients")

    
    
    return train_df, test_df

# Example usage
# df = pd.read_csv('your_data.csv')
# train_df, test_df = split_by_patient_id(df, patient_id_col='patient_id', test_size=0.2)