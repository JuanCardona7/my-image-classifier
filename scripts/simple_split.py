import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold


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
    # Calculamos los 'folds' necesarios para sacar el ratio que nos interesa
    desired = 1.0 / test_size
    n_folds = int(np.ceil(desired) )

    # Usamos la funcion StratifiedGroupKFold para hacer un spliter con el que haremos el split
    splitter = StratifiedGroupKFold(n_splits=n_folds,
                                  shuffle=True,
                                  random_state = random_state)
    
    
    # Split patients en train y test sets
    split = splitter.split(X=df,
                           y=df['target_biopsy'],
                           groups=df['patient_id'])
    train_inds, test_val_inds = next(split)
    
    # Creamos los df segun los indexs obtenidos para cada split
    train_df = df.iloc[train_inds]
    test_val_df = df.iloc[test_val_inds]

    # Hacemos otro splitter para sacar test y val, steando valores
    val_size = 0.5
    desired_val = 1.0 / val_size
    n_folds_val = int(np.ceil(desired_val) )
    splitter_val = StratifiedGroupKFold(n_splits=n_folds_val,
                                        shuffle=True,
                                        random_state=random_state)
    
    # Creamos el obj split asociado
    split_val = splitter_val.split(X=test_val_df,
                                   y=test_val_df['target_biopsy'],
                                   groups=test_val_df['patient_id'])
    
    # Aplicamos el split 1 iteracion
    test_inds, val_inds = next(split_val)

    # Obtenemos los df test y val
    test_df = test_val_df.iloc[test_inds]
    val_df = test_val_df.iloc[val_inds]

    # Chekeamos la longitud de los sets
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Val set: {len(val_df)} samples")
    
    # Chekeamos el overlap de patients entre los splits
    df_overlap_train_test = pd.merge(train_df,
                                      test_df,
                                      how="inner",
                                      on=["patient_id","patient_id"])

    

    df_overlap_val_test = pd.merge(val_df,
                                   test_df,
                                   how="inner",
                                   on=["patient_id","patient_id"])
    
    df_overlap_val_train = pd.merge(val_df,
                                    train_df,
                                    how="inner",
                                    on=["patient_id","patient_id"])
    
    print(f"Train ∩ Test overlap: {len(df_overlap_train_test)} patients")
    print(f"Validation ∩ Test overlap: {len(df_overlap_val_test)} patients")
    print(f"Validation ∩ Train overlap: {len(df_overlap_val_train)} patients")

    # Aun queda por implementar el chekeo de la stratificacion de los splits
    # y guardar los splits en parquets
    # todo esto ya esta hecho en el notebook
    
    return train_df, test_df

# Example usage
# df = pd.read_csv('your_data.csv')
# train_df, test_df = split_by_patient_id(df, patient_id_col='patient_id', test_size=0.2)