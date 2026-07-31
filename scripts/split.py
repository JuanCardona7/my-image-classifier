
import sys
from subprocess import run

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold


from skin_lesion_ai.utils.data_utils import (
    get_project_root,
    load_metadata_parquet,
    save_metadata_parquet,
)


def split_stratified_group(df, col_grouping='patient_id', col_target='target_biopsy', test_val_size=0.2, random_state=42):
    """
    Split data by patient_id to avoid data leakage while maintaining stratification of the target variable.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The full dataset
    col_grouping : str
        Name of the column to group by (e.g., patient_id)
    col_target : str
        Name of the target column
    test_val_size : float
        Proportion of cases to include in test_val set that later will be split into test and validation sets
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    main_df, second_df : pandas DataFrames
        Splits of the original dataframe
    main_inds, second_inds : numpy arrays
        Indices of the main and second splits
    """
    # Calculamos los 'folds' necesarios para sacar el ratio que nos interesa
    desired = 1.0 / test_val_size
    n_folds = int(np.round(desired) )

    # Usamos la funcion StratifiedGroupKFold para hacer un spliter con el que haremos el split
    splitter = StratifiedGroupKFold(n_splits=n_folds,
                                  shuffle=True,
                                  random_state = random_state)
    
    
    # Split patients en train y test sets
    split = splitter.split(X=df,
                           y=df[col_target],
                           groups=df[col_grouping])
    main_inds, second_inds = next(split)
    
    # Creamos los df segun los indexs obtenidos para cada split
    main_df = df.iloc[main_inds]
    second_df = df.iloc[second_inds]

    return main_df, second_df, main_inds, second_inds



#
# load_or_create_final_metadata function definition
#

def validation_split(df_base, train_df, test_df, val_df, col_grouping='patient_id', col_target='target_biopsy') -> None:

    """""
    Validate train, test and validation splits

    Parameters:
    -----------
    df_base : pandas DataFrame
        The original dataset
    train_df : pandas DataFrame
        Training set
    test_df : pandas DataFrame
        Test set
    val_df : pandas DataFrame
        Validation set
    col_grouping : str
        Name of the column to group by (e.g., patient_id)
    col_target : str
        Name of the target column

    """""
    # Check all sets length
    print(f"Base set: {len(df_base)} samples")
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Val set: {len(val_df)} samples")

    # Percentage of the original dataset
    print(f"Train set: {len(train_df)*100/len(df_base):.2f}% samples")
    print(f"Test set: {len(test_df)*100/len(df_base):.2f}% samples")
    print(f"Val set: {len(val_df)*100/len(df_base):.2f}% samples")


    # Group check overlaps

    df_overlap1 = pd.merge(train_df, test_df,how="inner", on=[col_grouping,col_grouping])
    print(f"Train ∩ Test overlap: {len(df_overlap1)} patients")

    df_overlap2 = pd.merge(val_df, test_df,how="inner", on=[col_grouping,col_grouping])
    print(f"Validation ∩ Test overlap: {len(df_overlap2)} patients")

    df_overlap3 = pd.merge(val_df, train_df,how="inner", on=[col_grouping,col_grouping])
    print(f"Validation ∩ Train overlap: {len(df_overlap3)} patients")

    if (len(df_overlap1)) + len(df_overlap2) + len(df_overlap3) == 0:
        print("✅ No patient overlap")
    else:
        print("❌ Patient overlap detected")


    # Checkear stratificacion de nuestro split

    df_base_y = df_base[col_target].value_counts()
    train_y = train_df[col_target].value_counts()
    test_y = test_df[col_target].value_counts()
    val_y = val_df[col_target].value_counts()

    print(f"Preprocessed_df {col_target} samples count:\n {df_base_y[0]} with value '0' \n {df_base_y[1]} with value '1' \n")
    print(f"As percentatge:\n {100*df_base_y[0]/len(df_base):.2f}% with value '0' \n {100*df_base_y[1]/len(df_base):.2f}% with value '1' \n" )

    print("\nAfter our split we have: \n")

    print(f"Train_df {col_target} samples count:\n {train_y[0]} with value '0' \n {train_y[1]} with value '1' \n")
    print(f"As percentatge:\n {100*train_y[0]/len(train_df):.2f}% with value '0' \n {100*train_y[1]/len(train_df):.2f}% with value '1' \n" )

    print(f"Test_df {col_target} samples count:\n {test_y[0]} with value '0' \n {test_y[1]} with value '1' \n")
    print(f"As percentatge:\n {100*test_y[0]/len(test_df):.2f}% with value '0' \n {100*test_y[1]/len(test_df):.2f}% with value '1' \n" )

    print(f"Val_df {col_target} samples count:\n {val_y[0]} with value '0' \n {val_y[1]} with value '1' \n")
    print(f"As percentatge:\n {100*val_y[0]/len(val_df):.2f}% with value '0' \n {100*val_y[1]/len(val_df):.2f}% with value '1' \n" )

    # Checkear stratification para cada split en % hasta 1 decimal

    if np.round(100*df_base_y[0]/len(df_base), 1) == np.round(100*train_y[0]/len(train_df), 1):
        print("✅ Stratification maintained for value '0' in train set")
    else:
        print("❌ Stratification not maintained for value '0' in train set")

    if np.round(100*df_base_y[1]/len(df_base), 1) == np.round(100*train_y[1]/len(train_df), 1):
        print("✅ Stratification maintained for value '1' in train set")
    else:
        print("❌ Stratification not maintained for value '1' in train set")

    # Check stratification for test set
    if np.round(100*df_base_y[0]/len(df_base), 1) == np.round(100*test_y[0]/len(test_df), 1):
        print("✅ Stratification maintained for value '0' in test set")
    else:
        print("❌ Stratification not maintained for value '0' in test set")

    if np.round(100*df_base_y[1]/len(df_base), 1) == np.round(100*test_y[1]/len(test_df), 1):
        print("✅ Stratification maintained for value '1' in test set")
    else:
        print("❌ Stratification not maintained for value '1' in test set")

    # Check stratification for validation set
    if np.round(100*df_base_y[0]/len(df_base), 1) == np.round(100*val_y[0]/len(val_df), 1):
        print("✅ Stratification maintained for value '0' in validation set")
    else:
        print("❌ Stratification not maintained for value '0' in validation set")

    if np.round(100*df_base_y[1]/len(df_base), 1) == np.round(100*val_y[1]/len(val_df), 1):
        print("✅ Stratification maintained for value '1' in validation set")
    else:
        print("❌ Stratification not maintained for value '1' in validation set")


    # Numero de lesiones en cada split y porcentaje del original (preprocessed)
    print(f"Preprocessed set:\n {df_base['isic_id'].nunique()} lesions\n")

    print(f"Train set:\n {train_df['isic_id'].nunique()} lesions")
    print(f" {100*train_df['isic_id'].nunique()/df_base['isic_id'].nunique()} % \nof preprocessed set\n")

    print(f"Test set:\n {test_df['isic_id'].nunique()} lesions")
    print(f" {100*test_df['isic_id'].nunique()/df_base['isic_id'].nunique()} % \nof preprocessed set\n")

    print(f"Validation set:\n {val_df['isic_id'].nunique()} lesions")
    print(f" {100*val_df['isic_id'].nunique()/df_base['isic_id'].nunique()} % \nof preprocessed set\n")

    # se cumple igualdad de train + test + val = preprocessed
    print(f"Total lesions in splits: {train_df['isic_id'].nunique() + test_df['isic_id'].nunique() + val_df['isic_id'].nunique()} lesions")
    print(f"Preprocessed lesions: {df_base['isic_id'].nunique()} lesions")

    if train_df['isic_id'].nunique() + test_df['isic_id'].nunique() + val_df['isic_id'].nunique() == df_base['isic_id'].nunique():
        print("✅ Train + Test + Val = Preprocessed")
    else:
        print("❌ Train + Test + Val != Preprocessed")


    # Numero de pacientes en cada split
    print(f"Preprocessed set:\n {df_base[col_grouping].nunique()} patients\n")

    print(f"Train set:\n {train_df[col_grouping].nunique()} patients")
    print(f" {100*train_df[col_grouping].nunique()/df_base[col_grouping].nunique()} % \nof preprocessed set\n")

    print(f"Test set:\n {test_df[col_grouping].nunique()} patients")
    print(f" {100*test_df[col_grouping].nunique()/df_base[col_grouping].nunique()} % \nof preprocessed set\n")

    print(f"Validation set:\n {val_df[col_grouping].nunique()} patients")
    print(f" {100*val_df[col_grouping].nunique()/df_base[col_grouping].nunique()} % \nof preprocessed set\n")

    # Se cumple igualdad de train + test + val = preprocessed
    print(f"Total patients in splits: {train_df[col_grouping].nunique() + test_df[col_grouping].nunique() + val_df[col_grouping].nunique()} patients")
    print(f"Preprocessed patients: {df_base[col_grouping].nunique()} patients")

    if train_df[col_grouping].nunique() + test_df[col_grouping].nunique() + val_df[col_grouping].nunique() == df_base[col_grouping].nunique():
        print("✅ Train + Test + Val = Preprocessed")
    else:
        print("❌ Train + Test + Val != Preprocessed")


#
# load_or_create_final_metadata function definition
#

def load_or_create_final_metadata() -> pd.DataFrame:
    """Load latest final metadata file or generate it from raw data."""

    repo_root = get_project_root()
    script_path = repo_root / "scripts" / "final_preprocess_data.py"

    try:
        return load_metadata_parquet(
            stage="processed",
            filename="final_preprocessed_from_raw",
            timestamp_flag=True,
        )

    except FileNotFoundError:
        run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            check=True,
        )

        return load_metadata_parquet(
            stage="processed",
            filename="final_preprocessed_from_raw",
            timestamp_flag=True,
        )

# provo
def main() -> None:

    # Load the preprocessed data
    df_preprocessed = load_or_create_final_metadata()
    print(f"Final preprocessed metadata loaded: {df_preprocessed.shape}")

    # split train test-val
    train_df, test_val_df, train_inds, test_val_inds = split_stratified_group(
        df_preprocessed,
        col_grouping='patient_id',
        col_target='target_biopsy',
        test_val_size=0.2,
        random_state=42
    )

    # split test-val
    test_df, val_df, test_inds, val_inds = split_stratified_group(
        test_val_df,
        col_grouping='patient_id',
        col_target='target_biopsy',
        test_val_size=0.5,
        random_state=42
    )

    # Validate the splits
    validation_split(df_preprocessed, train_df, test_df, val_df, 
                     col_grouping='patient_id', 
                     col_target='target_biopsy')

    
    # Save the splits and index
    train_path = save_metadata_parquet(train_df, stage="processed", name="train_split", timestamp=True)
    test_path = save_metadata_parquet(test_df, stage="processed", name="test_split", timestamp=True)
    val_path = save_metadata_parquet(val_df, stage="processed", name="val_split", timestamp=True)

    print(f"Train saved to: {train_path}")
    print(f"Test saved to: {test_path}")
    print(f"Validation saved to: {val_path}")

    split_indices_path = save_metadata_parquet(
        pd.DataFrame({"train_indices": pd.Series(train_inds), "test_val_indices": pd.Series(test_val_inds),
                        "test_indices": pd.Series(test_inds), "val_indices": pd.Series(val_inds)}),
        stage="interim",
        name="split_indices",
        timestamp=True,
    )

    print(f"Split indices saved to: {split_indices_path}")


if __name__ == "__main__":
    main()