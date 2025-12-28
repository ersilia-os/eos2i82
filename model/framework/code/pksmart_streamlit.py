import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import os
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from mordred import Calculator, descriptors
from dimorphite_dl.dimorphite_dl import DimorphiteDL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

root = os.path.dirname(os.path.abspath(__file__))
# use pH 7.4 https://git.durrantlab.pitt.edu/jdurrant/dimorphite_dl/
dimorphite = DimorphiteDL(min_ph=7.4, max_ph=7.4, pka_precision=0)


def standardize(smiles):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    try:
        mol = Chem.MolFromSmiles(smiles)
        # print(smiles)

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)
        # print(Chem.MolToSmiles(clean_mol))

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = (
            rdMolStandardize.Uncharger()
        )  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # print(uncharged_parent_clean_mol)

        protonated_smiles = dimorphite.protonate(
            Chem.MolToSmiles(uncharged_parent_clean_mol)
        )

        # print("protonated_smiles")

        if len(protonated_smiles) > 0:
            protonated_smile = protonated_smiles[0]

        protonated_mol = Chem.MolFromSmiles(protonated_smile)
        # protonated_mol= AddHs(protonated_mol)
        # protonated_smile = Chem.MolToSmiles(protonated_mol)

        # attempt is made at reionization at this step
        # at 7.4 pH

        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(protonated_mol)

        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)

    except:

        return "Cannot_do"


def calcdesc(data):
    # create descriptor calculator with all descriptors
    calc = Calculator(descriptors, ignore_3D=True)

    Ser_Mol = data["standardized_smiles"].apply(Chem.MolFromSmiles)
    Mordred_table = calc.pandas(Ser_Mol)
    Mordred_table = Mordred_table.astype("float")

    Morgan_fingerprint = Ser_Mol.apply(GetMorganFingerprintAsBitVect, args=(2, 2048))
    Morganfingerprint_array = np.stack(Morgan_fingerprint)

    Morgan_collection = [f"Mfp{x}" for x in range(Morganfingerprint_array.shape[1])]

    Morganfingerprint_table = pd.DataFrame(
        Morganfingerprint_array, columns=Morgan_collection
    )

    # Combine all features into one DataFrame
    result = pd.concat([data, Mordred_table, Morganfingerprint_table], axis=1)
    
    # Remove duplicate columns if any
    result = result.loc[:, ~result.columns.duplicated()]
    return result


def predict_individual_animal(data, endpoint, animal):
    # Load the feature list for the specific animal model
    with open(os.path.join(root, "..", "..", "checkpoints", "data",f"features_mfp_mordred_columns_{animal}_model.txt"), "r") as file:
        features = file.read().splitlines()

    # Load the pre-trained model for the given endpoint
    with open(os.path.join(root, "..", "..", "checkpoints", "data",f"log_{endpoint}_model_FINAL.sav"), "rb") as file:
        loaded_rf = pickle.load(file)

    # Select data based on the feature list
    X = data[features]

    # Replace missing descriptors with median

    animal_median = pd.read_csv(
        os.path.join(root, "..", "..", "checkpoints", "data",f"Median_mordred_values_{animal}_for_artificial_animal_data_mfp_mrd_model.csv")
    )

    for i in X.columns[X.isna().any()].tolist():
        X[i].fillna(float(animal_median[i]), inplace=True)

    # Load the scaler and apply it to the data
    with open(os.path.join(root, "..", "..", "checkpoints", "data",f"scaler_{animal}.pkl"), "rb") as file:
        scaler = pickle.load(file)

    # Scale the features and create a DataFrame with the scaled data
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    
    # Predict the target variable using the loaded model
    y_pred = loaded_rf.predict(X_scaled)
    
    return y_pred


def predict_animal(data):
    animal_endpoints = {
        "dog": ["dog_VDss_L_kg", "dog_CL_mL_min_kg", "dog_fup"],
        "monkey": ["monkey_VDss_L_kg", "monkey_CL_mL_min_kg", "monkey_fup"],
        "rat": ["rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"]
    }

    # Loop through each animal and its endpoints
    for animal, endpoints in animal_endpoints.items():
        for endpoint in endpoints:
            preds = predict_individual_animal(data, endpoint, animal)
            data[endpoint] = preds

    return data


def predict_VDss(data, features):  
    # Load the pre-trained random forest model
    with open(os.path.join(root, "..", "..", "checkpoints", "data","log_human_VDss_L_kg_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(root, "..", "..", "checkpoints", "data","artificial_animal_data_mfp_mrd_human_VDss_L_kg_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds


def predict_CL(data, features):

    # Load the pre-trained random forest model
    with open(os.path.join(root, "..", "..", "checkpoints", "data","log_human_CL_mL_min_kg_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(root, "..", "..", "checkpoints", "data","artificial_animal_data_mfp_mrd_human_CL_mL_min_kg_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)
    return y_preds


def predict_fup(data, features):
    # Load the pre-trained random forest model
    with open(os.path.join(root, "..", "..", "checkpoints", "data","log_human_fup_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(root, "..", "..", "checkpoints", "data","artificial_animal_data_mfp_mrd_human_fup_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds

def predict_MRT(data, features):
    # Load the pre-trained random forest model
    with open(os.path.join(root, "..", "..", "checkpoints", "data","log_human_mrt_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(root, "..", "..", "checkpoints", "data","artificial_animal_data_mfp_mrd_human_mrt_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds


def predict_thalf(data, features):
    # Load the pre-trained random forest model
    with open(os.path.join(root, "..", "..", "checkpoints", "data","log_human_thalf_withanimaldata_artificial_model_FINAL.sav"), "rb") as model_file:
        loaded_rf = pickle.load(model_file)

    # Extract and scale the feature data
    X = data[features].values

    # Load the scaler and apply it
    with open(os.path.join(root, "..", "..", "checkpoints", "data","artificial_animal_data_mfp_mrd_human_thalf_scaler.pkl"), "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    X_scaled = scaler.transform(X)

    # Convert the scaled data back to a DataFrame with the original feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)

    # Predict using the loaded model
    y_preds = loaded_rf.predict(X_scaled)

    return y_preds


def count(pred, true, min_val, max_val, endpoint):
    if endpoint == "human_fup":
        # Calculate the absolute ratios for human_fup case
        lst = [abs(a / b) for a, b in zip(pred, true)]
    else:
        # Calculate the absolute ratios for the other cases
        lst = [abs(10**a / 10**b) for a, b in zip(pred, true)]
    
    # Filter the list based on the given min and max values
    filtered_list = [x for x in lst if min_val <= x <= max_val]

    # Return the percentage of values within the range
    return (len(filtered_list) / len(lst)) * 100


def calc_gmfe(pred, true, endpoint):
    if endpoint == "human_fup":
        # Calculate log10 ratios for human_fup
        lst = [abs(np.log10(a / b)) for a, b in zip(pred, true)]
    else:
        # Calculate log10 ratios for the other cases
        lst = [abs(np.log10(10**a / 10**b)) for a, b in zip(pred, true)]
    
    # Calculate the mean of the absolute log differences and return GMFE
    mean_abs = np.mean(lst)
    return 10**mean_abs


def median_fold_change_error(pred, true, endpoint):
    if endpoint == "human_fup":
        # Calculate log10 ratios for human_fup
        lst = [abs(np.log10(a / b)) for a, b in zip(pred, true)]
    else:
        # Calculate log10 ratios for other cases
        lst = [abs(np.log10(10**a / 10**b)) for a, b in zip(pred, true)]
    
    # Compute the median of the absolute log differences
    median_abs = np.median(lst)
    
    # Return the fold change error using the natural logarithm base (e)
    return np.e**median_abs

def calc_bias(pred, true, endpoint):
    if endpoint == "human_fup":
        # Calculate differences for human_fup case
        lst = [(a - b) for a, b in zip(pred, true)]
    else:
        # Calculate differences for the other cases
        lst = [(10**a - 10**b) for a, b in zip(pred, true)]
    
    # Calculate and return the median of the differences
    bias = np.median(lst)
    return bias


animal_columns = [
    "dog_VDss_L_kg",
    "dog_CL_mL_min_kg",
    "dog_fup",
    "monkey_VDss_L_kg",
    "monkey_CL_mL_min_kg",
    "monkey_fup",
    "rat_VDss_L_kg",
    "rat_CL_mL_min_kg",
    "rat_fup",
]

def get_canonical_smiles(smiles_list, dataset_name):
    """
    Convert SMILES to canonical SMILES format, handling invalid SMILES with error logging.

    Parameters:
    - smiles_list (Series): List of SMILES strings.
    - dataset_name (str): The name of the dataset (for logging purposes).

    Returns:
    - list: A list of canonical SMILES.
    """
    canonical_smiles = []
    for smiles in smiles_list:
        try:
            canonical_smiles.append(Chem.CanonSmiles(smiles))
        except:
            print(f"Invalid SMILES in {dataset_name} dataset: {smiles}")
    return canonical_smiles


def calculate_similarity_test_vs_train(test, train):
    """
    Calculate Tanimoto similarity between test and train datasets based on their SMILES representations.

    Parameters:
    - test (DataFrame): Test dataset containing a 'smiles_r' column with SMILES strings.
    - train (DataFrame): Train dataset containing a 'smiles_r' column with SMILES strings.

    Returns:
    - DataFrame: A DataFrame with 'query' (test compounds), 'target' (train compounds), and 'MFP_Tc' (similarity scores).
    """

    # Convert SMILES to canonical SMILES format
    c_smiles_test = get_canonical_smiles(test['smiles_r'], "test")
    c_smiles_train = get_canonical_smiles(train['smiles_r'], "train")

    # Convert canonical SMILES to RDKit mol objects and generate fingerprints
    ms_test = [Chem.MolFromSmiles(smiles) for smiles in c_smiles_test]
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in ms_test]

    ms_train = [Chem.MolFromSmiles(smiles) for smiles in c_smiles_train]
    fps_train = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in ms_train]

    # Lists for query (test) compounds, target (train) compounds, and similarity scores
    query_list, target_list, similarity_list = [], [], []

    # Compare each test fingerprint against all train fingerprints and calculate Tanimoto similarity
    for i, fp_test in enumerate(fps_test):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_test, fps_train)
        query_smile = c_smiles_test[i]

        # Store query, target, and similarity score for each comparison
        for j, similarity in enumerate(similarities):
            query_list.append(query_smile)
            target_list.append(c_smiles_train[j])
            similarity_list.append(similarity)

    # Create DataFrame from the collected data
    similarity_df = pd.DataFrame({
        'query': query_list,
        'target': target_list,
        'MFP_Tc': similarity_list
    })

    return similarity_df

def avg_kNN_similarity(test_data, train_data_path=os.path.join(root, "..", "..", "checkpoints", "data","Train_data_log_transformed.csv"), n_neighbours=5):
    """
    Parameters:
    - test_data (DataFrame): Test data that will be compared against the training data.
    - train_data_path (str): Path to the CSV file containing the training data (default is '../Train_data_log_transformed.csv').
    - n_neighbours (int): Number of nearest neighbors to consider for similarity (default is 5).
    
    Returns:
    - DataFrame: A DataFrame with the mean similarity scores for each endpoint, rounded to 2 decimal places.
    """

    endpoints = ["human_VDss_L_kg", "human_CL_mL_min_kg", "human_fup", "human_mrt", "human_thalf"]

    # Load the training data
    train_data = pd.read_csv(train_data_path)

    # Create a DataFrame with all original test compounds
    all_smiles = pd.DataFrame({'query': test_data['smiles_r']})

    df_master = pd.DataFrame()

    for endpoint in endpoints:
        # Filter the training data for the current endpoint, removing rows with missing values
        df_filtered = train_data.dropna(subset=[endpoint]).reset_index(drop=True)
        
        # Calculate similarity between test and filtered training data
        df_similarity = calculate_similarity_test_vs_train(test_data, df_filtered)
        
        # Sort by similarity score (MFP_Tc) in descending order
        df_similarity_sorted = df_similarity.sort_values(['query', 'MFP_Tc'], ascending=[True, False]).reset_index(drop=True)
        
        # Select the top n_neighbours for each unique query (compound)
        df_top_neighbours = df_similarity_sorted.groupby('query').head(n_neighbours)

        # Group by query and calculate the mean of numeric values
        df_aggregated = df_top_neighbours.groupby('query').mean(numeric_only=True)
        
        # Merge with all_smiles to ensure all original compounds are included
        df_aggregated = all_smiles.merge(df_aggregated, on='query', how='left')
        
        # Assign the current endpoint to the results
        df_aggregated["endpoint"] = endpoint
        
        # Append the results to the master DataFrame
        df_master = pd.concat([df_master, df_aggregated])

    # Pivot the master DataFrame
    result_df = df_master.pivot_table(index='query', columns='endpoint', values='MFP_Tc').reset_index().round(2)

    # Ensure all original test compounds are included, even if they have no similarity scores
    result_df = all_smiles.merge(result_df, on='query', how='left')

    return result_df