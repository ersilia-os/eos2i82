import numpy as np
import os
import pandas as pd
from pksmart_streamlit import (
                     standardize, 
                     calcdesc, 
                     predict_animal, 
                     predict_VDss, 
                     predict_CL, 
                     predict_fup, 
                     predict_MRT, 
                     predict_thalf)

root = os.path.dirname(os.path.abspath(__file__))

def run_pksmart(smiles_list):
    # Create an empty DataFrame to hold the SMILES and predictions
    data = pd.DataFrame(smiles_list, columns=['smiles_r'])

    # Standardize and calculate descriptors for the input molecules
    data['standardized_smiles'] = data['smiles_r'].apply(standardize)
    
    valid_mask = data["standardized_smiles"] != "Cannot_do"
    valid_idx = data.index[valid_mask]
    valid_data = data.loc[valid_idx].copy()
    
    animal_columns = [
        "dog_VDss_L_kg", "dog_CL_mL_min_kg", "dog_fup",
        "monkey_VDss_L_kg", "monkey_CL_mL_min_kg", "monkey_fup",
        "rat_VDss_L_kg", "rat_CL_mL_min_kg", "rat_fup"
    ]
    human_columns = ["VDss_L_kg", "CL_mL_min_kg", "fup", "MRT_hr", "thalf_hr"]

    out = data[["smiles_r"]].copy()
    for c in human_columns + animal_columns:
        out[c] = np.nan
    if valid_data.empty:
        return out
    
    data_mordred = calcdesc(valid_data)

    # Run predictions for animal models
    animal_predictions = predict_animal(data_mordred)
    # Create a copy of animal_predictions to avoid modifying the original
    display_predictions = animal_predictions.copy()
    for key in animal_columns:
        if not key.endswith("_fup"):
            display_predictions[key] = 10**display_predictions[key]
    display_predictions = display_predictions[animal_columns]

    # human_columns = ['VDss', 'CL', 'fup', 'MRT', 'thalf']
    human_predictions = pd.DataFrame()
    with open(os.path.join(root, "..", "..", "checkpoints", "data","features_mfp_mordred_animal_artificial_human_modelcolumns.txt")) as f:
        model_features = f.read().splitlines()

    human_predictions['smiles_r'] = data_mordred['smiles_r']
    human_predictions['VDss_L_kg'] = 10**predict_VDss(data_mordred, model_features)
    human_predictions['CL_mL_min_kg'] = 10**predict_CL(data_mordred, model_features)
    human_predictions['fup'] = predict_fup(data_mordred, model_features)
    human_predictions['MRT_hr'] = 10**predict_MRT(data_mordred, model_features)
    human_predictions['thalf_hr'] = 10**predict_thalf(data_mordred, model_features)

    for c in animal_columns:
        out.loc[valid_idx, c] = display_predictions[c].values

    for c in human_columns:
        out.loc[valid_idx, c] = human_predictions[c].values
    
    return out

# Values and definitions
definitions = {
    # Animal Parameters
    "dog_VDss_L_kg": "Volume of distribution at steady state for dog (L/kg)",
    "dog_CL_mL_min_kg": "Clearance rate for dog (mL/min/kg)",
    "dog_fup": "Fraction unbound in plasma for dog",
    "monkey_VDss_L_kg": "Volume of distribution at steady state for monkey (L/kg)",
    "monkey_CL_mL_min_kg": "Clearance rate for monkey (mL/min/kg)",
    "monkey_fup": "Fraction unbound in plasma for monkey",
    "rat_VDss_L_kg": "Volume of distribution at steady state for rat (L/kg)",
    "rat_CL_mL_min_kg": "Clearance rate for rat (mL/min/kg)",
    "rat_fup": "Fraction unbound in plasma for rat",
    
    # Human Parameters
    "CL_fe": "Fold error for human clearance (CL)",
    "CL": "Predicted value for human clearance (CL, mL/min/kg)",
    "CL_min": "Minimum predicted value for human clearance (CL, mL/min/kg)",
    "CL_max": "Maximum predicted value for human clearance (CL, mL/min/kg)",
    "Vd_fe": "Fold error for human volume of distribution (VDss)",
    "VDss": "Predicted value for human volume of distribution (VDss, L/kg)",
    "Vd_min": "Minimum predicted value for human volume of distribution (VDss, L/kg)",
    "Vd_max": "Maximum predicted value for human volume of distribution (VDss, L/kg)",
    "MRT_fe": "Fold error for human mean residence time (MRT)",
    "MRT": "Predicted value for human mean residence time (MRT, min)",
    "MRT_min": "Minimum predicted value for human mean residence time (MRT, min)",
    "MRT_max": "Maximum predicted value for human mean residence time (MRT, min)",
    "thalf_fe": "Fold error for human half-life (t1/2)",
    "thalf": "Predicted value for human half-life (t1/2, min)",
    "thalf_min": "Minimum predicted value for human half-life (t1/2, min)",
    "thalf_max": "Maximum predicted value for human half-life (t1/2, min)",
    "fup_fe": "Fold error for human fraction unbound in plasma (fup)",
    "fup": "Predicted value for human fraction unbound in plasma (fup)",
    "fup_min": "Minimum predicted value for human fraction unbound in plasma (fup)",
    "fup_max": "Maximum predicted value for human fraction unbound in plasma (fup)"
}
