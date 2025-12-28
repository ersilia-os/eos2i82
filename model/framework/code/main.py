# imports
import os
import csv
import sys

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)

from run_batch import run_pksmart

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
outputs = run_pksmart(smiles_list)

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

outputs.drop(columns = ["smiles_r"], inplace=True)

outputs.columns = outputs.columns.str.strip().str.lower()
print(outputs.columns)

outputs.to_csv(output_file, index=False)
