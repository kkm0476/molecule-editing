from fcd_torch import FCD
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process arguments for the script.")
    parser.add_argument("--target", type=int, help="SMILES string for the target molecule")
    parser.add_argument("--scaffold", type=int, help="SMILES string for the scaffold molecule")
    parser.add_argument("--threshold", type=int, help="Threshold value")
    parser.add_argument("--result_path", type=str, help="Absolute path for the exported smiles file")
    return parser.parse_args()

# Parse command-line arguments
args = parse_arguments()

targets = ['CCC1COCC1C', # exported as "1_%d_%d.txt"
          'CCc1cncc(C)c1',
          'C#CCC1CC=CCO1',
          'c1cnn2cccc2c1',
          'c1ccoccoc1',
          'C#CC(C=O)CC#N',
          'C#CC(C#N)NCC#N',
          'N=CNC=CN=NC=O',
          'CCCC=CC(C)C',
          'CCCNC(=O)CCC'
          ]
scaffolds = ['c1ccncc1',
             'O=C', # exported as "%d_2_%d.txt"
             'N=O']
thresholds = [0,
              0.15,
              0.30, # exported as %d_%d_30.txt"
              1.00]

# Assign values from command-line arguments to variables
target = targets[args.target]
scaffold = scaffolds[args.scaffold]
threshold = thresholds[args.threshold]
result_path = args.result_path

print("Evaluation for ")
print("Target :", target, "Scaffold :", scaffold, "Threshold :", threshold, "Generated_smiles :", result_path)

# target = 'CCc1cncc(C)c1' # your target SMILES
# scaffold = 'O=C' # your scaffold SMILES
# threshold = 0.0 # your thresholdß
# result_path = '/content/molecule-editing/generated_smiles.txt' # absolute path for your samples

fcd = FCD(device='cuda:0', n_jobs=8)

data = []
index = []
columns = ['fid_0', 'fid_15', 'fid_30', 'fid_100', 'div_0', 'div_15', 'div_30', 'div_100']


smiles_list = []

f = open(result_path, 'r')
while True:
    smiles = f.readline().strip()
    if not smiles:
        break
    if smiles != 'None':
        smiles_list.append(smiles)
f.close()

# evaluate target fidelity
fcd_score = fcd([target, target], smiles_list)
fcd_t = (100. - fcd_score) / 100.

# evaluate IntDiv
molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]
num_molecules = len(fingerprints)
similarity_matrix = np.zeros((num_molecules, num_molecules))
for i in range(num_molecules):
    for j in range(num_molecules):
        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        similarity_matrix[i, j] = similarity
p = 1
intdiv = 1 - np.power(np.sum(np.power(similarity_matrix, p)) / (num_molecules ** 2), 1/p)

print(f'Target: {target}  /  Scaffold: {scaffold}  / threshold: {threshold:3.2f}')
print(f'Target Fidelity:  {fcd_t:4.3f}')
print(f'Sample Diversity: {intdiv:4.3f}')