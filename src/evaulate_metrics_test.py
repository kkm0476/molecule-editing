from fcd_torch import FCD
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd

target = 'CCc1cncc(C)c1' # your target SMILES
scaffold = 'O=C' # your scaffold SMILES
threshold = 0.0 # your threshold
result_path = '/content/molecule-editing/generated_smiles.txt' # absolute path for your samples

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