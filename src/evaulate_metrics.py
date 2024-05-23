import os
from fcd_score import FCD
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

result_path = '/content/molecule-editing/results/'
targets = [
    'CCC1COCC1C',
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
scaffolds = ['c1ccncc1', 'O=C', 'N=O']
target_decoder = {(i+1):target for i, target in enumerate(targets)}
scaff_decoder = {(i+1):scaff for i, scaff in enumerate(scaffolds)}

fcd = FCD(device='cuda:0', n_jobs=8)
file_list = os.listdir(result_path)

for file in file_list:
    if file.endswith('.txt'):
        target_idx, scaff_idx, threshold = map(int, file.split('_'))
        target = target_decoder[target_idx] # target SMILES
        scaffold = scaff_decoder[scaff_idx] # scaffold SMILES
        smiles_list = []
        
        f = open(result_path + file, 'r')
        while True:
            smiles = f.readline()
            if not smiles:
                break
            if smiles != 'None':
                smiles_list.append(smiles)
        f.close()
        
        fcd_score = fcd([target, target], smiles_list)

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

        print(f'{target} + {scaffold} with threshold {float(threshold)/100.}')
        print(f'FCD : {fcd_score}')
        print(f'IntDiv : {intdiv}\n')