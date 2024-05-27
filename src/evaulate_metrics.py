from fcd_torch import FCD
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd

result_path = '/content/molecule-editing/generated_smiles_sample/'
targets = ['CCC1COCC1C',
    'CCc1cncc(C)c1',
    'C#CCC1CC=CCO1',
    'c1cnn2cccc2c1',
    'c1ccoccoc1',
    'C#CC(C=O)CC#N',
    'C#CC(C#N)NCC#N',
    'N=CNC=CN=NC=O',
    'CCCC=CC(C)C',
    'CCCNC(=O)CCC']
scaffolds = ['c1ccncc1', 'O=C', 'N=O']

fcd = FCD(device='cuda:0', n_jobs=8)

data = []
index = []
columns = ['fid_0', 'fid_15', 'fid_30', 'fid_100', 'div_0', 'div_15', 'div_30', 'div_100']

for target_idx, target in enumerate(targets):
    for scaff_idx, scaffold in enumerate(scaffolds):
        fid_list = []
        div_list = []
        for t in [0, 15, 30, 100]:
            smiles_list = []
            file_name = f'{target_idx+1}_{scaff_idx+1}_{t}.txt'
            f = open(result_path + file_name, 'r')
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
            fid_list.append(fcd_t)
            div_list.append(intdiv)
        data.append(fid_list + div_list)
        index.append(f'{target_idx+1}_{scaff_idx+1}')
        
        if target_idx + scaff_idx == 0:
            print('file    fid_0   fid_15   fid_30  fid_100    div_0   div_15   div_30  div_100')
        
        print(f'{(target_idx+1):02d}_{scaff_idx+1}', end='')
        for score in (fid_list + div_list):
            print(f'    {score:>4.3f}', end='')
        print()

df = pd.DataFrame(data, columns=columns, index=index)
df.to_csv('/content/molecule-editing/metrics.csv')