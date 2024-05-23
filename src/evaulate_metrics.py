from fcd_torch import FCD
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import pandas as pd

result_path = '/content/molecule-editing/generated_smiles/'
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
columns = ['fcd_0', 'fcd_15', 'fcd_30', 'fcd_100', 'div_0', 'div_15', 'div_30', 'div_100']

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
            intdiv = 1 - np.power(np.sum(np.power(similarity_matrix, p)) / (num_molecules ** 2), 1/p)\
            
            if file_name == '1_1_0':
                print('Samples      fcd     div')
            
            print(f'{file_name[:-4]:<8}  {fcd_t:05.3f}  {intdiv:05.4f}')
            fid_list.append(fcd_t)
            div_list.append(intdiv)
        data.append(fid_list + div_list)
        index.append(f'{target_idx+1}_{scaff_idx+1}')
        print()

df = pd.DataFrame(data, columns=columns, index=index)
print(df)
df.to_csv('/content/molecule-editing/metrics.csv')