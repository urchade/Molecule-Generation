import matplotlib.pyplot as plt
import seaborn as sns

path = r'data\smiles_train.txt'

molecules = []
with open(path, 'r') as file:
    for line in file:
        molecules.append(line.strip('\n'))

combined = ''.join(molecules)

# 0 for padding, 1 for <sos>, 2 for <eos>
id2char = dict(enumerate(set(combined), start=3))
id2char[1] = '<sos>'
id2char[2] = '<eos>'

char2id = {v: k for k, v in id2char.items()}

id_molecules = [[char2id[char] for char in molecule] for molecule in molecules]

for i in range(len(id_molecules)):
    id_molecules[i].append(2)  # Insert <eos> at the end
    id_molecules[i].insert(0, 1)  # Insert <sos> at the start

len_mol = []
for mol in id_molecules:
    len_mol.append(len(mol))

sns.distplot(len_mol)
plt.show()
