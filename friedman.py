import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt

base_path = "C://Users//admin/Desktop//experimentos_fluo//7 - melanoma x nev_sek"

modelos = ["SVM", "KNN", "MLP"]
arquiteturas = sorted(os.listdir(base_path))

colunas = [] 
folds_data = [[] for _ in range(5)]


for arq in arquiteturas:
    arq_path = os.path.join(base_path, arq, "resultados_CLASSIFICADORES")
    
    for modelo in modelos:
        nome_coluna = f"{modelo}_{arq}"
        colunas.append(nome_coluna)

        file_name = f"resultados_folds_{modelo}.csv"
        csv_path = os.path.join(arq_path, file_name)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            aucs = df["AUC"].values

            if len(aucs) != 5:
                raise ValueError(f"A coluna 'AUC' em {csv_path} não tem 5 valores (folds)")

            for i in range(5):
                folds_data[i].append(aucs[i])
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

dados = pd.DataFrame(folds_data, columns=colunas)

stat, p = friedmanchisquare(*[dados[col] for col in dados.columns])

print(f"\nEstatística de Friedman: {stat:.4f}")
print(f"Valor-p: {p:.4f}")
if p < 0.05:
    print("✅ Há diferença estatística significativa entre os métodos (p < 0.05).")
else:
    print("❌ Não há diferença estatística significativa entre os métodos (p >= 0.05).")

nemenyi = sp.posthoc_nemenyi_friedman(dados.values)
nemenyi.index = dados.columns
nemenyi.columns = dados.columns
print("\nResultado do Nemenyi:\n", nemenyi.round(4))

plt.figure(figsize=(12, 3))
mean_ranks = dados.rank(axis=1, ascending=False).mean()
sp.critical_difference_diagram(mean_ranks, nemenyi)
plt.title("Critical Difference Diagram (AUC por Fold)")
plt.tight_layout()
plt.show()