import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    # Mostrar estatísticas básicas
    print("✅ Estatísticas gerais:\n", df.describe())
    print("\n✅ Dados ausentes:\n", df.isnull().sum())

    # Distribuição da variável alvo
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="is_at_risk", palette="Set2")
    plt.title("Distribuição da Variável Alvo")
    plt.xlabel("Está em risco? (0 = Não, 1 = Sim)")
    plt.ylabel("Contagem")
    plt.show()

    # Mapa de correlação
    plt.figure(figsize=(14,10))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
    plt.title("Mapa de Correlação entre Variáveis")
    plt.show()
