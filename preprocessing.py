import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Substitui valores inválidos (-9) por NaN
    df = df.replace(-9, pd.NA)

    # Remove colunas com muitos valores faltantes (opcional)
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))

    # Preenche valores faltantes com a mediana de cada coluna
    df = df.fillna(df.median(numeric_only=True))

    # Separar features (X) e alvo (y)
    X = df.drop("is_at_risk", axis=1)
    y = df["is_at_risk"]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
