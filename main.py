import streamlit as st
import joblib
import numpy as np

# Carrega o modelo
model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="Previsão de Risco de Crédito", layout="centered")
st.title("🔍 Previsão de Risco de Cliente - HELOC")

st.markdown("Preencha os dados do cliente para prever se há **risco de inadimplência**.")

# Entradas do usuário
def user_input_features():
    ExternalRiskEstimate = st.slider("External Risk Estimate", 0, 100, 50)
    MSinceOldestTradeOpen = st.slider("Meses desde a conta mais antiga", 0, 400, 100)
    MSinceMostRecentTradeOpen = st.slider("Meses desde a conta mais recente", 0, 400, 30)
    AverageMInFile = st.slider("Média de meses no arquivo", 0, 400, 150)
    NumSatisfactoryTrades = st.slider("Número de transações satisfatórias", 0, 100, 20)
    NumTrades60Ever2DerogPubRec = st.slider("Transações com atraso 60+ dias", 0, 20, 0)
    NumTrades90Ever2DerogPubRec = st.slider("Transações com atraso 90+ dias", 0, 20, 0)
    PercentTradesNeverDelq = st.slider("Percentual de transações sem inadimplência", 0, 100, 90)
    MSinceMostRecentDelq = st.slider("Meses desde última inadimplência", 0, 100, 12)
    MaxDelq2PublicRecLast12M = st.slider("Máx. atraso em 12 meses", 0, 9, 1)
    MaxDelqEver = st.slider("Máx. atraso já registrado", 0, 9, 2)
    NumTotalTrades = st.slider("Número total de transações", 0, 100, 30)
    NumTradesOpeninLast12M = st.slider("Transações abertas últimos 12 meses", 0, 20, 3)
    PercentInstallTrades = st.slider("Percentual de transações parceladas", 0, 100, 50)
    MSinceMostRecentInqexcl7days = st.slider("Meses desde última consulta (exceto últimos 7 dias)", 0, 24, 4)
    NumInqLast6M = st.slider("Consultas nos últimos 6 meses", 0, 10, 1)
    NumInqLast6Mexcl7days = st.slider("Consultas 6 meses (exceto últimos 7 dias)", 0, 10, 1)
    NetFractionRevolvingBurden = st.slider("Carga líquida de crédito rotativo", 0, 200, 60)
    NetFractionInstallBurden = st.slider("Carga líquida de crédito parcelado", 0, 200, 40)
    NumRevolvingTradesWBalance = st.slider("Transações rotativas com saldo", 0, 20, 5)
    NumInstallTradesWBalance = st.slider("Transações parceladas com saldo", 0, 20, 4)
    NumBank2NatlTradesWHighUtilization = st.slider("Utilização alta em contas bancárias", 0, 10, 1)
    PercentTradesWBalance = st.slider("Percentual de transações com saldo", 0, 100, 70)

    features = [
        ExternalRiskEstimate,
        MSinceOldestTradeOpen,
        MSinceMostRecentTradeOpen,
        AverageMInFile,
        NumSatisfactoryTrades,
        NumTrades60Ever2DerogPubRec,
        NumTrades90Ever2DerogPubRec,
        PercentTradesNeverDelq,
        MSinceMostRecentDelq,
        MaxDelq2PublicRecLast12M,
        MaxDelqEver,
        NumTotalTrades,
        NumTradesOpeninLast12M,
        PercentInstallTrades,
        MSinceMostRecentInqexcl7days,
        NumInqLast6M,
        NumInqLast6Mexcl7days,
        NetFractionRevolvingBurden,
        NetFractionInstallBurden,
        NumRevolvingTradesWBalance,
        NumInstallTradesWBalance,
        NumBank2NatlTradesWHighUtilization,
        PercentTradesWBalance
    ]
    return np.array(features).reshape(1, -1)

input_data = user_input_features()

# Botão de previsão
if st.button("🔮 Prever Risco"):
    prediction = model.predict(input_data)
    resultado = "⚠️ Cliente com **ALTO RISCO**" if prediction[0] == 1 else "✅ Cliente com **BAIXO RISCO**"
    st.subheader("Resultado da Previsão:")
    st.markdown(f"### {resultado}")
