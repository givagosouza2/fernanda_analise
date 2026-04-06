import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import itertools
import matplotlib.pyplot as plt

st.set_page_config(page_title="ANOVA de Medidas Repetidas", layout="wide")

st.title("ANOVA de uma via com medidas repetidas")
st.write(
    "Carregue um arquivo CSV com cabeçalhos e selecione 3 colunas numéricas. "
    "Cada linha será tratada como um participante, e cada coluna como uma condição."
)

uploaded_file = st.file_uploader("Carregue o arquivo CSV", type=["csv"])

def format_p_value(p):
    if pd.isna(p):
        return ""
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 3:
        st.error("O arquivo precisa ter pelo menos 3 colunas numéricas.")
        st.stop()

    st.subheader("Seleção das colunas")
    selected_cols = st.multiselect(
        "Escolha exatamente 3 colunas numéricas",
        options=numeric_cols,
        default=numeric_cols[:3]
    )

    alpha = st.number_input("Nível de significância (alpha)", min_value=0.001, max_value=0.2, value=0.05, step=0.001)

    correction_method = st.selectbox(
        "Correção para comparações múltiplas",
        options=["bonferroni", "holm", "fdr_bh"],
        index=0
    )

    if len(selected_cols) != 3:
        st.warning("Selecione exatamente 3 colunas.")
        st.stop()

    data = df[selected_cols].copy()

    st.subheader("Tratamento de dados ausentes")
    n_before = len(data)
    data = data.dropna()
    n_after = len(data)

    st.write(f"Linhas originais: **{n_before}**")
    st.write(f"Linhas após remover valores ausentes nas colunas selecionadas: **{n_after}**")

    if n_after < 2:
        st.error("Não há dados suficientes após remover valores ausentes.")
        st.stop()

    # Criação de identificador de sujeito
    data = data.reset_index(drop=True)
    data["Sujeito"] = np.arange(1, len(data) + 1)

    # Formato longo
    long_df = data.melt(
        id_vars="Sujeito",
        value_vars=selected_cols,
        var_name="Condicao",
        value_name="Valor"
    )

    st.subheader("Estatísticas descritivas")
    desc = pd.DataFrame({
        "Condição": selected_cols,
        "Média": [data[col].mean() for col in selected_cols],
        "Desvio-padrão": [data[col].std(ddof=1) for col in selected_cols],
        "N": [data[col].count() for col in selected_cols]
    })
    st.dataframe(desc.style.format({"Média": "{:.4f}", "Desvio-padrão": "{:.4f}"}))

    # ANOVA de medidas repetidas
    st.subheader("ANOVA de uma via para medidas repetidas")
    try:
        anova = AnovaRM(
            data=long_df,
            depvar="Valor",
            subject="Sujeito",
            within=["Condicao"]
        ).fit()

        anova_table = anova.anova_table.reset_index().rename(columns={"index": "Efeito"})
        st.dataframe(anova_table)

        # Extração dos valores principais para texto de interpretação
        f_value = anova_table.loc[0, "F Value"]
        num_df = anova_table.loc[0, "Num DF"]
        den_df = anova_table.loc[0, "Den DF"]
        p_value = anova_table.loc[0, "Pr > F"]

        st.markdown(
            f"""
**Resultado:** F({num_df:.0f}, {den_df:.0f}) = {f_value:.4f}, p = {format_p_value(p_value)}
"""
        )

        if p_value < alpha:
            st.success("Há diferença significativa entre as condições.")
        else:
            st.info("Não foi encontrada diferença significativa entre as condições.")

    except Exception as e:
        st.error(f"Erro ao executar a ANOVA de medidas repetidas: {e}")
        st.stop()

    # Comparações múltiplas pareadas
    st.subheader("Comparações múltiplas")
    pair_results = []

    for c1, c2 in itertools.combinations(selected_cols, 2):
        t_stat, p_raw = stats.ttest_rel(data[c1], data[c2], nan_policy="omit")

        # Effect size para medidas pareadas: dz = média das diferenças / dp das diferenças
        diff = data[c1] - data[c2]
        sd_diff = diff.std(ddof=1)
        if sd_diff == 0:
            dz = np.nan
        else:
            dz = diff.mean() / sd_diff

        pair_results.append({
            "Comparação": f"{c1} vs {c2}",
            "t": t_stat,
            "p_bruto": p_raw,
            "dz": dz
        })

    pairwise_df = pd.DataFrame(pair_results)

    reject, p_corr, _, _ = multipletests(pairwise_df["p_bruto"], alpha=alpha, method=correction_method)
    pairwise_df["p_corrigido"] = p_corr
    pairwise_df["Significativo"] = reject

    st.dataframe(
        pairwise_df.style.format({
            "t": "{:.4f}",
            "p_bruto": "{:.4f}",
            "p_corrigido": "{:.4f}",
            "dz": "{:.4f}",
        })
    )

    st.markdown(
        """
**Legenda:**
- **t**: teste t pareado
- **p_bruto**: valor de p sem correção
- **p_corrigido**: valor de p após correção para múltiplas comparações
- **dz**: tamanho de efeito para comparação pareada
"""
    )

    # Gráfico
    st.subheader("Visualização")
    fig, ax = plt.subplots(figsize=(8, 5))

    means = [data[col].mean() for col in selected_cols]
    sems = [stats.sem(data[col], nan_policy="omit") for col in selected_cols]

    ax.bar(selected_cols, means, yerr=sems, capsize=5)
    ax.set_ylabel("Valor")
    ax.set_title("Média ± EPM por condição")

    st.pyplot(fig)

    # Dados em formato longo para inspeção
    with st.expander("Ver dados em formato longo"):
        st.dataframe(long_df)

else:
    st.info("Carregue um arquivo CSV para começar.")
