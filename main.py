import io
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="TUG - Medidas Repetidas", layout="wide")

st.title("Análise de TUG simples e dupla tarefa")
st.write(
    """
Este aplicativo compara três condições repetidas do Timed Up and Go (TUG):
1. **TUG simples**
2. **TUG com tarefa cognitiva numérica**
3. **TUG com tarefa cognitiva de nomeação de palavras**

Cada linha do arquivo deve corresponder ao mesmo participante nas três condições.
"""
)

# ---------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------
def format_p(p):
    if pd.isna(p):
        return ""
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"

def paired_cohens_d(x, y):
    diff = np.asarray(x) - np.asarray(y)
    sd = np.std(diff, ddof=1)
    if np.isclose(sd, 0):
        return np.nan
    return np.mean(diff) / sd

def eta_p_from_f(F, df1, df2):
    return (F * df1) / (F * df1 + df2)

def rank_biserial_from_wilcoxon(x, y):
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[~np.isnan(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    abs_diff = np.abs(diff)
    ranks = stats.rankdata(abs_diff)
    pos = np.sum(ranks[diff > 0])
    neg = np.sum(ranks[diff < 0])
    return (pos - neg) / (pos + neg)

def confidence_interval_mean(diff, alpha=0.05):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return np.nan, np.nan
    mean_diff = np.mean(diff)
    se = stats.sem(diff, nan_policy="omit")
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    lower = mean_diff - t_crit * se
    upper = mean_diff + t_crit * se
    return lower, upper

def normality_of_differences(df, cols):
    rows = []
    for a, b in itertools.combinations(cols, 2):
        diff = df[a] - df[b]
        diff = diff.dropna()
        if len(diff) >= 3:
            if len(diff) <= 5000:
                W, p = stats.shapiro(diff)
            else:
                W, p = np.nan, np.nan
        else:
            W, p = np.nan, np.nan
        rows.append({
            "Diferença": f"{a} - {b}",
            "Shapiro-W": W,
            "p": p
        })
    return pd.DataFrame(rows)

def make_long(df_wide, id_col, cols):
    return df_wide.melt(
        id_vars=id_col,
        value_vars=cols,
        var_name="Condicao",
        value_name="Valor"
    )

def friedman_test(df, cols):
    x1, x2, x3 = [df[c].values for c in cols]
    stat, p = stats.friedmanchisquare(x1, x2, x3)
    n = len(df)
    k = len(cols)
    kendalls_w = stat / (n * (k - 1)) if n > 0 else np.nan
    return stat, p, kendalls_w

def pairwise_parametric(df, cols, alpha=0.05, method="holm"):
    results = []
    for a, b in itertools.combinations(cols, 2):
        t, p = stats.ttest_rel(df[a], df[b], nan_policy="omit")
        diff = df[a] - df[b]
        mean_diff = diff.mean()
        ci_low, ci_high = confidence_interval_mean(diff, alpha=alpha)
        dz = paired_cohens_d(df[a], df[b])
        results.append({
            "Comparação": f"{a} vs {b}",
            "Teste": "t pareado",
            "Estatística": t,
            "p_bruto": p,
            "Dif_média": mean_diff,
            "IC95%_inf": ci_low,
            "IC95%_sup": ci_high,
            "Effect_size_dz": dz
        })
    out = pd.DataFrame(results)
    reject, p_corr, _, _ = multipletests(out["p_bruto"], alpha=alpha, method=method)
    out["p_corrigido"] = p_corr
    out["Significativo"] = reject
    return out

def pairwise_nonparametric(df, cols, alpha=0.05, method="holm"):
    results = []
    for a, b in itertools.combinations(cols, 2):
        try:
            w, p = stats.wilcoxon(df[a], df[b], zero_method="wilcox", correction=False)
        except Exception:
            w, p = np.nan, np.nan
        diff = df[a] - df[b]
        median_diff = np.median(diff)
        rbc = rank_biserial_from_wilcoxon(df[a], df[b])
        results.append({
            "Comparação": f"{a} vs {b}",
            "Teste": "Wilcoxon pareado",
            "Estatística": w,
            "p_bruto": p,
            "Dif_mediana_aprox": median_diff,
            "Rank_biserial": rbc
        })
    out = pd.DataFrame(results)
    reject, p_corr, _, _ = multipletests(out["p_bruto"], alpha=alpha, method=method)
    out["p_corrigido"] = p_corr
    out["Significativo"] = reject
    return out

def descriptive_stats(df, cols):
    desc = []
    for c in cols:
        desc.append({
            "Condição": c,
            "Média": df[c].mean(),
            "DP": df[c].std(ddof=1),
            "Mediana": df[c].median(),
            "Q1": df[c].quantile(0.25),
            "Q3": df[c].quantile(0.75),
            "Mínimo": df[c].min(),
            "Máximo": df[c].max(),
            "N": df[c].count()
        })
    return pd.DataFrame(desc)

def dtc_table(df, single, dual_num, dual_word):
    out = pd.DataFrame(index=df.index.copy())
    out["DTC_num_%"] = ((df[dual_num] - df[single]) / df[single]) * 100
    out["DTC_palavras_%"] = ((df[dual_word] - df[single]) / df[single]) * 100
    return out

def summarize_findings(anova_p, friedman_p, posthoc_df, cols):
    txt = []
    if pd.notna(anova_p):
        if anova_p < 0.05:
            txt.append("A ANOVA de medidas repetidas indicou diferença global entre as três condições.")
        else:
            txt.append("A ANOVA de medidas repetidas não indicou diferença global significativa entre as três condições.")
    if pd.notna(friedman_p):
        if friedman_p < 0.05:
            txt.append("O teste de Friedman também sugeriu diferença entre as condições.")
        else:
            txt.append("O teste de Friedman não sugeriu diferença significativa entre as condições.")
    if posthoc_df is not None and not posthoc_df.empty:
        sig = posthoc_df.loc[posthoc_df["Significativo"] == True, "Comparação"].tolist()
        if sig:
            txt.append("Nas comparações múltiplas corrigidas, houve diferença em: " + "; ".join(sig) + ".")
        else:
            txt.append("Nas comparações múltiplas corrigidas, nenhuma comparação pareada permaneceu significativa.")
    txt.append(
        f"Do ponto de vista clínico, recomenda-se interpretar conjuntamente a diferença global, "
        f"as comparações pareadas e o custo da dupla tarefa entre {cols[0]} e as demais condições."
    )
    return " ".join(txt)

# ---------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------
example_expander = st.expander("Exemplo do formato esperado do CSV")
with example_expander:
    st.code(
        "TUG_simples,TUG_numérico,TUG_palavras\n"
        "9.80,11.10,10.90\n"
        "8.95,10.02,9.85\n"
        "11.20,12.45,12.10\n"
    )

uploaded = st.file_uploader("Carregue um arquivo CSV", type=["csv"])

if uploaded is None:
    st.info("Carregue um arquivo CSV para iniciar.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Erro ao ler o CSV: {e}")
    st.stop()

st.subheader("Pré-visualização")
st.dataframe(df.head(), use_container_width=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 3:
    st.error("O arquivo precisa conter pelo menos três colunas numéricas.")
    st.stop()

st.subheader("Configuração")
colA, colB, colC = st.columns(3)
with colA:
    single_col = st.selectbox("Condição 1: TUG simples", options=numeric_cols, index=0)
with colB:
    dual_num_col = st.selectbox("Condição 2: dupla tarefa numérica", options=numeric_cols, index=min(1, len(numeric_cols)-1))
with colC:
    dual_word_col = st.selectbox("Condição 3: dupla tarefa com palavras", options=numeric_cols, index=min(2, len(numeric_cols)-1))

if len({single_col, dual_num_col, dual_word_col}) < 3:
    st.warning("Escolha três colunas diferentes.")
    st.stop()

alpha = st.number_input("Nível de significância (alpha)", min_value=0.001, max_value=0.20, value=0.05, step=0.001)
correction = st.selectbox("Correção para comparações múltiplas", ["holm", "bonferroni", "fdr_bh"], index=0)

cols = [single_col, dual_num_col, dual_word_col]
work = df[cols].copy()
n0 = len(work)
work = work.dropna().reset_index(drop=True)
work["Sujeito"] = np.arange(1, len(work) + 1)

st.write(f"Linhas originais: **{n0}**")
st.write(f"Linhas completas após remoção de ausentes: **{len(work)}**")

if len(work) < 3:
    st.error("São necessárias pelo menos 3 linhas completas para prosseguir.")
    st.stop()

# ---------------------------------------------------------------------
# Estatísticas descritivas
# ---------------------------------------------------------------------
st.subheader("1) Estatísticas descritivas")
desc = descriptive_stats(work, cols)
st.dataframe(
    desc.style.format({
        "Média": "{:.4f}", "DP": "{:.4f}", "Mediana": "{:.4f}",
        "Q1": "{:.4f}", "Q3": "{:.4f}", "Mínimo": "{:.4f}", "Máximo": "{:.4f}"
    }),
    use_container_width=True
)

# ---------------------------------------------------------------------
# Long format
# ---------------------------------------------------------------------
long_df = make_long(work, "Sujeito", cols)

# ---------------------------------------------------------------------
# Verificação de normalidade das diferenças
# ---------------------------------------------------------------------
st.subheader("2) Normalidade das diferenças entre pares (Shapiro-Wilk)")
norm_df = normality_of_differences(work, cols)
st.dataframe(norm_df.style.format({"Shapiro-W": "{:.4f}", "p": "{:.4f}"}), use_container_width=True)
st.caption("Em medidas repetidas, é mais útil verificar a normalidade das diferenças entre condições do que a normalidade bruta de cada coluna.")

# ---------------------------------------------------------------------
# ANOVA de medidas repetidas
# ---------------------------------------------------------------------
st.subheader("3) ANOVA de uma via para medidas repetidas")
anova_p = np.nan
try:
    anova = AnovaRM(data=long_df, depvar="Valor", subject="Sujeito", within=["Condicao"]).fit()
    anova_table = anova.anova_table.reset_index().rename(columns={"index": "Efeito"})
    st.dataframe(anova_table, use_container_width=True)

    F = float(anova_table.loc[0, "F Value"])
    df1 = float(anova_table.loc[0, "Num DF"])
    df2 = float(anova_table.loc[0, "Den DF"])
    anova_p = float(anova_table.loc[0, "Pr > F"])
    eta_p = eta_p_from_f(F, df1, df2)

    st.markdown(
        f"**Resultado:** F({df1:.0f}, {df2:.0f}) = {F:.4f}, p = {format_p(anova_p)}, "
        f"eta parcial² = {eta_p:.4f}"
    )
except Exception as e:
    st.error(f"Não foi possível calcular a ANOVA de medidas repetidas: {e}")

# ---------------------------------------------------------------------
# Friedman
# ---------------------------------------------------------------------
st.subheader("4) Teste de Friedman (alternativa não paramétrica)")
friedman_p = np.nan
try:
    friedman_stat, friedman_p, kendalls_w = friedman_test(work, cols)
    friedman_df = pd.DataFrame([{
        "Qui-quadrado de Friedman": friedman_stat,
        "p": friedman_p,
        "Kendall_W": kendalls_w
    }])
    st.dataframe(
        friedman_df.style.format({
            "Qui-quadrado de Friedman": "{:.4f}",
            "p": "{:.4f}",
            "Kendall_W": "{:.4f}",
        }),
        use_container_width=True
    )
except Exception as e:
    st.error(f"Não foi possível calcular o teste de Friedman: {e}")

# ---------------------------------------------------------------------
# Comparações múltiplas paramétricas
# ---------------------------------------------------------------------
st.subheader("5) Comparações múltiplas paramétricas")
param_posthoc = pairwise_parametric(work, cols, alpha=alpha, method=correction)
st.dataframe(
    param_posthoc.style.format({
        "Estatística": "{:.4f}",
        "p_bruto": "{:.4f}",
        "Dif_média": "{:.4f}",
        "IC95%_inf": "{:.4f}",
        "IC95%_sup": "{:.4f}",
        "Effect_size_dz": "{:.4f}",
        "p_corrigido": "{:.4f}",
    }),
    use_container_width=True
)

# ---------------------------------------------------------------------
# Comparações múltiplas não paramétricas
# ---------------------------------------------------------------------
st.subheader("6) Comparações múltiplas não paramétricas")
nonparam_posthoc = pairwise_nonparametric(work, cols, alpha=alpha, method=correction)
st.dataframe(
    nonparam_posthoc.style.format({
        "Estatística": "{:.4f}",
        "p_bruto": "{:.4f}",
        "Dif_mediana_aprox": "{:.4f}",
        "Rank_biserial": "{:.4f}",
        "p_corrigido": "{:.4f}",
    }),
    use_container_width=True
)

# ---------------------------------------------------------------------
# Custo da dupla tarefa
# ---------------------------------------------------------------------
st.subheader("7) Custo da dupla tarefa (Dual-task cost, %)")
dtc = dtc_table(work, single_col, dual_num_col, dual_word_col)

dtc_desc = pd.DataFrame({
    "Métrica": ["DTC numérico (%)", "DTC palavras (%)"],
    "Média": [dtc["DTC_num_%"].mean(), dtc["DTC_palavras_%"].mean()],
    "DP": [dtc["DTC_num_%"].std(ddof=1), dtc["DTC_palavras_%"].std(ddof=1)],
    "Mediana": [dtc["DTC_num_%"].median(), dtc["DTC_palavras_%"].median()]
})
st.dataframe(dtc_desc.style.format({"Média": "{:.4f}", "DP": "{:.4f}", "Mediana": "{:.4f}"}), use_container_width=True)

# Testes nos DTCs
col1, col2 = st.columns(2)
with col1:
    t_dtc, p_dtc = stats.ttest_rel(dtc["DTC_num_%"], dtc["DTC_palavras_%"], nan_policy="omit")
    dz_dtc = paired_cohens_d(dtc["DTC_num_%"], dtc["DTC_palavras_%"])
    st.markdown(
        f"""
**Comparação entre custos de dupla tarefa**
- t pareado = {t_dtc:.4f}
- p = {format_p(p_dtc)}
- dz = {dz_dtc:.4f}
"""
    )
with col2:
    t0_num, p0_num = stats.ttest_1samp(dtc["DTC_num_%"].dropna(), popmean=0.0)
    t0_word, p0_word = stats.ttest_1samp(dtc["DTC_palavras_%"].dropna(), popmean=0.0)
    st.markdown(
        f"""
**Teste do custo contra zero**
- DTC numérico: t = {t0_num:.4f}, p = {format_p(p0_num)}
- DTC palavras: t = {t0_word:.4f}, p = {format_p(p0_word)}
"""
    )

with st.expander("Ver tabela individual de DTC"):
    dtc_show = pd.concat([work[["Sujeito", single_col, dual_num_col, dual_word_col]], dtc], axis=1)
    st.dataframe(dtc_show, use_container_width=True)

# ---------------------------------------------------------------------
# Correlações
# ---------------------------------------------------------------------
st.subheader("8) Correlações entre condições")
corr_rows = []
for a, b in itertools.combinations(cols, 2):
    r_p, p_p = stats.pearsonr(work[a], work[b])
    r_s, p_s = stats.spearmanr(work[a], work[b])
    corr_rows.append({
        "Comparação": f"{a} vs {b}",
        "Pearson_r": r_p,
        "Pearson_p": p_p,
        "Spearman_rho": r_s,
        "Spearman_p": p_s
    })
corr_df = pd.DataFrame(corr_rows)
st.dataframe(
    corr_df.style.format({
        "Pearson_r": "{:.4f}", "Pearson_p": "{:.4f}",
        "Spearman_rho": "{:.4f}", "Spearman_p": "{:.4f}",
    }),
    use_container_width=True
)

# ---------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------
st.subheader("9) Visualizações")

tab1, tab2, tab3 = st.tabs(["Médias ± EPM", "Linhas por sujeito", "Boxplot + pontos"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [work[c].mean() for c in cols]
    sems = [stats.sem(work[c], nan_policy="omit") for c in cols]
    ax.bar(cols, means, yerr=sems, capsize=5)
    ax.set_ylabel("Tempo")
    ax.set_title("Média ± EPM por condição")
    plt.xticks(rotation=10)
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cols))
    for _, row in work[cols].iterrows():
        ax.plot(x, row.values, marker="o", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(cols, rotation=10)
    ax.set_ylabel("Tempo")
    ax.set_title("Trajetória individual por sujeito")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(9, 5))
    data_to_plot = [work[c].values for c in cols]
    ax.boxplot(data_to_plot, tick_labels=cols)
    jitter = 0.05
    rng = np.random.default_rng(123)
    for i, c in enumerate(cols, start=1):
        xvals = rng.normal(i, jitter, size=len(work))
        ax.plot(xvals, work[c], "o", alpha=0.55)
    ax.set_ylabel("Tempo")
    ax.set_title("Distribuição por condição")
    plt.xticks(rotation=10)
    st.pyplot(fig)

# ---------------------------------------------------------------------
# Relato estilo artigo
# ---------------------------------------------------------------------
st.subheader("10) Resumo interpretativo")
summary_text = summarize_findings(anova_p, friedman_p, param_posthoc, cols)
st.write(summary_text)

st.subheader("11) Texto sugerido para resultados")
anova_sentence = "A ANOVA de medidas repetidas não pôde ser calculada."
if pd.notna(anova_p):
    anova_sentence = (
        f"Uma ANOVA de uma via para medidas repetidas foi aplicada para comparar os tempos nas três condições "
        f"({cols[0]}, {cols[1]} e {cols[2]}). "
        f"O efeito de condição foi "
        f"{'significativo' if anova_p < alpha else 'não significativo'} "
        f"[p = {format_p(anova_p)}]."
    )

friedman_sentence = ""
if pd.notna(friedman_p):
    friedman_sentence = (
        f"Como análise complementar, o teste de Friedman foi "
        f"{'significativo' if friedman_p < alpha else 'não significativo'} "
        f"(p = {format_p(friedman_p)})."
    )

sig_posthoc = param_posthoc[param_posthoc["Significativo"] == True]
if len(sig_posthoc) > 0:
    posthoc_sentence = "Nas comparações múltiplas corrigidas, observaram-se diferenças em: " + \
                       "; ".join(
                           [
                               f"{r['Comparação']} (p corrigido = {format_p(r['p_corrigido'])}, dz = {r['Effect_size_dz']:.4f})"
                               for _, r in sig_posthoc.iterrows()
                           ]
                       ) + "."
else:
    posthoc_sentence = "Nas comparações múltiplas corrigidas, não foram observadas diferenças significativas entre os pares."

dtc_sentence = (
    f"O custo médio da dupla tarefa foi de {dtc['DTC_num_%'].mean():.2f}% para a condição numérica "
    f"e {dtc['DTC_palavras_%'].mean():.2f}% para a condição de nomeação de palavras."
)

article_text = " ".join([anova_sentence, friedman_sentence, posthoc_sentence, dtc_sentence])
st.text_area("Texto editável", value=article_text, height=220)

# ---------------------------------------------------------------------
# Exportação
# ---------------------------------------------------------------------
st.subheader("12) Exportação de resultados")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    desc.to_excel(writer, sheet_name="Descritiva", index=False)
    norm_df.to_excel(writer, sheet_name="Normalidade_diff", index=False)
    param_posthoc.to_excel(writer, sheet_name="Posthoc_param", index=False)
    nonparam_posthoc.to_excel(writer, sheet_name="Posthoc_nonparam", index=False)
    dtc_show = pd.concat([work[["Sujeito", single_col, dual_num_col, dual_word_col]], dtc], axis=1)
    dtc_show.to_excel(writer, sheet_name="DTC_individual", index=False)
    corr_df.to_excel(writer, sheet_name="Correlacoes", index=False)

st.download_button(
    "Baixar resultados em Excel",
    data=buffer.getvalue(),
    file_name="resultados_tug_medidas_repetidas.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.download_button(
    "Baixar resumo em TXT",
    data=article_text.encode("utf-8"),
    file_name="resumo_resultados_tug.txt",
    mime="text/plain"
)
