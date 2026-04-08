import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import zscore

st.set_page_config(page_title="Análise Multivariada do Custo Cognitivo", layout="wide")

st.title("Análise Multivariada do Custo Cognitivo")

st.markdown("""
Este aplicativo permite explorar múltiplas variáveis de custo cognitivo por sujeito, incluindo:
- estatística descritiva
- correlação entre variáveis
- PCA
- clusterização
- comparação entre clusters
""")

uploaded_file = st.file_uploader("Carregue um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dados carregados")
    st.dataframe(df.head())

    colunas = df.columns.tolist()

    id_col = st.selectbox(
        "Selecione a coluna identificadora dos sujeitos (opcional)",
        options=["Nenhuma"] + colunas
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    selected_vars = st.multiselect(
        "Selecione as variáveis numéricas de custo cognitivo",
        options=numeric_cols,
        default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
    )

    if len(selected_vars) >= 2:
        dados = df[selected_vars].copy()

        if id_col != "Nenhuma":
            sujeitos = df[id_col].copy()
        else:
            sujeitos = pd.Series(np.arange(1, len(df) + 1), name="sujeito")

        st.subheader("Resumo descritivo")
        desc = dados.describe().T
        desc["cv_%"] = (desc["std"] / desc["mean"].replace(0, np.nan)) * 100
        st.dataframe(desc)

        st.subheader("Valores ausentes")
        missing = dados.isna().sum()
        st.dataframe(missing.to_frame("n_missing"))

        if dados.isna().sum().sum() > 0:
            metodo_nan = st.selectbox(
                "Como lidar com valores ausentes?",
                ["Remover linhas com NA", "Preencher com média"]
            )
            if metodo_nan == "Remover linhas com NA":
                mask = dados.notna().all(axis=1)
                dados = dados.loc[mask].reset_index(drop=True)
                sujeitos = sujeitos.loc[mask].reset_index(drop=True)
            else:
                dados = dados.fillna(dados.mean())

        st.subheader("Boxplot das variáveis")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.boxplot([dados[col].dropna() for col in dados.columns], tick_labels=dados.columns)
        ax.set_ylabel("Valor")
        ax.set_title("Distribuição das variáveis")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Heatmap de correlação")
        corr = dados.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlação entre variáveis")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

        st.subheader("Padronização")
        scaler = StandardScaler()
        X = scaler.fit_transform(dados)

        st.subheader("PCA")
        n_comp = st.slider("Número de componentes principais", 2, min(len(selected_vars), 10), 2)

        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)

        explained = pca.explained_variance_ratio_
        explained_df = pd.DataFrame({
            "Componente": [f"PC{i+1}" for i in range(len(explained))],
            "Variância explicada": explained,
            "Variância explicada acumulada": np.cumsum(explained)
        })
        st.dataframe(explained_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(explained) + 1), np.cumsum(explained), marker="o")
        ax.set_xlabel("Número de componentes")
        ax.set_ylabel("Variância explicada acumulada")
        ax.set_title("Curva da variância explicada")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=selected_vars,
            columns=[f"PC{i+1}" for i in range(n_comp)]
        )
        st.subheader("Loadings da PCA")
        st.dataframe(loadings)

        if n_comp >= 2:
            pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
            pca_df["sujeito"] = sujeitos.values

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(pca_df["PC1"], pca_df["PC2"])
            for _, row in pca_df.iterrows():
                ax.text(row["PC1"] + 0.02, row["PC2"] + 0.02, str(row["sujeito"]), fontsize=8)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Dispersão dos sujeitos no espaço PCA")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.subheader("Clusterização")
        n_clusters = st.slider("Número de clusters", 2, 6, 3)

        use_pca_for_cluster = st.checkbox("Usar componentes da PCA para clusterização", value=True)

        if use_pca_for_cluster:
            X_cluster = X_pca[:, :min(n_comp, 3)]
        else:
            X_cluster = X

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        clusters = kmeans.fit_predict(X_cluster)

        resultado = dados.copy()
        resultado["sujeito"] = sujeitos.values
        resultado["cluster"] = clusters

        sil = silhouette_score(X_cluster, clusters)
        st.write(f"**Silhouette score:** {sil:.3f}")

        if X_cluster.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            for c in sorted(np.unique(clusters)):
                subset = resultado[resultado["cluster"] == c]
                coords = X_cluster[resultado["cluster"] == c]
                ax.scatter(coords[:, 0], coords[:, 1], label=f"Cluster {c}")
                for idx, row in subset.iterrows():
                    ax.text(coords[list(subset.index).index(idx), 0] + 0.02,
                            coords[list(subset.index).index(idx), 1] + 0.02,
                            str(row["sujeito"]),
                            fontsize=8)
            ax.set_xlabel("Dimensão 1")
            ax.set_ylabel("Dimensão 2")
            ax.set_title("Clusters dos sujeitos")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        st.subheader("Resumo por cluster")
        resumo_cluster = resultado.groupby("cluster")[selected_vars].agg(["mean", "std", "median"])
        st.dataframe(resumo_cluster)

        st.subheader("Tabela final dos sujeitos")
        st.dataframe(resultado.sort_values("sujeito"))

        st.subheader("Heatmap das médias padronizadas por cluster")
        z = dados.apply(zscore)
        z["cluster"] = clusters
        medias_z = z.groupby("cluster").mean()

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(medias_z, aspect="auto")
        ax.set_xticks(range(len(medias_z.columns)))
        ax.set_xticklabels(medias_z.columns, rotation=90)
        ax.set_yticks(range(len(medias_z.index)))
        ax.set_yticklabels([f"Cluster {i}" for i in medias_z.index])
        ax.set_title("Perfil padronizado médio dos clusters")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

        st.subheader("Dendrograma hierárquico")
        Z = linkage(X, method="ward")
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Z, labels=sujeitos.astype(str).tolist(), leaf_rotation=90, ax=ax)
        ax.set_title("Dendrograma dos sujeitos")
        ax.set_xlabel("Sujeitos")
        ax.set_ylabel("Distância")
        st.pyplot(fig)

        csv = resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar tabela final com clusters",
            data=csv,
            file_name="resultado_clusters.csv",
            mime="text/csv"
        )

    else:
        st.warning("Selecione pelo menos 2 variáveis numéricas.")
else:
    st.info("Carregue um arquivo CSV para iniciar a análise.")
