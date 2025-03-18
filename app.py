import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplcyberpunk
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

# Criação da barra lateral
sidebar = st.sidebar

# Adicionar logo à barra lateral
logo = 'logo.png'  # Substitua pelo caminho correto para o seu logo
sidebar.image(logo, use_container_width=True)

# Adicionar espaço entre a logo e o botão de upload
sidebar.markdown("<br><br>", unsafe_allow_html=True)

# Criar abas
tab1, tab2 = st.tabs(["Diagnóstico de Brucelose", "Outra Análise"])

def carregar_dados(uploaded_file):
    """Carrega e interpola os dados do arquivo CSV."""
    dataframe = pd.read_csv(uploaded_file, header=0, index_col=0, delimiter=',',
                            names=['Número de Onda', 'Transmitância'])

    new_index = np.arange(round(dataframe.index[0]), round(dataframe.index[-1]) + 0.5, 0.5)
    dados_interp = pd.DataFrame(index=new_index)
    dados_interp.index.name = dataframe.index.name

    for colname, col in dataframe.items():
        dados_interp[colname] = np.interp(new_index, dataframe.index, col)

    dados_interp.sort_index(ascending=False, inplace=True)
    return dados_interp.loc[1800:900]

def exibir_grafico(dados, titulo):
    """Exibe um gráfico do espectro FTIR."""
    fig = plt.figure(figsize=(13, 6))
    plt.style.use("cyberpunk")
    plt.plot(dados, lw=2, color='green')
    mplcyberpunk.add_glow_effects()

    # Formatação
    plt.gca().invert_xaxis()
    plt.title(titulo, pad=10, fontsize=30, fontname='Cambria')
    plt.xlabel('Número de Onda ($\mathregular{cm^-¹}$)', labelpad=17, fontsize=26, fontname='Cambria')
    plt.ylabel('Transmitância Normalizada', labelpad=15, fontsize=28, fontname='Cambria')
    plt.xticks(np.arange(900, 1800 + 100, 100), fontsize=18, fontname='Cambria')
    plt.xlim(1800, 900)
    plt.ylim(dados.min().min() - 0.5, 100.5)
    st.pyplot(fig)

def aplicar_modelo(dados):
    """Aplica o modelo PCA + SVM e exibe a previsão."""
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
        
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    dados_intervalo = dados.loc[1500:900]
    dados_filtrados = pd.DataFrame(savgol_filter(dados_intervalo, 27, 1, axis=0))
    dados_filtrados.index = dados_intervalo.index

    dados_centrados = dados_filtrados - dados_filtrados.mean()
    dados_tratados = dados_centrados / dados_filtrados.std()

    X = np.transpose(dados_tratados)
    X_pca = pca.transform(X)

    prob = model.predict_proba(X_pca)[0]
    classes = ['Brucelose', 'Controle']
    
    probabilidade_bru = prob[0] * 100
    probabilidade_controle = prob[1] * 100

    cores = ['red', 'gray'] if probabilidade_bru > probabilidade_controle else ['gray', 'green']

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(classes, [probabilidade_bru, probabilidade_controle], color=cores)
    ax.set_xlabel('Probabilidade (%)', fontsize=12)
    ax.set_title('Distribuição das Probabilidades', fontsize=14)
    ax.set_xlim(0, 100)

    for i, v in enumerate([probabilidade_bru, probabilidade_controle]):
        ax.text(v + 2, i, f"{v:.2f}%", color='white', va='center', fontsize=10)

    st.pyplot(fig)
    st.write(f"**Diagnóstico:** {classes[np.argmax(prob)]}")

with tab1:
    st.markdown("## Diagnóstico de Brucelose 🐄")
    uploaded_file_1 = sidebar.file_uploader("Carregue o espectro FTIR para Brucelose", type="csv", key="upload_1")
    
    if uploaded_file_1 is not None:
        dados_brucelose = carregar_dados(uploaded_file_1)
        exibir_grafico(dados_brucelose, "Espectro FTIR - Diagnóstico de Brucelose")
        aplicar_modelo(dados_brucelose)

with tab2:
    st.markdown("## Outra Análise 🔬")
    uploaded_file_2 = sidebar.file_uploader("Carregue o espectro FTIR para outra análise", type="csv", key="upload_2")

    if uploaded_file_2 is not None:
        dados_outro = carregar_dados(uploaded_file_2)
        exibir_grafico(dados_outro, "Espectro FTIR - Outra Análise")
        # Se houver outro modelo, ele pode ser chamado aqui com uma função diferente
