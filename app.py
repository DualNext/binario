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

# Widget de upload de arquivo na barra lateral
uploaded_file = sidebar.file_uploader('Use um arquivo CSV (separado por vírgula)', type="csv")

# Inicializar a variável de estado para exibir o botão e a mensagem
if "show_button" not in st.session_state:
    st.session_state.show_button = True

# Verifica se um arquivo foi carregado
if uploaded_file is not None:
    # Ler o conteúdo do arquivo em um DataFrame
    dataframe = pd.read_csv(uploaded_file, header=0, index_col=0, delimiter=',',
                            names=['Número de Onda', 'Transmitância'])

    # Interpolação de dados oriundos de Agilent
    def interp(df, new_index):
        df_out = pd.DataFrame(index=new_index)
        df_out.index.name = df.index.name
        for colname, col in df.items():
            df_out[colname] = np.interp(new_index, df.index, col)
        return df_out

    new_index = np.arange(round(dataframe.index[0]), round(dataframe.index[-1]) + 0.5, 0.5)
    dados = interp(dataframe, new_index)
    dados.sort_index(ascending=False, inplace=True)

    # Filtrar a faixa de 1800 a 900
    dados_coletados = dados.loc[1800:900]

    # Exibir as primeiras cinco linhas do DataFrame na barra lateral
    sidebar.write('Arquivo Carregado!')
    sidebar.dataframe(dados_coletados.head(5))

    # Criar colunas para o gráfico e resultados
    col1 = st.columns(1)[0]  # Pegando a primeira (e única) coluna

    # Exibir gráfico FTIR antes de pressionar o botão
    if st.session_state.show_button:
        with col1:
            fig = plt.figure(figsize=(13, 6))
            plt.style.use("cyberpunk")

            # Criar sua linha
            plt.plot(dados_coletados, lw=2, color='green')  # Linha na cor verde

            # Adicionar efeitos de brilho
            mplcyberpunk.add_glow_effects()

            # Formatação do gráfico
            plt.gca().invert_xaxis()
            plt.title('Espectro FTIR', pad=10, fontsize=30, fontname='Cambria')
            plt.xlabel('Número de Onda ($\mathregular{cm^-¹}$)', labelpad=17, fontsize=26, fontname='Cambria')
            plt.ylabel('Transmitância Normalizada', labelpad=15, fontsize=28, fontname='Cambria')
            plt.xticks(np.arange(900, 1800 + 100, 100), fontsize=18, fontname='Cambria')
            plt.gca().tick_params(axis='x', pad=20)  # Ajusta o espaço entre os rótulos e a linha
            plt.xlim(1800, 900)
            plt.ylim(dados_coletados.min().min() - 0.5, 100.5)
            plt.yticks(fontsize=18, fontname='Cambria')
            plt.gca().tick_params(axis='y', pad=20)  # Ajusta o espaço entre os rótulos e a linha
            st.pyplot(fig)

    # Exibir a mensagem e o botão "Continuar" apenas se for permitido
    st.info('Espectro medido corretamente! Clique em "continuar"')

    if st.button('Continuar'):
        st.session_state.show_button = False  # Ocultar mensagem e botão após o clique

    # Exibir o gráfico de pizza com as probabilidades após o botão ser pressionado
    if not st.session_state.show_button:
        
        # Carregar os modelos treinados
        with open('pca.pkl', 'rb') as f:
            pca = pickle.load(f)
            
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Pré-tratamento (Savitzky-Golay + Normalização)
        dados_intervalo = dados.loc[1500:900]
        
        dados_filtrados = pd.DataFrame(savgol_filter(dados_intervalo, 27, 1, axis = 0))
        dados_filtrados.index = dados_intervalo.index

        dados_centrados = dados_filtrados - dados_filtrados.mean()
        dados_tratados = dados_centrados / dados_centrados.std()

        # Aplicar PCA
        X = np.transpose(dados_tratados)
        X_pca = pca.transform(X)

        # Fazer previsões com SVM
        prob = model.predict_proba(X_pca)[0]
        
        # Definir as classes e probabilidades
        classes = ['Brucelose', 'Controle']
        probabilidade_bru = prob[0] * 100  # Probabilidade de Brucelose
        probabilidade_controle = prob[1] * 100       # Probabilidade de Controle

        # Definir cores dinamicamente
        if probabilidade_bru > probabilidade_controle:
            cores = ['red', 'gray']  # Vermelho para Brucelose, Cinza para Controle
        else:
            cores = ['gray', 'green']  # Cinza para Brucelose, Verde para Controle
    
        # Exibir gráfico de pizza com as probabilidades
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie([probabilidade_bru, probabilidade_controle], labels=classes, autopct='%1.2f%%', startangle=90, colors=cores)
        ax.set_title('Probabilidades de Diagnóstico', fontsize=10)
        st.pyplot(fig)

else:
    st.markdown('''<h1 style="color: orange; font-size: 35px;">Diagnóstico de Brucelose Bovina</h1>''', unsafe_allow_html=True)
    # Subtítulo (h3)
    st.markdown('''<h3 style="color: white; font-size: 20px;">Carregue um espectro FTIR para análise</h3>''', unsafe_allow_html=True)
