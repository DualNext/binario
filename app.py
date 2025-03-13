import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplcyberpunk
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from tensorflow.keras.models import load_model

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
    dados_intervalo = dados.loc[1800:900]

    # Exibir as primeiras cinco linhas do DataFrame na barra lateral
    sidebar.write('Arquivo Carregado!')
    sidebar.dataframe(dados_intervalo.head(5))

    # Criar colunas para o gráfico e resultados
    col1, col2 = st.columns((5, 1))  # Ajuste a proporção conforme necessário

    # Exibir um gráfico de linhas com os dados filtrados na primeira coluna
    with col1:
        fig = plt.figure(figsize=(13, 6))
        plt.style.use("cyberpunk")

        # Criar sua linha
        plt.plot(dados_intervalo, lw=2, color='green')  # Linha na cor verde

        # Adicionar efeitos de brilho
        mplcyberpunk.add_glow_effects()

        # Formatação do gráfico
        plt.gca().invert_xaxis()
        plt.title('Espectro FTIR', pad=10, fontsize=26, fontname='Cambria')
        plt.xlabel('Número de Onda ($\mathregular{cm^-¹}$)', labelpad=15, fontsize=22, fontname='Cambria')
        plt.ylabel('Transmitância Normalizada', labelpad=15, fontsize=24, fontname='Cambria')
        plt.xticks(np.arange(900, 1800 + 100, 100), fontsize=18, fontname='Cambria')
        plt.xlim(1800, 900)
        plt.yticks(fontsize=16, fontname='Cambria')
        st.pyplot(fig)

    # Exibir a mensagem e o botão "Continuar" apenas se for permitido
    if st.session_state.show_button:
        st.info('Espectro medido corretamente! Clique em "continuar"')

        if st.button('Continuar'):
            st.session_state.show_button = False  # Ocultar mensagem e botão após o clique

    # Exibir o gráfico de barras apenas após o botão ser pressionado
    if not st.session_state.show_button:
        # Carregar os modelos
        model1 = load_model('model1.keras')
        model2 = load_model('model2.keras')
        
        # Pré-tratamento (SG)
        dados_filtrado = savgol_filter(dados_intervalo, 27, 1, axis=0)

        # Pré-tratamento (SNV)
        scaler = StandardScaler()
        dados_norm = scaler.fit_transform(dados_filtrado)
        X = np.transpose(dados_norm)  # Matriz
        X = X.reshape((X.shape[0], X.shape[1]))

        # Fazer previsões com os modelos
        prob1 = model1.predict(X)[0]
        prob2 = model2.predict(X)[0]
        
        # Determinar as classes e as probabilidades para os dois modelos
        classes_model1 = ['Controle', 'Positivo']
        classes_model2 = ['Brucelose', 'Tuberculose']

        # Probabilidades para as classes do primeiro modelo (Controle/Positivo)
        probabilidade_controle = prob1[0] * 100  # Probabilidade de Controle
        probabilidade_doente = prob1[1] * 100    # Probabilidade de Positivo

        # Se for "Positivo", usar o modelo2 para prever Brucelose/Tuberculose
        if probabilidade_doente > probabilidade_controle:
            probabilidade_bru = prob2[0] * 100  # Probabilidade de Brucelose
            probabilidade_tub = prob2[1] * 100  # Probabilidade de Tuberculose
        else:
            probabilidade_bru = 0  # Se não for "Positivo", probabilidade de Brucelose é 0
            probabilidade_tub = 0  # Se não for "Positivo", probabilidade de Tuberculose é 0

        # Exibir o gráfico de barras
        with col1:
            fig, ax = plt.subplots(figsize=(5, 3))
            
            # Definir as cores para as barras
            cores = ['red' if c == 'Positivo' else 'gray' for c in ['Controle', 'Positivo', 'Brucelose', 'Tuberculose']]
            
            # Criar as barras horizontais
            ax.barh(['Controle', 'Positivo', 'Brucelose', 'Tuberculose'], 
                    [probabilidade_controle, probabilidade_doente, probabilidade_bru, probabilidade_tub], color=cores)
            
            # Definir o título e o rótulo do eixo X
            ax.set_xlabel('Probabilidade (%)', fontsize=12)
            ax.set_title('Distribuição das Probabilidades', fontsize=14)
            ax.set_xlim(0, 100)
            
            # Adicionar texto nas barras para exibir as porcentagens
            for i, v in enumerate([probabilidade_controle, probabilidade_doente, probabilidade_bru, probabilidade_tub]):
                ax.text(v + 2, i, f"{v:.2f}%", color='white', va='center', fontsize=10)
            
            # Exibir o gráfico
            st.pyplot(fig)

else:
    st.markdown('''<h1 style="color: orange; font-size: 35px;">Diagnóstico Multiclasse de Doenças Bovinas</h1>''', unsafe_allow_html=True)

    # Subtítulo (h3)
    st.markdown('''<h3 style="color: white; font-size: 20px;">Carregue um espectro FTIR para análise</h3>''', unsafe_allow_html=True)
