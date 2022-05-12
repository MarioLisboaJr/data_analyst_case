'''
Utilizando o app no localhost:
    Para utilizar o app é necessário ter o streamlit instalado no seu gerenciador, como o Anaconda por exemplo.

    No Prompt do Anaconda, abra o terminal em seu ambiente e insira o comando "streamlit run "app.py"

    Assim que você executar o script conforme mostrado acima, um servidor Streamlit local será ativado e seu aplicativo será aberto em uma nova guia no navegador da Web padrão. O aplicativo é sua prória tela.

    É necessário ter no mesmo diretório deste arquivo os arquivos:
        - modelo_classificacao.joblib
        
Utilizando o app na web:
    Acessar https://share.streamlit.io/mariolisboajr/data_analyst_case/main/app_classificacao.py
    
'''


#importar bibliotecas
import pandas as pd
import streamlit as st
import joblib
import time


#configuração padrão da página
st.set_page_config(
    page_title='iFood Case',
    page_icon='https://entregador.ifood.com.br/wp-content/themes/iFood/assets/images/iFood_logo.svg',
    layout='centered'
)


#descrição
st.write('Mário José Lisbôa Júnior')
st.title('MODELO PREDITIVO DE CLASSIFICAÇÃO')
st.caption('# Classifica potenciais clientes para maximizar o sucesso da próxima campanha de marketing.\n ## Insira as informações sobre os cliente abaixo e descubra se vale a pena direcionar a campanha para ele ou não:')


#parâmetros do modelo
x = {
    'Número de dias desde a última compra:': 5,
    'Tempo como cliente da empresa:': 10,
    'Valor gasto em carnes nos últimos dois anos:': 1,
    'Número de compras realizadas nas lojas físicas:': 2,
    'Número de visitas no site da empresa no último mês:': 6,
    'Total de campanhas de marketing que o cliente já participou:': 1,
    'Grau de escolaridade:': 1    
    }


#ajustar campos no streamlit
for item in x:
    
    if item == 'Tempo como cliente da empresa:':
        valor = st.number_input(label=f'{item}', value=10 ,step=1, min_value=0, help='Informação em Anos')
    
    elif item == 'Grau de escolaridade:':
        valor = st.selectbox(label=f'{item}', options=('Ensino Médio', 'Ensino Superior', 'Pós-Graduação'))
        if valor == 'Ensino Médio':
            valor = 1
        elif valor == 'Ensino Superior':
            valor = 2
        else:
            valor = 3
    
    elif item == 'Total de campanhas de marketing que o cliente já participou:':
        valor = st.slider(label=f'{item}' ,min_value=0, value=1, max_value=5, step=1)
    
    else:
        valor = st.number_input(label=f'{item}',value=x[item], step=10, min_value=0)
    
    #receber valores do input    
    x[item] = valor


#tratar dados do input e padronizar com os dados do treino    
data = {
    'Recency': [49.012635, 28.948352], 
    'Enrollment_Time': [8.971570, 0.685618],
    'MntMeatProducts': [166.995939, 224.283273],
    'NumStorePurchases': [5.800993, 3.250785],
    'NumWebVisitsMonth': [5.319043, 2.425359],
    'AcceptedAnyCmp': [0.298285, 0.679209], 
    'Education': [2.267148, 0.652084]    
}
df = pd.DataFrame(data, index=['mean', 'std'])
mean = df[df.index=='mean']
std = df[df.index=='std']

#padronizar input
valores_x = pd.DataFrame(x, index=[0])
valores_x = (valores_x.values - mean.values)/std.values


#criar botao para rodar modelo
botao = st.button('Classificar Cliente')
if botao:
    
    #importar modelo treinado
    modelo = joblib.load('modelo_classificacao.joblib')
    
    #fazer previsão
    classificar = modelo.predict(valores_x)
    
    #aguardar conclusão da previsão
    with st.spinner('Aguarde um momento...'):
        time.sleep(1)
    
    #exibir resultado
    if classificar[0] == 1:
        st.success('### Cliente em potencial. Oferecer 6ª campanha!')
    else:
        st.error('### Cliente não possui perfil para a 6ª campanha!')