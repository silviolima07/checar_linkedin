import sys
import sqlite3

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import pandas as pd
import streamlit as st
from crewai import Crew, Process
from my_agents import criar_agente_revisor
from my_tasks import criar_task_analise
from my_tools import save_uploaded_pdf, read_txt
import pdfplumber
import os
from PIL import Image
import time

import chardet
from MyLLM import LLMModels

from dotenv import load_dotenv
import groq

#os.environ["CREWAI_DISABLE_SQLITE"] = "1"  # Desativa o SQLite no CrewAI

#import sqlite3
#conn = sqlite3.connect(":memory:")  # Usa memória em vez de arquivo

# Carregar variáveis de ambiente
load_dotenv()

# Obter a chave da API GROQ
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Certifique-se de que a chave não é None antes de continuar, se for crítico.
if GROQ_API_KEY is None:
    st.error("Erro: A variável de ambiente GROQ_API_KEY não foi carregada. Verifique seu arquivo .env.")
    st.stop() # Interrompe a execução do Streamlit se a chave não estiver presente

llama_groq = LLMModels.GROQ_LLAMA_70B


# Criando o checkbox para mostrar ou não os comandos das tasks
#mostrar_comandos = st.checkbox("Mostrar progresso das tarefas em execução", value=True)

# Função para mostrar o progresso da execução das tarefas e capturar o resultado final
def executar_tarefas(crew, inputs):
    st.write("### Executando as tasks...")

    # Variável para armazenar o resultado final após a execução de todas as tasks
    result = None

    # Executa as tasks uma por uma e exibe o progresso no Streamlit
    for i, task in enumerate(crew.tasks):
        task_agent = (task.agent.role) # Nome do agente responsavel, definido em my_agents
        task_name = (task.name).upper()  # Nome da tasks, definido tem my_tasks
        st.write(f"Agent : **{task_agent}**")  # Mostra o nome do agent
        st.write(f"Executando task : **{task_name}**")  # Mostra o nome da task
        st.write(f"Descrição:")
        st.write(f"{task.description}")
        time.sleep(2)  # Simula o tempo de execução da task
        # Aqui você pode simular o progresso de cada task, ou capturar a execução real

    # Após a execução de todas as tasks, salva o resultado
    result = crew.kickoff(inputs=inputs)
    
    return result  # Retorna o resultado final


# Função para ler o PDF e extrair o texto
def extract_text_from_pdf(uploaded_file):
    text_content = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text_content += page.extract_text() + "\n"
    #st.write('Text extracted:', text_content)           
    return text_content

# Função para salvar o conteúdo extraído em um arquivo txt
def save_to_txt(text_content, output_filename="profile.txt"):
    with open(output_filename, "w", encoding="utf-8") as text_file:
        text_file.write(text_content)

# Função para ler o conteúdo de um arquivo markdown
def read_markdown_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado.")

html_page_title = """
     <div style="background-color:black;padding=60px">
         <p style='text-align:center;font-size:50px;font-weight:bold'>Revisão Perfil no Linkedin</p>
     </div>
               """               
st.markdown(html_page_title, unsafe_allow_html=True)


robo = Image.open("img/revisor.png")
st.sidebar.image(robo,caption="",use_container_width=True)

st.sidebar.markdown("# Menu")
option = st.sidebar.selectbox("Menu", ["Profile", 'About'], label_visibility='hidden')

if option == 'Profile':
    try:
        st.markdown("## Upload Profile Linkedin")
        uploaded_file = st.file_uploader("Envie o seu profile em PDF", type=["pdf"])
        if uploaded_file is not None:
            # Salvar PDF e extrair texto
            
            #save_uploaded_pdf(uploaded_file, 'profile.pdf')  # save pdf
            #st.write("Arquivo lido com sucesso")
        
            # Extrair texto do PDF
            text_content = extract_text_from_pdf(uploaded_file)
            #st.write('Text extracted:', text_content)
            save_to_txt(text_content, 'profile.txt')  # save txt
            #st.write("Arquivo lido e salvo com sucesso")
        
            with open("profile.txt", "rb") as f:
                raw_data = f.read()
        
            #st.write(raw_data)
            
            result_char = chardet.detect(raw_data)
            encoding = result_char['encoding']
        
            file_txt = read_txt('profile.txt', encoding)
            
            #st.markdown("#### Profile lido")
        
        
            # Configuração da crew com o agente recrutador
            revisor = criar_agente_revisor(llama_groq)
            # Cria a task usando o agente criado
            analise = criar_task_analise(revisor)
        
            st.markdown("## Analisar Perfil no Linkedin")   
            st.info("#### Avalie sempre a resposta final. O agente tem razão ou não?")

            crew = Crew(
                agents=[revisor],
                tasks=[analise],
                process=Process.sequential,  # Processamento sequencial das tarefas
                verbose=True
             )

            if st.button("INICIAR"):
                inputs = {
                      'profile': 'profile.txt',
                      'profile': file_txt,
                      'encoding': encoding,
                      'sugestao': 'sugestao_profile.md'}
                      
                with st.spinner('Wait for it...we are working...please'):
                    # Executa o CrewAI
                    try:
                        #result = crew.kickoff(inputs=inputs)
                        executar_tarefas(crew, inputs)
                
                    # Exibir resultado - ajuste para o tipo de dado CrewOutput
                    #if hasattr(result, 'raw'):
                    #    st.write("Resposta do agente:", result.raw)
                    #else:
                    #    st.write("Resposta do agente:", result)  # Exibir resultado diretamente
                
                        # Caminho para o arquivo Markdown
                        markdown_file_path = "sugestao_profile.md"
                
                        # Verificar se o arquivo Markdown existe e exibir
                        try:
                        
                            html_page_final = """
     <div style="background-color:black;padding=60px">
         <p style='text-align:center;font-size:50px;font-weight:bold'>Resultado da Análise</p>
     </div>
               """               
                            st.markdown(html_page_final, unsafe_allow_html=True)
                            
                            markdown_content = read_markdown_file(markdown_file_path)
                            
                            
                            st.markdown(markdown_content, unsafe_allow_html=True)
                    
                            # Adicionar botão de download para o arquivo Markdown
                            with open(markdown_file_path, "r", encoding="utf-8") as file:
                                st.download_button(
                                    label="Baixar Sugestão em Markdown",
                                    data=file,
                                    file_name="sugestao_profile.md",
                                    mime="text/markdown"
                                )
                        
                            st.markdown("## Boa sorte.")  
                                               
                        
                        except FileNotFoundError:
                            st.error(f"O arquivo Markdown {markdown_file_path} não foi encontrado.")
                    except Exception as e:
                        st.error(f"Erro ao executar o CrewAI: {str(e)}")
        else:
            st.markdown("##### Formato PDF")        
    except:
        st.error("Houston, we have a problem. Verifique o arquivo enviado. Deve ser o profile gerado no Linkedin.")
if option == 'About':
    robo = Image.open("img/get_profile.png")
    st.sidebar.image(robo,caption="",use_column_width=True)
    st.markdown("### Baixe o profile do seu perfil no Linkedin.")
    st.markdown("### Um agente revisor faz a análise e recomenda melhorias nas seções.")   
