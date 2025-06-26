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
from kokoro import KPipeline
import soundfile as sf
import torch
import warnings
import textwrap
import re

def extrair_secoes(markdown_content):
    padrao = re.compile(r"^#+\s*(.+)", re.MULTILINE)
    secoes = padrao.findall(markdown_content)
    return secoes
    

def limpar_markdown(texto_md):
    texto_limpo = re.sub(r"[#><*`_\[\]\(\)\-]", "", texto_md)
    texto_limpo = re.sub(r"\s{2,}", " ", texto_limpo)
    texto_limpo = texto_limpo.replace('</span>','')
    texto_limpo = texto_limpo.replace('<span style="font-size: 16px;">', '')
    texto_limpo = texto_limpo.replace('<br>', '')
    return texto_limpo.strip()


def extrair_secao(markdown, secao):
    padrao = rf"(?<=# {secao})(.*?)(?=\n#|\Z)"
    resultado = re.search(padrao, markdown, re.DOTALL | re.IGNORECASE)
    return resultado.group(0).strip() if resultado else None


HF_TOKEN = st.secrets["HF_TOKEN"]
os.environ["HF_TOKEN"] = HF_TOKEN

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

#os.environ.pop("HF_TOKEN", None)
#os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)

from huggingface_hub import login

warnings.filterwarnings('ignore')

#print('\nCarregar chaves\n')
#load_dotenv()

#GROQ_API_KEY = os.getenv('GROQ_API_KEY')
#HF_TOKEN = os.getenv('HF_TOKEN')

#st.write("HF_TOKEN:",  HF_TOKEN)

#os.environ["GROQ_API_KEY"] = GROQ_API_KEY
#os.environ["HF_TOKEN"] = HF_TOKEN

@st.cache_resource(show_spinner="Carregando modelo Kokoro...")
def carregar_pipeline():
    return KPipeline(lang_code='p', repo_id='hexgrad/Kokoro-82M')

pipeline = carregar_pipeline()

def ler_com_kokoro(texto, pipeline):
    with st.spinner('🎧 Gerando áudio com Kokoro, aguarde...'):
        #st.info("🎧 Gerando áudio com Kokoro, aguarde...")
        texto_limpo = limpar_markdown(texto)
        partes = textwrap.wrap(texto_limpo, width=800)
        audio_total = []
        for parte in partes:
            for _, _, audio in pipeline(parte, voice='bf_isabella'):
                audio_total.append(audio)
        audio_concat = torch.cat(audio_total).cpu().numpy()
        sf.write("voz_sugestao.wav", audio_concat, 24000)
        with open("voz_sugestao.wav", "rb") as f:
           st.audio(f.read(), format="audio/wav")

def executar_tarefas(crew, inputs):
    st.write("### Executando as tasks...")
    result = None
    for i, task in enumerate(crew.tasks):
        st.write(f"Agent : **{task.agent.role}**")
        st.write(f"Executando task : **{task.name.upper()}**")
        st.write("Descrição:")
        st.write(f"{task.description}")
        time.sleep(2)
    result = crew.kickoff(inputs=inputs)
    return result

def extract_text_from_pdf(uploaded_file):
    text_content = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text_content += page.extract_text() + "\n"
    return text_content

def save_to_txt(text_content, output_filename="profile.txt"):
    with open(output_filename, "w", encoding="utf-8") as text_file:
        text_file.write(text_content)

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
st.sidebar.image(robo, caption="", use_container_width=True)

st.sidebar.markdown("# Menu")
option = st.sidebar.selectbox("Menu", ["Profile", 'About'], label_visibility='hidden')

if option == 'Profile':
    try:
        st.markdown("## Upload Profile Linkedin")
        uploaded_file = st.file_uploader("Envie o seu profile em PDF", type=["pdf"])
        if uploaded_file is not None:
            text_content = extract_text_from_pdf(uploaded_file)
            save_to_txt(text_content, 'profile.txt')

            with open("profile.txt", "rb") as f:
                raw_data = f.read()

            result_char = chardet.detect(raw_data)
            encoding = result_char['encoding']
            file_txt = read_txt('profile.txt', encoding)

            revisor = criar_agente_revisor(LLMModels.GROQ_LLAMA_70B)
            analise = criar_task_analise(revisor)

            st.markdown("## Analisar Perfil no Linkedin")   
            st.info("#### Avalie sempre a resposta final. O agente tem razão ou não?")

            crew = Crew(
                agents=[revisor],
                tasks=[analise],
                process=Process.sequential,
                verbose=True
            )

            if st.button("INICIAR"):
                inputs = {
                    'profile': file_txt,
                    'encoding': encoding,
                    'sugestao': 'sugestao_profile.md'
                }

                with st.spinner('Wait for it...we are working...please'):
                    try:
                        executar_tarefas(crew, inputs)
                        markdown_file_path = "sugestao_profile.md"

                        try:
                            html_page_final = """
                                <div style="background-color:black;padding=60px">
                                    <p style='text-align:center;font-size:50px;font-weight:bold'>Resultado da Análise</p>
                                </div>
                            """               
                            st.markdown(html_page_final, unsafe_allow_html=True)

                            markdown_content = read_markdown_file(markdown_file_path)
                            st.session_state['markdown'] = markdown_content
                            st.markdown(markdown_content)
                            #lista_secoes = markdown_content.startswith('#')
                            #st.write('Seções:', lista_secoes.)

                            with open(markdown_file_path, "r", encoding="utf-8") as file:
                                st.download_button(
                                    label="📥 Baixar Sugestão em Markdown",
                                    data=file,
                                    file_name="sugestao_profile.md",
                                    mime="text/markdown"
                                )

                        except FileNotFoundError:
                            st.error(f"O arquivo Markdown {markdown_file_path} não foi encontrado.")
                    except Exception as e:
                        st.error(f"Erro ao executar o CrewAI: {str(e)}")
        else:
            st.write("Formato PDF")        
    except:
        st.error("Houston, we have a problem. Verifique o arquivo enviado. Deve ser o profile gerado no Linkedin.")

    if 'markdown' in st.session_state and uploaded_file is None:
        st.markdown('### Análise do Arquivo por Seções')
        lista_secoes = extrair_secoes(st.session_state['markdown'])
        
        secao_escolhida = st.selectbox("Escolha a seção para ouvir", lista_secoes)

        if st.button("🔊 Ouvir Seção Selecionada"):
            secao_texto = extrair_secao(st.session_state['markdown'], secao_escolhida)
            if secao_texto:
                ler_com_kokoro(secao_texto, pipeline)
            else:
                st.warning("Seção não encontrada no texto.")

if option == 'About':
    robo = Image.open("img/get_profile.png")
    st.sidebar.image(robo, caption="", use_column_width=True)
    st.markdown("### Este aplicativo faz a leitura do profile do Linkedin em pdf.")
    st.markdown("### Baixe seu profile no botão Mais do seu perfil.")
    st.markdown("### Um agente revisor faz a leitura e recomenda melhorias nas seções.")
    st.markdown("### Modelos acessados via Groq.")
