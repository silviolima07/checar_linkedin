from crewai import Agent
from crewai_tools import FileReadTool
#from app import modelo
import streamlit as st


# Ferramenta para leitura de arquivo cv.txt
reader_tool = FileReadTool()

# Configuração do agente

def criar_agente_revisor(modelo):
    
    revisor_link = Agent(
        role="revisor",
        goal="Usar o arquivo {profile}."
             " Recomendar melhorias no texto do perfil do usuario no linkedin.",
        backstory=
            "Você é um experiente recrutador atualizado com os critérios de busca feitos pelos recrutadores. "
            "Você trabalha numa grande empresa de recrutamento e sabe os critérios usados na seleção de candidatos a vagas."
            "Você sugere palavras chaves que devem ser inseridas no texto do perfil no linkedin."
            "Você orienta melhorias que o usuário deve fazer no perfil para que seja chamado pelos recrutadores para participar de processos de contratação."
            "Você sempre apresenta exemplos das melhorias que devem ser feitas no texto do perfil."
        ,
        llm=modelo,
        verbose=True,
        memory=False
        #tools=[reader_tool]
    )
    return revisor_link    
 
