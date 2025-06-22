from crewai import Agent
import streamlit as st


# Configuração do agente


def criar_agente_revisor(modelo):
    
    revisor = Agent(
        role="revisor",
        goal="Usar o arquivo {profile}."
             " Recomendar melhorias no texto do perfil do usuario no linkedin.",
        backstory=
            "Você é um experiente recrutador atualizado com os critérios de busca feitos pelos recrutadores. "
            "Utilize o sistemas ATS (Applicant Tracking Systems) para avaliasr o arquivo enviado."
            "Você trabalha numa grande empresa de recrutamento e sabe os critérios usados na seleção de candidatos a vagas."
            "Você sugere palavras chaves que devem ser inseridas no texto do perfil no linkedin."
            "Você orienta melhorias que o usuário deve fazer no perfil para que seja chamado pelos recrutadores para participar de processos de contratação."
            "Você sempre apresenta exemplos das melhorias que devem ser feitas no texto do perfil."
        ,
        llm=modelo,
        verbose=True,
        memory=False,
        tools=[]
    )
    #st.markdown("#### - Agente Revisor acionado")
    #st.write(revisor_link.goal)
    return revisor   
 
