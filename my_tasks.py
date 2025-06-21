from crewai import Task
import streamlit as st

# my_tasks.py

def criar_task_analise(revisor):
    analise = Task(
        name='analise',
        description=(
            """
             Usar o arquivo profile enviado.
             
             Numa escala de 0 a 10, onde 0 é a menor importância e 10 a maior importância.  
             Avaliar o arquivo lido e apresentar uma nota.            
             Analisar o texto do perfil do usuário  e recomende melhorias.
             Use a experiência como recrutador de uma grande empresa de recolocação para orientar nas melhorias.
             
             Faça comentários relevantes em cada seção e não repita recomendações.
             Recomendar sempre baseado nos conhecimentos citados no arquivo atual lido.
             Todas seções devem ser avaliadas e melhorias recomendadas caso haja algo para melhorar.
             
             Recomendar as melhores palavras chaves e melhorias no texto do perfil lido.
             Identificar cada seção analisada tais como Titulo onde existem os termos separados por '|'.
             Na seção Titulo devem estar palavras chaves que identificam conhecimentos ou o cargo desejado.
             
             Recomendar na seção Titulo termos que sejam compativeis com o texto em Sumário.
             Identificar a seção Sumário e reescrever o texto do Sumário mostrando as melhorias que devem ser feitas.
             As melhorias devem elevar a nota atribuída inicialmente ao arquivo sendo analisado para 10.
             
             Faça comentários em Português do Brasil.
             Revisar a acentuação do texto, garanta que esta correta.
             Salvar as recomendações num arquivo chamado {sugestao}.
             """  )           ,
        expected_output=
             """
             Arquivo markdown(.md), um texto claro, em Português do Brasil.         
             Usar fonte size igual a 16 e dar espaço de uma linha nas respostas.
             Um relatório detalhado com:            
             1 - Classificação segundo os critérios definidos ;
             2 - Recomendação do titulo;
             3 - Recomendação do Sumário;
             4 - Recomendação da seção Experiencia;
             5 - Recomendação da seção Educação;
             6 - Tops Skills deve ser compatível com o texto do Sumário;
             7 - Recomendação da seção Idiomas;
             8 - Recomendação da seção Certificações
             9 - Conhecimentos gerais
              """
         ,
         agent=revisor,
         output_file='sugestao_profile.md'
     )
    #st.markdown("### - Task analise criada.")
    return analise



    

   
