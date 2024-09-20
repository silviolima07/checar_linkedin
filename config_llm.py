from dotenv import load_dotenv
import os
import groq

# Carregar variáveis de ambiente
load_dotenv()



# Obter a chave da API GROQ
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

llama_groq = "groq/llama3-70b-8192" 
      