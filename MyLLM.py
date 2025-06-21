from crewai import LLM

class LLMModels:
    GPT4o_mini    = LLM(model='gpt-4o-mini')
    GPT4o_mini_2024_07_18    = LLM(model='gpt-4o-mini-2024-07-18')
    GPT4o    = LLM(model='gpt-4o')
    GPT_o1    = LLM(model='01-preview')
    GPT3_5    = LLM(model='gpt-3.5-turbo')
    LLAMA3_70B    = LLM(model='llama3-70b-8192')
    GROQ_LLAMA_8B    = LLM(model='groq/llama-3.1-8b-instant')
    GROQ_MIXTRAL    = LLM(model='groq/mixtral-8x7b-32768')
    GROQ_LLAMA_70B    = LLM(model='groq/llama-3.3-70b-versatile')
    GROQ_LLAMA_MM    = LLM(model='groq/llama-3.2-11b-vision-preview')
