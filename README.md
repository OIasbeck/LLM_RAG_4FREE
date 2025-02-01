## Sumário

- [Sobre](#Sobre)
- [Requisitos](#Requisitos)

## Sobre
IA generativa textual com aplicação RAG

Este projeto teve como premissa uma abordagem do poder comunitário do setor de tecnologia, explorando um modelo generativo textual gratuito, disposto no site HuggingFace. Ademais, afim de boas práticas de exibições, realizei a integração do sistema de chat com um ambiente front pelo framework Streamlit.  

![image](https://github.com/user-attachments/assets/13a8bbb8-1260-459b-afe8-11d3766f4f2f)

## Requisitos
- Python 3.x 
- Ollama (software para gerenciar modelos de IA localmente) (Esse software carrega os modelos em memória no computador e o código se comunica através dele por servidor local que ele mesmo cria em nossa máquina)

Frameworks
- streamlit        # Para interface web
- langchain       # Para processamento de documentos e RAG
- transformers    # Para modelos de linguagem
- faiss-cpu      # Para busca vetorial (Banco em memória que falei para usarmos RAG)
- nltk           # Para processamento de texto
- plotly         # Para visualizações
- requests       # Para comunicação HTTP
- PyPDF2         # Para leitura de PDFs

Artefatos
Modelos de IA
  - Modelo base do Ollama (mistral, neural-chat, etc.)
  - neuralmind/bert-base-portuguese-cased (modelo em português para embeddings)

Recursos NLP
  - punkt (tokenizador)
  - stopwords (palavras de parada em português)
