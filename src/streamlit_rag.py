import streamlit as st
import tempfile
import os
from portuguese_rag import PortugueseRAG
import requests
from collections import Counter
import re
import nltk
import numpy as np
from nltk.corpus import stopwords

class StreamlitRAGChat:
    def __init__(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'rag' not in st.session_state:
            st.session_state.rag = PortugueseRAG()

        if 'documento_carregado' not in st.session_state:
            st.session_state.documento_carregado = False

        if 'contexto_atual' not in st.session_state:
            st.session_state.contexto_atual = None
            
        try:
            nltk.download('stopwords')
        except:
            pass
        self.stop_words = set(stopwords.words('portuguese'))

    def consultar_ollama(self, question, model_name="neural-chat", temperature=0.7):
        url = "http://localhost:11434/api/generate"#<----- isso é um servidor criado na sua própria máquina pelo software Ollama (vide localhost)
#                                                     #       onde o modelo fica carregado na memória pra agilizar o retorno do modelo
        data = {
            "model": model_name,
            "prompt": question,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            return f"Erro ao gerar resposta: {str(e)}"

    def salvar_arquivo_carregado(self, uploaded_file):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                return file_path
        except Exception as e:
            st.error(f"Erro ao salvar o documento: {str(e)}")
            return None

    def extrair_frequencia_palavras(self, text):
        words = re.findall(r'\w+', text.lower())
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        word_freq = Counter(words)
        return dict(word_freq.most_common(20))

    def processar_pdfs(self, uploaded_files):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                num_chunks = st.session_state.rag.carregar_documentos_pdf(temp_dir)
                st.session_state.documento_carregado = True
                return num_chunks
        except Exception as e:
            st.error(f"Erro ao processar o PDF: {str(e)}")
            return None

    def obter_resposta_rag(self, question):
        try:
            context = st.session_state.rag.obter_contexto_relevante(question, num_chunks=3)
            if context:
                st.session_state.contexto_atual = context
                response = st.session_state.rag.consultar_ollama("mistral", question, context)
                return response
            return "Não foi possível encontrar contexto relevante para a pergunta."
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

    def criar_barra_lateral(self):
        with st.sidebar:
            st.header("Upload seu documento")
            st.write("Adicione um arquivo .pdf para acionar o RAG")
            uploaded_files = st.file_uploader(
                "Carregue seu documento", 
                type=['pdf'], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Processar documento"):
                    with st.spinner("Processando documento..."):
                        num_chunks = self.processar_pdfs(uploaded_files)
                        if num_chunks:
                            st.success(f"Documento processado com sucesso em {num_chunks} chunks!")

            st.divider()
            st.subheader("Status do sistema")
            st.write(f"Modo RAG: {'✅' if st.session_state.documento_carregado else '❌'}")
            st.write(f"Histórico de Chat: {len(st.session_state.messages)} mensagens")

            if st.session_state.documento_carregado:
                if st.button("Desabilitar modo RAG"):
                    st.session_state.documento_carregado = False
                    st.session_state.contexto_atual = None
                    st.rerun()

    def exibir_mensagens_chat(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def gerenciar_entrada_chat(self):
        if prompt := st.chat_input("Bora! Solta o verbo"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Calma, tô pensando..."):
                    if st.session_state.documento_carregado:
                        response = self.obter_resposta_rag(prompt)
                    else:
                        response = self.consultar_ollama(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    def executar(self):
        st.title("📚 Assistente Chatbot - Aplicação RAG")
        
        self.criar_barra_lateral()
        
        st.header("💬 Chat")
        
        if st.session_state.documento_carregado:
            st.info("🔍 Modo RAG Ativado, agora as respostas serão baseadas no documento carregado")
        
        self.exibir_mensagens_chat()
        
        self.gerenciar_entrada_chat()


if __name__ == "__main__":
    chat_app = StreamlitRAGChat()
    chat_app.executar()