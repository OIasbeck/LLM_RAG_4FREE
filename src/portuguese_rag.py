import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

class PortugueseRAG:
    def __init__(self, base_url="http://localhost:11434"): #<----- isso é um servidor criado na sua própria máquina pelo software Ollama (vide localhost)
#                                                     #       onde o modelo fica carregado na memória pra agilizar o retorno do modelo
        self.base_url = base_url
        self.vector_store = None
        
        self.tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="neuralmind/bert-base-portuguese-cased",  #<----- Passo para a classe do modelo a config. do modelo
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        try:
            nltk.download('punkt')
        except:
            print("Dados NLTK já baixados")
        
        self.text_splitter = TokenTextSplitter( #<---- Configuração de Chunks (divisão do documento de texto passado)
            chunk_size=500,
            chunk_overlap=50,
            encoding_name="cl100k_base"
        )

    def preprocessar_texto_portugues(self, text: str) -> str:
        text = text.replace('\n', ' ')
        text = ' '.join(text.split())
        
        sentences = sent_tokenize(text, language='portuguese')
        
        return ' '.join(sentences)

    def carregar_documentos_pdf(self, pdf_directory: str):
        documents = []
        
        if not os.path.exists(pdf_directory):
            raise ValueError(f"Diretório {pdf_directory} não existe")

        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_directory, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    
                    for doc in pdf_docs:
                        doc.page_content = self.preprocessar_texto_portugues(doc.page_content)
                    
                    documents.extend(pdf_docs)
                    print(f"Processado {filename}")
                except Exception as e:
                    print(f"Erro ao carregar {filename}: {str(e)}")

        if not documents:
            raise ValueError("Nenhum documento PDF carregado")

        texts = self.text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(   #< ------- Database indexado em memória para colocar o RAG
            documents=texts,
            embedding=self.embeddings
        )
        
        self.vector_store.save_local("faiss_index")
        
        print(f"Documentos processados: {len(documents)} com {len(texts)} chunks")
        return len(texts)

    def obter_contexto_relevante(self, question: str, num_chunks: int = 3) -> str:
        if not self.vector_store:
            raise ValueError("Nenhum documento foi carregado...")

        docs_and_scores = self.vector_store.similarity_search_with_score(
            question, 
            k=num_chunks
        )
        
        relevant_chunks = []
        for doc, score in docs_and_scores:
            if score < 1.0:
                relevant_chunks.append(doc.page_content)
        
        return " ".join(relevant_chunks)

    def consultar_ollama(self, model_name: str, question: str, context: str, 
                    temperature: float = 0.7) -> str:
        
        # Neste cenário, utiliza a busca estratégica no documento fragmentado pela configuração de chunk e disponibiliza no próprio prompt
        prompt = f"""Com base no seguinte contexto, responda à pergunta.
        Contexto: {context}
        
        Pergunta: {question}
        
        Resposta:"""

        url = f"{self.base_url}/api/generate"
        data = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            print(f"Erro na geração da resposta: {str(e)}")
            return None