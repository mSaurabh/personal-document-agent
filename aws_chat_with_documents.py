import boto3
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st
import json
import os
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore

# Initialize the Bedrock runtime client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

def load_document(file):
    import os
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    
    name, extension = os.path.splitext(file)
    
    print(f'Loading {file}...')
    if extension == ".pdf":
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        loader = TextLoader(file)
    else:
        print(f'Document extension {extension} is not supported.')
        return None
        
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks
    
def generate_embeddings(chunk):    
    # Prepare the payload        
    payload = {"inputText": str(chunk)}

    # Invoke the model
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps(payload)
    )
    
    # Parse the response
    result = json.loads(response['body'].read().decode())
    embedding = result['embedding']  # Adjust based on actual response structure
    return embedding

def create_embeddings(chunks):
    # Assuming you have a way to store these embeddings, e.g., in a vector store
    vector_store = PineconeVectorStore.from_documents(chunks, generate_embeddings, index_name='doc-chats')
    return vector_store

def ask_and_get_answer(vector_store,q,k=3):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chat_models import BedrockChat

    model_kwargs =  { 
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    model_id = "anthropic.claude-v2"

    model = BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs
    )

    template = """
Answer the question based on the given information:
{context}

Follow each of the following guardrails:
Use only the information provided in the document.
Provide concise and accurate answers.
Do not include any external information or assumptions.

Question: {question}
"""

    prompt = PromptTemplate.from_template(template)

    # Retrieve and Generate
    retriever = vector_store.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": 8},
    )

    print ("RETRIEVED")
    print (retriever)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(q)

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    load_dotenv(find_dotenv(), override=True)
    
    pc = Pinecone(os.environ.get("PINECONE_API_KEY"))

    st.image('image.png')
    st.divider()
    st.subheader('LLM Question-Answering Application 🤖')
    
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID') or ""
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or ""
    aws_session_token = os.environ.get('AWS_SESSION_TOKEN') or ""
    
    with st.sidebar:
        access_key = st.text_input('AWS Access Key ID:', type='password', value=aws_access_key_id)
        secret_key = st.text_input('AWS Secret Access Key:', type='password', value=aws_secret_access_key)
        session_token = st.text_input('AWS Session Token:', type='password', value=aws_session_token)
        
        if access_key and secret_key:
            os.environ['AWS_ACCESS_KEY_ID'] = access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
            os.environ['AWS_SESSION_TOKEN'] = session_token

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            if access_key: # and secret_key:
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./',uploaded_file.name)

                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                vector_store = create_embeddings(chunks)
                # Further processing with vector_store
                st.session_state.vs = vector_store
                st.success('File Uploaded, chunked and embedded successfully.')

    q = st.text_input('Ask a question about the content of your file:',disabled= access_key == "")
    if q:
        if access_key == "":
            st.error("Please enter a valid AWS Access Key.",icon="🚫")
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store,q,k)
            st.text_area('LLM Answer: ',value=answer);
    
            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history=''
            
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n { st.session_state.history}'
            h = st.session_state.history
            st.text_area(label="Chat History", value =h,key="history", 
                         height=400,placeholder="Search History is empty...")