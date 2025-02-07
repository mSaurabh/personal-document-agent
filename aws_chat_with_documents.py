# Import necessary libraries
import boto3
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st
import json
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as PineconeVectorStore

# Initialize the Bedrock runtime client
# This client will be used to interact with Amazon Bedrock for model inference
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

def load_document(file):
    """
    Load a document based on its file extension.
    Supports PDF, DOCX, and TXT formats.
    """
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
        
    # Load the document using the appropriate loader
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """
    Split the document data into smaller chunks for processing.
    Uses RecursiveCharacterTextSplitter from langchain.
    
    Parameters:
    - data: The document data to be split.
    - chunk_size: The size of each chunk.
    - chunk_overlap: The number of characters to overlap between chunks.
    
    Returns:
    - chunks: A list of document chunks.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def filter_chunks_by_role(chunks, role):
    # Implement your filtering logic here
    # For example, you can filter chunks that contain the role keyword
    filtered_chunks = [chunk for chunk in chunks if role.lower() in chunk.page_content.lower()]
    return filtered_chunks

def generate_embeddings(chunk):    
    """
    Generate embeddings for a given text chunk using Amazon Bedrock.
    
    Parameters:
    - chunk: The text chunk to generate embeddings for.
    
    Returns:
    - embedding: The generated embeddings.
    """
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
    """
    Create embeddings for all chunks of the document.
    
    Parameters:
    - chunks: The list of document chunks.
    
    Returns:
    - vector_store: A vector store containing the embeddings.
    """
    # Assuming you have a way to store these embeddings, e.g., in a vector store
    vector_store = PineconeVectorStore.from_documents(chunks, generate_embeddings, index_name='doc-chats')
    return vector_store

def ask_and_get_answer(vector_store, q, role=None, k=3):
    """
    Ask a question and get an answer using the Bedrock model and a vector store.
    
    Parameters:
    - vector_store: The vector store to use for retrieving relevant documents.
    - q: The question to ask.
    - k: The number of relevant documents to retrieve.
    
    Returns:
    - answer: The generated answer.
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chat_models import BedrockChat
    
    global guardrails_list

    model_kwargs =  { 
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

    model_id = "anthropic.claude-v2"

    # Initialize the BedrockChat model
    model = BedrockChat(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs
    )


    '''
    Here is an example of what our prompt

    {
        "role": "system",
        "content": f"""You are an AI assistant that summarizes financial data received from a
        semantic search based on a user prompt
        Output a well structured answer.
        If the user query is a greeting ignore semantic search data and return a greeting.

        Here is the User's question: {q}
        Here is the semantic search data:"""
        + result_string,
    }
    '''

    # Define the prompt template
    if role:
        template = f"You are answering as a {role}. Only provide answers relevant to this role based on the given information:\n"
    else:
        template =  "Answer the question based on the given information:\n" 
        
    template += "{context}\n" + \
                "Follow each of the following guardrails:\n" + \
                guardrails_list + "\n" + \
                "Question: {question}\n"

    prompt = PromptTemplate.from_template(template)

    # Retrieve and Generate
    retriever = vector_store.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": 8},
    )

    # Define the chain of operations
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Invoke the chain with the question
    return chain.invoke(q)

def clear_history():
    """
    Clear the session history in Streamlit.
    """
    if 'history' in st.session_state:
        del st.session_state['history']

def save_guardrails(rules):
    """
    Save guardrails for the application.
    
    Parameters:
    - rules: The guardrails to save.
    """
    global guardrails_list
    guardrails_list = rules 

if __name__ == "__main__":
    # Clear the system cache for ChromaDB
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Load environment variables
    load_dotenv(find_dotenv(), override=True)
    
    # Initialize Pinecone
    pc = Pinecone(os.environ.get("PINECONE_API_KEY"))

    # Define default guardrails
    default_rules = "Use only the information provided in the document.\n" + \
                    "Provide concise and accurate answers.\n" + \
                    "Do not include any external information or assumptions.\n"
    save_guardrails(default_rules)

    # Streamlit UI setup
    st.image('image.png')
    st.divider()
    st.subheader('LLM Question-Answering Application 🤖')
    
    # Retrieve AWS credentials from environment variables
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID') or ""
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY') or ""
    aws_session_token = os.environ.get('AWS_SESSION_TOKEN') or ""
    
    # Streamlit sidebar for user inputs
    with st.sidebar:
        # Input fields for AWS credentials
        access_key = st.text_input('AWS Access Key ID:', type='password', value=aws_access_key_id)
        secret_key = st.text_input('AWS Secret Access Key:', type='password', value=aws_secret_access_key)
        session_token = st.text_input('AWS Session Token:', type='password', value=aws_session_token)
        
        # Set environment variables if credentials are provided
        if access_key and secret_key:
            os.environ['AWS_ACCESS_KEY_ID'] = access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
            os.environ['AWS_SESSION_TOKEN'] = session_token

        # File uploader for document upload
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        # Input field for chunk size
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        # Input field for number of relevant documents to retrieve
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # Checkbox to enable or disable guardrails
        guardrails_on = st.checkbox('Enable Guardrails', value=True)
        if guardrails_on:
            st.write('Guardrails are enabled.')
            # Expander to edit guardrails
            with st.expander('Edit Guardrails'):
                rules = st.text_area('Guardrails', label_visibility="collapsed", value=default_rules, height=100)
                save_guardrails(rules if rules else "")
        else:
            st.write('Guardrails are disabled.')
            save_guardrails("")
            
        # Button to add data
        
        # Dropdown for role selection
        role = st.selectbox('Select Role', ['Incident Commander', 'Operational manager'])
        st.session_state['selected_role'] = role  # Store the selected role in session state

        add_data = st.button('Add Data', on_click=clear_history)

        # Process the uploaded file when the button is clicked
        if uploaded_file and add_data:
            if access_key and secret_key:
                # Read the uploaded file
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)

                # Save the uploaded file locally
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                # Load the document, chunk it, and create embeddings
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                filtered_chunks = filter_chunks_by_role(chunks, role)
                vector_store = create_embeddings(filtered_chunks)
                # Further processing with vector_store
                st.session_state.vs = vector_store
                st.success('File Uploaded, chunked and embedded successfully.')

    # Input field for asking a question about the uploaded document
    q = st.text_input('Ask a question about the content of your file:', disabled=access_key == "")
    if q:
        if access_key == "":
            st.error("Please enter a valid AWS Access Key.", icon="🚫")
        if 'vs' in st.session_state:
            # Retrieve the vector store from the session state
            vector_store = st.session_state.vs
            role = st.session_state.get('selected_role', None)  # Get the selected role from session state
            answer = ask_and_get_answer(vector_store, q, role, k)
            st.text_area('LLM Answer: ',value=answer)
    
            st.divider()

            # Maintain chat history in the session state
            if 'history' not in st.session_state:
                st.session_state.history = ''
            
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            # Display the chat history
            st.text_area(label="Chat History", value=h, key="history", height=400, placeholder="Search History is empty...")