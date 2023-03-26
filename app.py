import streamlit as st
import openai
import PyPDF2
import io
import numpy as np
import faiss
from io import BytesIO
from wand.image import Image
from wand.color import Color
import pytesseract
import tiktoken
import re
from pdf2image import convert_from_path
import openai.api_resources.completion
from streamlit_chat import message


# Function to extract text from PDF
@st.cache_data
def extract_text_from_pdf(pdf_file):
    # Use BytesIO to read the file content directly
    with BytesIO(pdf_file.getbuffer()) as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        if len(text) == 0:
            # Convert BytesIO back to a temporary file for image extraction
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.getvalue())
                temp_pdf.flush()

            text = extract_text_from_pdf_image(temp_pdf.name)
            text = clean_text(text)
        else:
            text = clean_text(text)
    return text


def extract_text_from_pdf_image(pdf_file):
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_file, dpi=300)
        text = ''
            
        for image in images:
            # Extract text using OCR
            text += pytesseract.image_to_string(image, lang='ita', config='--psm 1') + '\n'

        return text

    except Exception as e:  # Replace FileNotFoundError with Exception
        print("Error:", e)


def clean_text(text):
    # rimuove spazi multipli
    text = re.sub(r'\s+', ' ', text)
    # rimuove caratteri di nuova linea e tabulazioni
    text = re.sub(r'[\n\t]', '', text)
    # rimuove caratteri non ASCII
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    # rimuove spazi all'inizio e alla fine del testo
    text = text.strip()
    return text

# Function to chunk text
def chunk_text(text, max_length=2000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_length:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Function to get embeddings
@st.cache_data
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Function to interact with chatbot
def interact_with_chatbot(context, message, max_response_length=200):
    prompt = f"{message}\n\n"
    for idx, chunk in enumerate(context):
        prompt += "basati sul contesto che ti viene dato qui di seguito: \n\n"
        prompt += f"Context-{idx + 1}: {chunk}\n\n"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Sei un bot che risponde alle domande che l'utente ti fa in base al contesto che ti viene dato, se la risposta non è disponibile nel contesto allora prova ad usare la tu abase di conoscenza, in questo caso però fai presente che l'informazione non proviene dal contesto dato."}, {"role": "user", "content": prompt}],
        max_tokens=max_response_length,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response['choices'][0]['message']['content'].strip()


# Main app
def main():
    
    st.set_page_config(page_title='Chat With PDF', layout='wide',
                   initial_sidebar_state=st.session_state.get('sidebar_state', 'expanded'))
    st.title("Chat with PDF")

    # Set OpenAI API key
    st.sidebar.title("API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    openai.api_key = api_key

    # Upload PDFs
    st.sidebar.title("Upload PDFs")
    pdf_files = st.sidebar.file_uploader("Select one or more PDFs", type="pdf", accept_multiple_files=True)

    #Settings
    st.sidebar.title("Impostazioni")
    max_response_length = st.sidebar.slider("Lunghezza massima della risposta", min_value=100, max_value=1000, value=500)

    # Clear Conversation Button
    st.sidebar.title("Clear Conversation")
    clear_conversation = st.sidebar.button("Clear Conversation")

    # Extract and store text from PDFs
    pdf_texts = {}
    for pdf_file in pdf_files:
        pdf_name = pdf_file.name
        pdf_texts[pdf_name] = extract_text_from_pdf(pdf_file)


    pdf_name_to_index_map = {}  # To store the mapping of PDF names to their index ranges in the vectorial database
    vectorial_db = {}  # To store embeddings along with the PDF names and text chunks

    index_offset = 0
    for pdf_name, text in pdf_texts.items():
        chunks = chunk_text(text)
        embeddings = [get_embedding(chunk) for chunk in chunks]
        vectorial_db[pdf_name] = {
            'chunks': chunks,
            'embeddings': embeddings,
        }

        pdf_name_to_index_map[pdf_name] = {
            'start_idx': index_offset,
            'end_idx': index_offset + len(embeddings) - 1
        }
        index_offset += len(embeddings)

    # Index embeddings in a vectorial database

    if vectorial_db:  # Add this check to ensure vectorial_db is not empty
        embedding_size = len(vectorial_db[list(vectorial_db.keys())[0]]['embeddings'][0])
        index = faiss.IndexFlatL2(embedding_size)
        for pdf_name, data in vectorial_db.items():
            for embedding in data['embeddings']:
                index.add(np.array([embedding]))

    # Chatbot interaction
    # Initialize the conversation history
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = st.text_input("Type your message here")
    submit_button = st.button("Submit")
    
    if submit_button and user_input:
        context = []  # Initialize an empty context list

        if vectorial_db:  # Add this check to ensure vectorial_db is not empty
        # Query vectorial database
            user_embedding = get_embedding(user_input)
            D, I = index.search(np.array([user_embedding]), 5)

            # Set context
            for idx in I[0]:
                for pdf_name, index_range in pdf_name_to_index_map.items():
                    if index_range['start_idx'] <= idx <= index_range['end_idx']:
                        context.append(vectorial_db[pdf_name]['chunks'][idx - index_range['start_idx']])
                        break

        response = interact_with_chatbot(context, user_input, max_response_length)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

    # Manage context
    st.sidebar.title("Gestione Informazioni")
    clear_context = st.sidebar.button("Cancella Informazioni")
    if clear_context:
        context = []

    if clear_conversation:
        st.session_state['past'] = []
        st.session_state['generated'] = []

if __name__ == "__main__":
    main()
