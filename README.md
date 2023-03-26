# Chat with PDF

Chat with PDF is a Streamlit-based web application that allows users to interact with a chatbot that has access to the content of uploaded PDF documents. Users can ask questions related to the PDFs, and the chatbot will respond with relevant information extracted from the documents.

## Features

- Upload one or more PDF documents.
- Interact with the chatbot using natural language.
- Receive relevant information from the uploaded PDFs in the chatbot's responses.
- Customize the maximum response length for the chatbot.
- Clear conversation history and reset the context.

## Installation

To set up and run the application locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/FilCor/chat_with_pdf.git
cd chat_with_pdf

2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # For Linux and macOS
.\venv\Scripts\activate  # For Windows

3. Install the required dependencies:
pip install -r requirements.txt

4.Run the Streamlit app:
streamlit run app.py

The application should now be accessible at http://localhost:8501.

Usage
Enter your OpenAI API key in the "API Key" section in the sidebar.
Upload one or more PDF documents in the "Upload PDFs" section in the sidebar.
Adjust the maximum response length using the slider in the "Impostazioni" section in the sidebar.
Type your message or question in the input field and click "Submit" to interact with the chatbot.
To clear the conversation history, click the "Clear Conversation" button in the sidebar.
To manage the context, click the "Cancella Informazioni" button in the "Gestione Informazioni" section in the sidebar.
License
This project is licensed under the MIT License. See the LICENSE file for details.



