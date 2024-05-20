from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from speech_recognition import UnknownValueError, RequestError

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'
@app.route('/contactInfo')
def contactInf():
     return 'Contact Page'

# Endpoint to receive both JSON and audio data
@app.route('/audio', methods=["GET", 'POST'])
def receive_data():
    if request.method == 'POST':
        if request.headers['Content-Type'] == 'application/json':
            # If JSON data is received, extract the text
            data = request.get_json()
            print("Received data:", data)  # Print received JSON data
            transcribed_text = data.get('audio', '')
            
            ####################################################
            try:
                response = retrieval_chain.invoke({"input": transcribed_text})
                response_text = response["answer"]

                if "reset chat" in transcribed_text.lower():
                    data = ""
                    messages = ""
                
            except UnknownValueError:
                    return jsonify({'message': "I didn't catch that. Could you repeat?"}), 200
                    
            except RequestError:
                    return jsonify({'message': "There was an issue with the service; please try again later."}), 200
            
            print("Received transcribed text:", response_text)
            return jsonify({'message': response_text}), 200
            ####################################################

            
        elif request.headers['Content-Type'] == 'audio/wav':
            # If audio data is received, save it to a file
            audio_data = request.data
            with open('audio.wav', 'wb') as f:
                f.write(audio_data)
            print("Audio data saved to audio.wav")
            return jsonify({'message': 'Audio data received and saved successfully.'}), 200

        else:
            return jsonify({'error': 'Unsupported Content-Type.'}), 400
    else:
        return "ok!"

######################appointment#########################
def append_to_file(filename, data):
    with open(filename, 'a') as file:
        file.write(data)
        file.write('\n')

@app.route('/appointment', methods=['POST'])
def appointment():
    if request.method == 'POST':
        # Extract form data
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        date = request.form.get('date')
        time = request.form.get('time')
        area = request.form.get('area')
        city = request.form.get('city')
        state = request.form.get('state')
        post_code = request.form.get('post-code')

        # Construct data string
        data = f'{name}, {phone}, {email}, {date}, {time}, {area},  {city}, {state}, {post_code}'

        # Append data to file
        append_to_file("appointments.csv",data)

        return 'Appointment booked successfully!'
    
    #####################appointment#####################



    ####################contact##########################
@app.route('/contact', methods=['POST'])
def contact():
    if request.method == 'POST':
        # Extract form data
        name = request.form.get('name')
        phone = request.form.get('phone')
        message = request.form.get('message')

        # Construct data string
        data = f'{name}, {phone}, {message}'

        # Append data to file
        append_to_file('contacts.csv',data)

        return 'Message sent successfully!'


    ###################contact###########################

if __name__ == '__main__':
        # Initialize language processing components
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 300,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        }
    ]

    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings)

    genai.configure(api_key="AIzaSyCHJNAo2JGwSVJk72EbImyO-kGwHMhbIZQ")

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyCHJNAo2JGwSVJk72EbImyO-kGwHMhbIZQ")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                            google_api_key="AIzaSyCHJNAo2JGwSVJk72EbImyO-kGwHMhbIZQ")

    loader = PyPDFLoader("Data.pdf")
    pdf_text = loader.load()
    # print("Check 1")
    # print(pdf_text[:1000])
    # print("Check 2")
    # print(loader)
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    pages = loader.load_and_split(text_splitter)
    vectordb = Chroma.from_documents(pages, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    prompt_template = """Answer the question as specific as possible in a formal way , make sure to provide all the 
    specific details, if the answer is not in the provided context just say "I couldn't find any specific data on that. 
    Can you specify which professor or " "topic you are asking about?"\n\n 
    context:\n{context}?\n 
    input: \n{input}\n

    answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    app.run(debug=True)




