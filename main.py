from transformers import pipeline
import streamlit as st
import os
import io
import docx
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    text = [paragraph.text for paragraph in doc.paragraphs]
    return " ".join(text)

def load_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.docx')]
    return files

def chunk_text(text, max_length=1024, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

def answer_question(nlp, question, chunk, page_num):
    QA_input = {'question': question, 'context': chunk}
    result = nlp(QA_input, max_answer_len=300, top_k=3)  # Increase max_answer_len for longer answers
    for res in result:
        res['page_num'] = page_num
    return result

def main():
    st.title("Welcome to CurioQueries for Engineers")

    page = st.sidebar.selectbox("Select a page", ["Home", "Ask"])

    if page == "Home":
        st.write("This is the main page. Explore and have fun!")

    elif page == "Ask":
        st.header("Ask Your Question")

        directory = st.text_input("Enter the directory containing the .docx files")
        question = st.text_input('Enter your question')

        if not directory and not question.strip():
            st.warning("Please enter a directory and write your question in the question field.")
        elif not directory:
            st.warning("Please enter a directory.")
        elif not question.strip():
            st.warning("This column cannot be empty. Please write your question in the question field.")
        else:
            if not os.path.exists(directory):
                st.error("The specified directory does not exist.")
            else:
                files = load_files(directory)
                if not files:
                    st.warning("No .docx files found in the specified directory.")
                else:
                    # Use a smaller but efficient model
                    model_name = "distilbert-base-uncased-distilled-squad"
                    logger.info(f"Loading model: {model_name}")
                    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

                    answers = []
                    logger.info("Processing files")
                    for file_name in files:
                        file_path = os.path.join(directory, file_name)
                        with open(file_path, 'rb') as file:
                            context = read_docx(file)
                            chunks = chunk_text(context)

                            with ThreadPoolExecutor() as executor:
                                futures = [executor.submit(answer_question, nlp, question, chunk, i+1) for i, chunk in enumerate(chunks)]
                                for future in futures:
                                    results = future.result()
                                    for res in results:
                                        score = res['score']
                                        answer = res['answer']
                                        page_num = res['page_num']
                                        answers.append((file_name, answer, score, page_num))

                    logger.info("Sorting answers")
                    # Sort answers by score in descending order and take top 3
                    top_answers = sorted(answers, key=lambda x: x[2], reverse=True)[:3]

                    st.write("Top Answers found:")
                    for file_name, answer, score, page_num in top_answers:
                        st.write(f"**{file_name} (Page {page_num})**: {answer} (Score: {score})")

if __name__ == '__main__':
    main()
