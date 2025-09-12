import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import re
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from fpdf import FPDF

# --- New Function to Create PDF ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Generated Research Report', 0, 0, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_download_pdf(report_content):
    """
    Generates a PDF file from the report string and returns its bytes.
    """
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    lines = report_content.split('\n')
    for line in lines:
        if line.startswith('## '):
            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 8, line.replace('## ', ''))
            pdf.ln(4)
            pdf.set_font("Arial", size=12)
        elif line.startswith('# '):
            pdf.set_font("Arial", 'B', 18)
            pdf.multi_cell(0, 10, line.replace('# ', ''))
            pdf.ln(6)
            pdf.set_font("Arial", size=12)
        else:
            pdf.multi_cell(0, 6, line)
            if line.strip():
                pdf.ln(3)
                
    return pdf.output(dest='S').encode('latin-1')


# --- Updated PDF Processor with OCR ---
def process_pdf(pdf_files):
    """
    Reads PDFs, extracts native text, and performs OCR on images.
    Args:
        pdf_files: A list of uploaded PDF file objects.
    Returns:
        A list of text chunks.
    """
    if not pdf_files:
        return []
    
    full_text = ""
    st.write("Extracting text and performing OCR on images...")
    progress_bar = st.progress(0)
    
    for i, pdf_file in enumerate(pdf_files):
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 1. Extract native text
                full_text += page.get_text()
                
                # 2. Extract images and perform OCR
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        ocr_text = pytesseract.image_to_string(image)
                        if ocr_text.strip():
                            full_text += f"\n[OCR Text from Image on page {page_num + 1}]:\n{ocr_text}\n"
                    except Exception as e:
                        st.warning(f"Could not process image {img_index+1} on page {page_num+1} in {pdf_file.name}. Skipping. Error: {e}")

            doc.close()
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {e}")
        progress_bar.progress((i + 1) / len(pdf_files))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=full_text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks using HuggingFace embeddings.
    Args:
        text_chunks: A list of text chunks.
    Returns:
        A FAISS vector store instance.
    """
    if not text_chunks:
        return None
        
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        st.info("This may be due to a network issue or missing model files. Please check your connection.")
        return None


def create_rag_chain(vector_store, groq_api_key, prompt_template):
    """
    Creates and returns a RAG chain for document retrieval and generation.
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain


def main():
    """
    The main function that runs the Streamlit application.
    """
    st.set_page_config(page_title="Advanced RAG Report Generator", layout="wide")

    st.title("📚 Advanced AI Research Report Generator")
    st.markdown("""
    Welcome! This tool generates detailed research reports with PDF download and OCR capabilities.
    1.  **RAG Mode:** Upload PDFs (including scanned documents) to generate a report from their content.
    2.  **Generative Mode:** Leave the file uploader empty and provide a topic to generate a report from scratch.
    """)

    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
        page_count = st.number_input("Enter Desired Number of Pages:", min_value=1, max_value=200, value=10, step=1)
        
        user_instructions = st.text_area("Optional: Specific Instructions for the Report", 
                                         height=150,
                                         placeholder="e.g., 'Focus on financial analysis.', 'Write in an academic tone.'")

        if not groq_api_key:
            st.warning("Please enter your Groq API key to proceed.")
        
        st.markdown("---")
        st.info("Your API key is used solely for this session and is not stored.")

    st.subheader("Step 1: Provide Content Source")
    
    report_topic = st.text_input("Enter a Topic (if not uploading files)", placeholder="e.g., 'The Future of Renewable Energy'")
    
    uploaded_files = st.file_uploader("OR Upload PDF Document(s) with OCR (Optional)", type="pdf", accept_multiple_files=True)

    if st.button("Generate Detailed Report"):
        if not groq_api_key:
            st.error("Cannot proceed without a Groq API Key.")
            return
        if not uploaded_files and not report_topic:
            st.error("Please either upload PDF documents or enter a topic to generate a report.")
            return

        if uploaded_files:
            with st.spinner("Processing documents with OCR and generating RAG report... This will take some time."):
                try:
                    start_time = time.time()
                    
                    text_chunks = process_pdf(uploaded_files)
                    if not text_chunks:
                        st.error("Could not extract any text from the PDFs. They might be empty, corrupted, or OCR failed.")
                        return

                    st.write("Step 2/5: Creating vector store...")
                    vector_store = get_vector_store(text_chunks)
                    if vector_store is None: return
                    
                    instruction_addition = f"IMPORTANT USER INSTRUCTIONS: \"{user_instructions}\"" if user_instructions else ""

                    st.write("Step 3/5: Generating a detailed Table of Contents...")
                    outline_prompt = f"""Based on the provided document context, generate a detailed, multi-level table of contents for a research report of about {page_count} pages. {instruction_addition} Do not write the report, ONLY the table of contents. CONTEXT: {{context}} QUERY: {{input}}"""
                    outline_chain = create_rag_chain(vector_store, groq_api_key, outline_prompt)
                    toc = outline_chain.invoke({"input": "Generate a table of contents."})['answer']

                    with st.expander("Generated Table of Contents", expanded=True): st.markdown(toc)

                    st.write("Step 4/5: Generating content for each section...")
                    section_titles = re.findall(r'^\s*[\d\w]+\..*$', toc, re.MULTILINE)
                    if not section_titles: section_titles = [line.strip() for line in toc.split('\n') if line.strip() and len(line.strip()) > 5]
                    
                    full_report = []
                    words_per_section = (page_count * 400) // len(section_titles) if section_titles else 500
                    
                    section_prompt = f"""As a research analyst, write a comprehensive section on: "{{input}}". Write about {words_per_section} words. Use the CONTEXT to elaborate and provide insights. {instruction_addition} Ensure the analysis is strictly derived from the context. DO NOT write a title. CONTEXT: {{context}} QUERY: {{input}}"""
                    section_chain = create_rag_chain(vector_store, groq_api_key, section_prompt)
                    
                    progress_bar = st.progress(0, text="Generating report sections...")
                    for i, title in enumerate(section_titles):
                        clean_title = re.sub(r'^\s*[\d\w]+\.\s*', '', title).strip()
                        st.write(f"-> Generating: {clean_title}")
                        response = section_chain.invoke({"input": clean_title})
                        full_report.append(f"## {title}\n\n{response['answer']}")
                        progress_bar.progress((i + 1) / len(section_titles), text=f"Generated: {clean_title}")

                    st.write("Step 5/5: Compiling and generating PDF...")
                    final_report = "# Your In-Depth Research Report\n\n" + "\n\n".join(full_report)
                    
                    st.subheader("🎉 Your Detailed Research Report is Ready!")
                    st.markdown(final_report)
                    st.success(f"Full report generated in {time.time() - start_time:.2f} seconds.")
                    
                    pdf_bytes = create_download_pdf(final_report)
                    st.download_button(label="Download Report as PDF", data=pdf_bytes, file_name="rag_report.pdf", mime="application/pdf")

                except Exception as e:
                    st.error(f"An unexpected error occurred during RAG generation: {e}")

        else: # Generative Workflow
            with st.spinner(f"Generating a new report on '{report_topic}'... This will take some time."):
                try:
                    start_time = time.time()
                    llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
                    instruction_addition = f"IMPORTANT USER INSTRUCTIONS: \"{user_instructions}\"" if user_instructions else ""

                    st.write("Step 1/3: Generating a detailed Table of Contents...")
                    outline_prompt = PromptTemplate.from_template(f"Generate a detailed, multi-level table of contents for a research report of about {page_count} pages on the topic: '{{topic}}'. {instruction_addition} Do not write the report itself, ONLY the table of contents.")
                    outline_chain = outline_prompt | llm
                    toc = outline_chain.invoke({"topic": report_topic}).content

                    with st.expander("Generated Table of Contents", expanded=True): st.markdown(toc)

                    st.write("Step 2/3: Generating content for each section...")
                    section_titles = re.findall(r'^\s*[\d\w]+\..*$', toc, re.MULTILINE)
                    if not section_titles: section_titles = [line.strip() for line in toc.split('\n') if line.strip() and len(line.strip()) > 5]

                    full_report = []
                    words_per_section = (page_count * 400) // len(section_titles) if section_titles else 500

                    section_prompt = PromptTemplate.from_template(f"As a research analyst, write a comprehensive and in-depth report section on the topic: '{{section}}'. The main topic is '{report_topic}'. Write about {words_per_section} words. {instruction_addition} Be insightful and well-structured. DO NOT write a title. CONTEXT: {{context}} QUERY: {{input}}")
                    section_chain = section_prompt | llm

                    progress_bar = st.progress(0, text="Generating report sections...")
                    for i, title in enumerate(section_titles):
                        clean_title = re.sub(r'^\s*[\d\w]+\.\s*', '', title).strip()
                        st.write(f"-> Generating: {clean_title}")
                        response = section_chain.invoke({"section": clean_title, "context": "", "input": clean_title}) # Context and input are dummy here
                        full_report.append(f"## {title}\n\n{response.content}")
                        progress_bar.progress((i + 1) / len(section_titles), text=f"Generated: {clean_title}")

                    st.write("Step 3/3: Compiling and generating PDF...")
                    final_report = f"# Your In-Depth Research Report: {report_topic}\n\n" + "\n\n".join(full_report)

                    st.subheader("🎉 Your Detailed Research Report is Ready!")
                    st.markdown(final_report)
                    st.success(f"Full report generated in {time.time() - start_time:.2f} seconds.")
                    
                    pdf_bytes = create_download_pdf(final_report)
                    st.download_button(label="Download Report as PDF", data=pdf_bytes, file_name="generated_report.pdf", mime="application/pdf")

                except Exception as e:
                    st.error(f"An unexpected error occurred during generative process: {e}")

if __name__ == "__main__":
    # Check for Tesseract installation
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        st.error("Tesseract OCR is not installed or not in your PATH. OCR functionality will not work.")
        st.error("Please see installation instructions: https://tesseract-ocr.github.io/tessdoc/Installation.html")
    main()

