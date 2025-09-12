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
import unicodedata

def deep_clean_text(text):
    """
    Performs a deep cleaning of the text to remove all non-printable and problematic
    Unicode characters that can cause crashes in PDF generation libraries.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    cleaned_chars = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith(('L', 'N', 'P', 'S', 'Z')):
            cleaned_chars.append(char)
    return "".join(cleaned_chars)

def create_download_pdf(report_content, topic="AI Research Report"):
    """
    Generates a Unicode-aware PDF file and returns a 'bytes' object, which is the
    correct data type for Streamlit's download button.
    """
    pdf = FPDF()
    pdf.add_page()
    
    try:
        # Assumes a subfolder named 'dejavu-sans' exists with the font files
        pdf.add_font('DejaVu', '', 'dejavu-sans/DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'dejavu-sans/DejaVuSans-Bold.ttf', uni=True)
        font_family = 'DejaVu'
    except RuntimeError:
        st.error("CRITICAL: DejaVu font files not found in the 'dejavu-sans' subfolder. Please ensure the folder and .ttf files are present.")
        return None

    cleaned_topic = deep_clean_text(topic)
    pdf.set_font(font_family, 'B', 18)
    pdf.multi_cell(0, 10, cleaned_topic, 0, 'C')
    pdf.ln(10)

    pdf.set_font(font_family, '', 12)
    lines = report_content.split('\n')
    
    for line in lines:
        cleaned_line = deep_clean_text(line)
        if not cleaned_line.strip():
            pdf.ln(3)
            continue
        if cleaned_line.startswith('# '):
            continue
        elif cleaned_line.startswith('## '):
            pdf.set_font(font_family, 'B', 14)
            pdf.multi_cell(0, 8, cleaned_line.replace('## ', ''))
            pdf.ln(2)
            pdf.set_font(font_family, '', 12)
        else:
            pdf.multi_cell(0, 6, cleaned_line)
            pdf.ln(1)

    # --- THE DEFINITIVE FIX FOR THE bytearray ERROR ---
    # Explicitly cast the output to 'bytes' to guarantee the correct data type for Streamlit.
    return bytes(pdf.output())

def process_pdf(pdf_files):
    if not pdf_files: return []
    full_text = ""
    st.write("Extracting text and performing OCR on images...")
    progress_bar = st.progress(0)
    for i, pdf_file in enumerate(pdf_files):
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text()
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
                    except Exception:
                        pass
            doc.close()
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {e}")
        progress_bar.progress((i + 1) / len(pdf_files))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text=full_text)

def get_vector_store(text_chunks):
    if not text_chunks: return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def create_rag_chain(vector_store, groq_api_key, prompt_template):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

def main():
    st.set_page_config(page_title="Advanced RAG Report Generator", layout="wide")
    st.title("📚 Advanced AI Research Report Generator")
    st.markdown("""
    Welcome! This tool generates detailed research reports with PDF download and OCR capabilities.
    1.  **RAG Mode:** Upload PDFs to generate a report from their content.
    2.  **Generative Mode:** Leave the file uploader empty and provide a topic to generate a report from scratch.
    """)

    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
        page_count = st.number_input("Desired Number of Pages:", min_value=1, max_value=200, value=10, step=1)
        user_instructions = st.text_area("Optional Instructions:", height=150, placeholder="e.g., 'Focus on financial analysis.'")

    st.subheader("Step 1: Provide Content Source")
    report_topic = st.text_input("Enter a Topic (if not uploading files)", placeholder="e.g., 'The Future of Renewable Energy'")
    uploaded_files = st.file_uploader("OR Upload PDF Document(s) with OCR (Optional)", type="pdf", accept_multiple_files=True)

    if st.button("Generate Detailed Report"):
        if not groq_api_key: st.error("A Groq API Key is required."); return
        if not uploaded_files and not report_topic: st.error("Please upload PDFs or enter a topic."); return

        # --- RAG WORKFLOW ---
        if uploaded_files:
            with st.spinner("Processing documents with OCR and generating RAG report..."):
                start_time = time.time()
                
                text_chunks = process_pdf(uploaded_files)
                if not text_chunks:
                    st.error("Could not extract any text from the PDFs.")
                    return

                st.write("Step 2/5: Creating vector store...")
                vector_store = get_vector_store(text_chunks)
                if vector_store is None: return
                
                instruction_addition = f"USER INSTRUCTIONS: \"{user_instructions}\"" if user_instructions else ""

                st.write("Step 3/5: Generating a detailed Table of Contents...")
                outline_prompt = f"""Based on the provided context, generate a detailed table of contents for a research report of about {page_count} pages. {instruction_addition} CONTEXT: {{context}} QUERY: {{input}}"""
                outline_chain = create_rag_chain(vector_store, groq_api_key, outline_prompt)
                toc = outline_chain.invoke({"input": "Generate a table of contents."})['answer']

                with st.expander("Generated Table of Contents", expanded=True): st.markdown(toc)

                st.write("Step 4/5: Generating content for each section...")
                section_titles = re.findall(r'^\s*[\d\w]+\..*$', toc, re.MULTILINE)
                if not section_titles: section_titles = [line.strip() for line in toc.split('\n') if line.strip() and len(line.strip()) > 5]
                
                full_report = []
                words_per_section = (page_count * 400) // len(section_titles) if section_titles else 500
                
                section_prompt = f"""As a research analyst, write a comprehensive section on: "{{input}}". Write about {words_per_section} words. Use the CONTEXT to elaborate. {instruction_addition} DO NOT write a title. CONTEXT: {{context}} QUERY: {{input}}"""
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
                pdf_bytes = create_download_pdf(final_report, "RAG-Generated Research Report")
                if pdf_bytes:
                    st.download_button(label="Download Report as PDF", data=pdf_bytes, file_name="rag_report.pdf", mime="application/pdf")
        
        # --- GENERATIVE WORKFLOW ---
        else:
            with st.spinner(f"Generating a new report on '{report_topic}'..."):
                start_time = time.time()
                llm = ChatGroq(groq_api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
                instruction_addition = f"USER INSTRUCTIONS: \"{user_instructions}\"" if user_instructions else ""

                st.write("Step 1/3: Generating a detailed Table of Contents...")
                outline_prompt = PromptTemplate.from_template(f"Generate a detailed table of contents for a report of about {page_count} pages on the topic: '{{topic}}'. {instruction_addition}")
                outline_chain = outline_prompt | llm
                toc = outline_chain.invoke({"topic": report_topic}).content

                with st.expander("Generated Table of Contents", expanded=True): st.markdown(toc)

                st.write("Step 2/3: Generating content for each section...")
                section_titles = re.findall(r'^\s*[\d\w]+\..*$', toc, re.MULTILINE)
                if not section_titles: section_titles = [line.strip() for line in toc.split('\n') if line.strip() and len(line.strip()) > 5]

                full_report = []
                words_per_section = (page_count * 400) // len(section_titles) if section_titles else 500

                section_prompt = PromptTemplate.from_template(f"As a research analyst, write a comprehensive section on: '{{section}}'. The main report topic is '{report_topic}'. Write about {words_per_section} words. {instruction_addition} DO NOT write a title.")
                section_chain = section_prompt | llm

                progress_bar = st.progress(0, text="Generating report sections...")
                for i, title in enumerate(section_titles):
                    clean_title = re.sub(r'^\s*[\d\w]+\.\s*', '', title).strip()
                    st.write(f"-> Generating: {clean_title}")
                    response = section_chain.invoke({"section": clean_title})
                    full_report.append(f"## {title}\n\n{response.content}")
                    progress_bar.progress((i + 1) / len(section_titles), text=f"Generated: {clean_title}")

                st.write("Step 3/3: Compiling and generating PDF...")
                final_report = f"# Your In-Depth Research Report: {report_topic}\n\n" + "\n\n".join(full_report)
                
                st.subheader("🎉 Your Detailed Research Report is Ready!")
                st.markdown(final_report)
                pdf_bytes = create_download_pdf(final_report, report_topic)
                if pdf_bytes:
                    st.download_button(label="Download Report as PDF", data=pdf_bytes, file_name="generated_report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
