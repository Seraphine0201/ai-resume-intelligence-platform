import streamlit as st
import fitz
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()

def create_pdf(content):
    pdf_file = "resume_analysis.pdf"
    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    story = [Paragraph(content.replace("\n", "<br/>"), styles['BodyText'])]
    doc.build(story)
    return pdf_file

st.set_page_config(page_title="AI Resume Intelligence Platform")

st.title("AI Resume Intelligence Platform")
st.markdown("""
Analyze resumes against job descriptions using:

- RAG (Retrieval-Augmented Generation)
- ChromaDB Vector Database
- Sentence Transformers
- Gemini 2.5 Flash

Upload your resume and paste a job description to get ATS-style insights.
""")

#Temperature slider
temperature = st.sidebar.slider(
    "LLM Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)

#top p slider
top_p = st.sidebar.slider(
    "Top P",
    min_value=0.0,
    max_value=1.0,
    value=0.95,
    step=0.05
)

#top k slider
top_k = st.sidebar.slider(
    "Top K",
    min_value=1,
    max_value=100,
    value=40,
    step=1
)

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file:

    st.success("Resume uploaded successfully!")

    # Read pdf
    # Read PDF using PyMuPDF
    doc = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )
    resume_text = ""
    for page in doc:
        resume_text += page.get_text()

    #chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(resume_text) 
 

    #embeddings
    embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
    )
    embeddings = embedding_model.encode(chunks)
   

    #chromadb
    client = chromadb.Client()

    client = chromadb.Client()

    collection = client.get_or_create_collection(
    name="resume_collection"
)
    collection = client.get_or_create_collection(
        name="resume_collection"
    )
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[str(i)]
    )
    # st.write("Number of chunks:", len(chunks))       
    # st.write("Embedding Shape:", embeddings.shape)
    # st.write("Stored in ChromaDB:", collection.count())

    #Resume preview
    st.subheader("Extracted Resume Text")
    st.text_area(
        "Resume Content",
        resume_text,
        height=300
    )

    #jd input
    job_description = st.text_area(
        "Paste Job Description",
        height=250
    )

    if st.button("Analyze Resume"):

        if not job_description:
            st.warning("Please paste a Job Description")
        else:
            #Retrieval    
            jd_embedding = embedding_model.encode(job_description)

            results = collection.query(
                query_embeddings=[jd_embedding.tolist()],
                n_results=5
           )

            relevant_chunks = "\n\n".join(
                results["documents"][0]
            )

            st.subheader("Retrieved Resume Chunks")
            with st.expander("View Retrieved Resume Chunks"):
                st.write(relevant_chunks)
        

            with st.spinner("Analyzing Resume..."):

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )

                prompt = f"""
                Analyze this resume against the job description.

                Provide:

                1. ATS Score (0-100) with explanation
                2. Matching Skills
                3. Missing Skills
                4. Resume Improvements
                5. Interview Questions
                6. Learning Roadmap

                Relevant Resume Information:
                {relevant_chunks}

                Job Description:
                {job_description}
                """

                response = llm.invoke(prompt)

                st.subheader("Analysis Result")

                st.markdown(response.content)
                pdf_file = create_pdf(response.content)

                with open(pdf_file, "rb") as pdf:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf,
                        file_name="resume_analysis.pdf",
                        mime="application/pdf"
                   )