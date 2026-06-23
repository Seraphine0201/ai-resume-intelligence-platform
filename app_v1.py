import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="AI Resume Intelligence Platform")

st.title("AI Resume Intelligence Platform")

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file:

    st.success("Resume uploaded successfully!")

    pdf_reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf_reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text + "\n"

    st.subheader("Extracted Resume Text")

    st.text_area(
        "Resume Content",
        resume_text,
        height=300
    )

    job_description = st.text_area(
        "Paste Job Description",
        height=250
    )

    if st.button("Analyze Resume"):

        if not job_description:
            st.warning("Please paste a Job Description")

        else:

            with st.spinner("Analyzing Resume..."):

                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0
                )

                prompt = f"""
                Analyze this resume against the job description.

                Provide:

                1. Match Score
                2. Matching Skills
                3. Missing Skills
                4. Resume Improvements
                5. Interview Questions

                Resume:
                {resume_text}

                Job Description:
                {job_description}
                """

                response = llm.invoke(prompt)

                st.subheader("Analysis Result")

                st.write(response.content)