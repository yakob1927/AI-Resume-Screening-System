
import streamlit as st
import pandas as pd
import re
import plotly.express as px

from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)


# ---------------- DARK THEME CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label {
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background-color: #172238;
    border: 1px solid #293952;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    color: #60a5fa;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #172238;
    border-radius: 10px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SKILLS DATABASE ----------------

SKILLS = [
    "python", "java", "c++", "sql",
    "machine learning", "deep learning",
    "nlp", "data science", "data analysis",
    "pandas", "numpy", "power bi",
    "excel", "tensorflow", "pytorch",
    "scikit-learn", "html", "css",
    "javascript", "flask", "django",
    "streamlit", "aws", "git",
    "statistics", "tableau", "mysql"
]


# ---------------- FUNCTIONS ----------------

def extract_resume_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text

    except Exception:
        return ""


def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


def calculate_match(resume_text, job_description):

    documents = [
        resume_text.lower(),
        job_description.lower()
    ]

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return round(similarity * 100, 2)


def analyze_resumes(uploaded_files, job_description):

    results = []

    job_skills = set(extract_skills(job_description))

    for pdf in uploaded_files:

        resume_text = extract_resume_text(pdf)

        if not resume_text.strip():
            continue

        resume_skills = set(extract_skills(resume_text))

        matched = sorted(job_skills.intersection(resume_skills))
        missing = sorted(job_skills - resume_skills)

        score = calculate_match(
            resume_text,
            job_description
        )

        results.append({
            "Candidate": pdf.name,
            "Match Score (%)": score,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing),
            "Total Skills": len(resume_skills)
        })

    return results


# ---------------- SESSION STATE ----------------

if "results" not in st.session_state:
    st.session_state.results = []

if "job_description" not in st.session_state:
    st.session_state.job_description = ""


# ---------------- SIDEBAR ----------------

st.sidebar.title("🤖 AI Recruiter")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Resume Analysis",
        "Candidate Ranking",
        "Charts & Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI Resume Screening System")
st.sidebar.caption("Built with Python & NLP")


# ---------------- MAIN HEADER ----------------

st.title("📄 AI-Based Resume Screening System")

st.markdown(
    "Smart resume analysis and candidate matching platform."
)

st.markdown("---")


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Dashboard Overview")

    results = st.session_state.results

    total = len(results)

    average = (
        round(sum(r["Match Score (%)"] for r in results) / total, 2)
        if total else 0
    )

    best = max(
        results,
        key=lambda x: x["Match Score (%)"],
        default=None
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Resumes", total)

    col2.metric("Average Match Score", f"{average}%")

    col3.metric(
        "Top Candidate Score",
        f"{best['Match Score (%)']}%" if best else "0%"
    )

    st.markdown("---")

    st.subheader("Welcome to AI Recruiter")

    st.write(
        "Upload candidate resumes and compare them with "
        "your job description using NLP-based text similarity."
    )

    if results:

        st.subheader("Recent Candidate Results")

        df = pd.DataFrame(results)

        st.dataframe(
            df[["Candidate", "Match Score (%)", "Total Skills"]],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No resumes analyzed yet. Go to Resume Analysis to begin."
        )


# ---------------- RESUME ANALYSIS ----------------

elif page == "Resume Analysis":

    st.subheader("Resume Analysis")

    job_description = st.text_area(
        "Enter Job Description",
        height=180,
        placeholder="Example: Python developer with SQL, Machine Learning, Pandas and Data Analysis skills."
    )

    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Analyze Resumes", use_container_width=True):

        if not job_description.strip():
            st.error("Please enter a job description.")

        elif not uploaded_files:
            st.error("Please upload at least one resume.")

        else:

            with st.spinner("Analyzing candidate resumes..."):

                results = analyze_resumes(
                    uploaded_files,
                    job_description
                )

                st.session_state.results = results
                st.session_state.job_description = job_description

            if results:
                st.success("Resume analysis completed successfully!")

                st.info(
                    f"Successfully analyzed {len(results)} readable resumes."
                )

            else:
                st.error(
                    "Could not extract readable text from the uploaded PDFs."
                )

    if st.session_state.results:

        st.markdown("---")
        st.subheader("Analysis Results")

        df = pd.DataFrame(st.session_state.results)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


# ---------------- CANDIDATE RANKING ----------------

elif page == "Candidate Ranking":

    st.subheader("🏆 Candidate Ranking")

    results = st.session_state.results

    if results:

        df = pd.DataFrame(results)

        df = df.sort_values(
            by="Match Score (%)",
            ascending=False
        ).reset_index(drop=True)

        df.insert(0, "Rank", range(1, len(df) + 1))

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        best = df.iloc[0]

        st.markdown("---")

        st.subheader("🥇 Highest Similarity Score")

        st.success(
            f"Top-ranked resume: {best['Candidate']}"
        )

        st.metric(
            "Text Similarity",
            f"{best['Match Score (%)']}%"
        )

        st.write("Matched Skills:")
        st.write(best["Matched Skills"] or "No matching skills found.")

        st.write("Missing Skills:")
        st.write(best["Missing Skills"] or "None")

    else:

        st.info(
            "Analyze resumes first to see candidate rankings."
        )


# ---------------- CHARTS AND REPORTS ----------------

elif page == "Charts & Reports":

    st.subheader("📊 Charts & Reports")

    results = st.session_state.results

    if results:

        df = pd.DataFrame(results)

        col1, col2 = st.columns(2)

        with col1:

            st.write("Candidate Match Score Comparison")

            fig = px.bar(
                df,
                x="Candidate",
                y="Match Score (%)",
                color="Match Score (%)",
                title="Resume Similarity Scores",
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="#0b1120",
                plot_bgcolor="#172238",
                xaxis_title="Candidate",
                yaxis_title="Similarity (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            st.write("Candidate Skill Count")

            skill_data = pd.DataFrame({
                "Candidate": df["Candidate"],
                "Skills": df["Total Skills"]
            })

            fig2 = px.pie(
                skill_data,
                names="Candidate",
                values="Skills",
                title="Skills Distribution",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("Download Candidate Report")

        df.insert(0, "Rank", range(1, len(df) + 1))

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download CSV Report",
            data=csv,
            file_name="AI_Resume_Screening_Report.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "No report available. Analyze resumes first."
        )


# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "AI Resume Screening System | Python | NLP | Scikit-learn | Streamlit"
)

st.caption(
    "Educational prototype. Human review is required for hiring decisions."
)
