#!/bin/bash
set -e

mkdir -p src/agents src/parsers src/vectorstore src/api data/resumes

touch src/__init__.py
touch src/agents/__init__.py
touch src/parsers/__init__.py
touch src/vectorstore/__init__.py
touch src/api/__init__.py

# ---------- cv_parser.py ----------
cat > src/parsers/cv_parser.py << 'PYEOF'
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

EXTRACT_PROMPT = """You are a CV parsing assistant. Extract structured data from the CV text below.

Return ONLY a valid JSON object. No markdown, no explanation, no code fences.

The JSON must have exactly these keys:
- "name": string (candidate full name)
- "email": string (email address or empty string)
- "skills": list of strings (technical and soft skills)
- "experience_years": number (total years of professional experience, 0 if unknown)
- "work_history": list of objects, each with keys "company", "role", "duration"
- "education": list of strings (degrees, institutions)

CV TEXT:
{cv_text}

JSON:"""


class CVParser:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.llm = OllamaLLM(model=model_name, base_url=base_url, temperature=0.1)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(EXTRACT_PROMPT)

    def load_pdf(self, file_path: str) -> str:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    def _clean_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def extract_structured_data(self, cv_text: str) -> dict:
        chain = self.prompt | self.llm
        raw = chain.invoke({"cv_text": cv_text[:6000]})
        try:
            return json.loads(self._clean_json(raw))
        except json.JSONDecodeError as e:
            return {
                "parse_error": True, "error": str(e), "raw_response": raw[:500],
                "name": "", "email": "", "skills": [], "experience_years": 0,
                "work_history": [], "education": [],
            }

    def parse(self, file_path: str) -> dict:
        cv_text = self.load_pdf(file_path)
        structured = self.extract_structured_data(cv_text)
        chunks = self.splitter.split_text(cv_text)
        return {"structured": structured, "chunks": chunks, "full_text": cv_text}
PYEOF

# ---------- supabase_store.py ----------
cat > src/vectorstore/supabase_store.py << 'PYEOF'
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

load_dotenv()


class RecruitmentVectorStore:
    def __init__(self):
        self.supabase: Client = create_client(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"],
        )
        self.embeddings = OllamaEmbeddings(
            model=os.environ["OLLAMA_EMBED_MODEL"],
            base_url=os.environ["OLLAMA_BASE_URL"],
        )
        self.vector_store = SupabaseVectorStore(
            client=self.supabase, embedding=self.embeddings,
            table_name="documents", query_name="match_documents",
        )

    def add_cv_chunks(self, candidate_id: str, chunks: list):
        docs = [Document(page_content=c, metadata={"candidate_id": candidate_id, "type": "cv"}) for c in chunks]
        self.vector_store.add_documents(docs)

    def add_job_description(self, job_id: str, description: str):
        doc = Document(page_content=description, metadata={"job_id": job_id, "type": "job"})
        self.vector_store.add_documents([doc])

    def search_candidates(self, job_description: str, k: int = 10):
        return self.vector_store.similarity_search(job_description, k=k, filter={"type": "cv"})

    def search_jobs(self, cv_text: str, k: int = 5):
        return self.vector_store.similarity_search(cv_text, k=k, filter={"type": "job"})
PYEOF

# ---------- recruiter_agent.py ----------
cat > src/agents/recruiter_agent.py << 'PYEOF'
import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.parsers.cv_parser import CVParser
from src.vectorstore.supabase_store import RecruitmentVectorStore

load_dotenv()


class RecruitmentAgent:
    def __init__(self, model_name=None):
        model = model_name or os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        self.llm = OllamaLLM(model=model, base_url=base_url, temperature=0.3)
        self.cv_parser = CVParser(model_name=model, base_url=base_url)
        self.vector_store = RecruitmentVectorStore()
        self.supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    def process_cv_upload(self, file_path, candidate_name, email):
        parsed = self.cv_parser.parse(file_path)
        candidate_data = {"name": candidate_name, "email": email, "parsed_data": parsed["structured"]}
        result = self.supabase.table("candidates").insert(candidate_data).execute()
        candidate_id = result.data[0]["id"]
        self.vector_store.add_cv_chunks(candidate_id, parsed["chunks"])
        return {"candidate_id": candidate_id, "structured_data": parsed["structured"]}

    def create_job(self, title, description, skills, experience):
        result = self.supabase.table("jobs").insert({
            "title": title, "description": description,
            "required_skills": skills, "experience_level": experience,
        }).execute()
        job_id = result.data[0]["id"]
        self.vector_store.add_job_description(job_id, description)
        return job_id

    def match_candidates_to_job(self, job_id, top_k=10):
        job = self.supabase.table("jobs").select("*").eq("id", job_id).execute()
        if not job.data:
            return []
        job_desc = job.data[0]["description"]
        matches = self.vector_store.search_candidates(job_desc, k=top_k * 3)

        scores = {}
        for m in matches:
            cid = m.metadata.get("candidate_id")
            if cid:
                scores[cid] = scores.get(cid, 0) + 1
        if not scores:
            return []

        max_s = max(scores.values())
        ranked = []
        for cid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            cand = self.supabase.table("candidates").select("*").eq("id", cid).execute()
            if cand.data:
                ranked.append({"candidate": cand.data[0], "match_score": round((sc / max_s) * 100, 1)})
        return ranked

    def generate_interview_questions(self, candidate_id, job_id):
        cand = self.supabase.table("candidates").select("*").eq("id", candidate_id).execute()
        job = self.supabase.table("jobs").select("*").eq("id", job_id).execute()
        if not cand.data or not job.data:
            return {"error": "not found"}

        prompt = ChatPromptTemplate.from_template(
            "You are a senior technical recruiter. Generate 5 targeted interview questions "
            "probing the gap between the candidate and the job.\n\n"
            "Candidate: {candidate_data}\nJob: {job_description}\nSkills: {required_skills}\n\n"
            "Return a numbered list of 5 questions only."
        )
        chain = prompt | self.llm | StrOutputParser()
        q = chain.invoke({
            "candidate_data": str(cand.data[0].get("parsed_data", {})),
            "job_description": job.data[0]["description"],
            "required_skills": str(job.data[0].get("required_skills", [])),
        })
        return {"questions": q}

    def ask_about_candidate(self, candidate_id, question):
        cand = self.supabase.table("candidates").select("*").eq("id", candidate_id).execute()
        if not cand.data:
            return {"error": "not found"}

        ctx_docs = self.vector_store.vector_store.similarity_search(
            question, k=3, filter={"candidate_id": candidate_id}
        )
        ctx = "\n\n".join(d.page_content for d in ctx_docs) or "(no CV context)"

        prompt = ChatPromptTemplate.from_template(
            "Answer the recruiter's question using ONLY the CV context.\n"
            "If not found, say 'Not found in CV.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        return {"answer": chain.invoke({"context": ctx, "question": question})}
PYEOF

# ---------- main.py ----------
cat > src/api/main.py << 'PYEOF'
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from src.agents.recruiter_agent import RecruitmentAgent

app = FastAPI(title="AI Recruitment Agent API")
agent = RecruitmentAgent()
UPLOAD_DIR = "data/resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/candidates/upload")
async def upload_cv(file: UploadFile = File(...), name: str = Form(...), email: str = Form(...)):
    safe = os.path.basename(file.filename or "cv.pdf")
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return {"status": "success", "data": agent.process_cv_upload(path, name, email)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/create")
async def create_job(title: str = Form(...), description: str = Form(...),
                     skills: str = Form(...), experience: str = Form(...)):
    skill_list = [s.strip() for s in skills.split(",") if s.strip()]
    return {"status": "success", "job_id": agent.create_job(title, description, skill_list, experience)}


@app.get("/api/jobs/{job_id}/match")
def match(job_id: str, top_k: int = 10):
    return {"status": "success", "candidates": agent.match_candidates_to_job(job_id, top_k)}


@app.get("/api/candidates/{candidate_id}/questions")
def questions(candidate_id: str, job_id: str):
    return agent.generate_interview_questions(candidate_id, job_id)


@app.post("/api/candidates/{candidate_id}/ask")
def ask(candidate_id: str, question: str = Form(...)):
    return agent.ask_about_candidate(candidate_id, question)
PYEOF

echo "All files written."
