import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from src.agents.recruiter_agent import RecruitmentAgent


app = FastAPI(title="AI Recruitment Agent API")
agent = RecruitmentAgent()
UPLOAD_DIR = "data/resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/candidates/upload")
async def upload_cv(
    file: UploadFile = File(...),
    name: str = Form(...),
    email: str = Form(...),
):
    safe = os.path.basename(file.filename or "cv.pdf")
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return {"status": "success", "data": agent.process_cv_upload(path, name, email)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jobs/create")
async def create_job(
    title: str = Form(...),
    description: str = Form(...),
    skills: str = Form(...),
    experience: str = Form(...),
):
    skill_list = [s.strip() for s in skills.split(",") if s.strip()]
    job_id = agent.create_job(title, description, skill_list, experience)
    return {"status": "success", "job_id": job_id}


@app.get("/api/jobs/list")
def list_jobs():
    res = agent.supabase.table("jobs").select("*").order("created_at", desc=True).execute()
    return {"status": "success", "jobs": res.data}


@app.get("/api/jobs/{job_id}/match")
def match(job_id: str, top_k: int = 10):
    return {
        "status": "success",
        "candidates": agent.match_candidates_to_job(job_id, top_k),
    }


@app.get("/api/candidates/list")
def list_candidates():
    res = agent.supabase.table("candidates").select("*").order("created_at", desc=True).execute()
    return {"status": "success", "candidates": res.data}


@app.get("/api/candidates/{candidate_id}/questions")
def questions(candidate_id: str, job_id: str):
    return agent.generate_interview_questions(candidate_id, job_id)


@app.post("/api/candidates/{candidate_id}/ask")
def ask(candidate_id: str, question: str = Form(...)):
    return agent.ask_about_candidate(candidate_id, question)