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
        self.supabase = create_client(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"]
        )

    def process_cv_upload(self, file_path, candidate_name, email):
        parsed = self.cv_parser.parse(file_path)
        candidate_data = {
            "name": candidate_name,
            "email": email,
            "parsed_data": parsed["structured"],
        }
        result = self.supabase.table("candidates").insert(candidate_data).execute()
        candidate_id = result.data[0]["id"]
        self.vector_store.add_cv_chunks(candidate_id, parsed["chunks"])
        return {"candidate_id": candidate_id, "structured_data": parsed["structured"]}

    def create_job(self, title, description, skills, experience):
        result = (
            self.supabase.table("jobs")
            .insert({
                "title": title,
                "description": description,
                "required_skills": skills,
                "experience_level": experience,
            })
            .execute()
        )
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
                ranked.append({
                    "candidate": cand.data[0],
                    "match_score": round((sc / max_s) * 100, 1),
                })
        return ranked

    def generate_interview_questions(self, candidate_id, job_id):
        cand = self.supabase.table("candidates").select("*").eq("id", candidate_id).execute()
        job = self.supabase.table("jobs").select("*").eq("id", job_id).execute()
        if not cand.data or not job.data:
            return {"error": "not found"}

        prompt = ChatPromptTemplate.from_template(
            "You are a senior technical recruiter. Generate 5 targeted interview "
            "questions probing the gap between the candidate and the job.\n\n"
            "Candidate: {candidate_data}\n"
            "Job: {job_description}\n"
            "Skills: {required_skills}\n\n"
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
            "If not found, say 'Not found in CV.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        return {"answer": chain.invoke({"context": ctx, "question": question})}