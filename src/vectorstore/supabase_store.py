import os
from dotenv import load_dotenv
from supabase import create_client
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

load_dotenv()


class RecruitmentVectorStore:
    def __init__(self):
        self.supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
        self.embeddings = OllamaEmbeddings(
            model=os.environ["OLLAMA_EMBED_MODEL"],
            base_url=os.environ["OLLAMA_BASE_URL"],
        )
        self.vector_store = SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents",
        )

    def add_cv_chunks(self, candidate_id, chunks):
        docs = [
            Document(page_content=c, metadata={"candidate_id": candidate_id, "type": "cv"})
            for c in chunks
        ]
        self.vector_store.add_documents(docs)

    def add_job_description(self, job_id, description):
        doc = Document(page_content=description, metadata={"job_id": job_id, "type": "job"})
        self.vector_store.add_documents([doc])

    def search_candidates(self, job_description, k=10):
        return self.vector_store.similarity_search(
            job_description, k=k, filter={"type": "cv"}
        )

    def search_jobs(self, cv_text, k=5):
        return self.vector_store.similarity_search(
            cv_text, k=k, filter={"type": "job"}
        )