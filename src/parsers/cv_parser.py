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
- "experience_years": number (total years, 0 if unknown)
- "work_history": list of objects with keys "company", "role", "duration"
- "education": list of strings

CV TEXT:
{cv_text}

JSON:"""


class CVParser:
    def __init__(self, model_name="llama3.2", base_url="http://localhost:11434"):
        self.llm = OllamaLLM(model=model_name, base_url=base_url, temperature=0.1)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(EXTRACT_PROMPT)

    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    def _clean_json(self, text):
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def extract_structured_data(self, cv_text):
        chain = self.prompt | self.llm
        raw = chain.invoke({"cv_text": cv_text[:6000]})
        try:
            return json.loads(self._clean_json(raw))
        except json.JSONDecodeError as e:
            return {
                "parse_error": True,
                "error": str(e),
                "raw_response": raw[:500],
                "name": "",
                "email": "",
                "skills": [],
                "experience_years": 0,
                "work_history": [],
                "education": [],
            }

    def parse(self, file_path):
        cv_text = self.load_pdf(file_path)
        structured = self.extract_structured_data(cv_text)
        chunks = self.splitter.split_text(cv_text)
        return {"structured": structured, "chunks": chunks, "full_text": cv_text}