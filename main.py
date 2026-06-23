import json
import os
import shutil
from pathlib import Path
from typing import List

try:
  from dotenv import load_dotenv
  load_dotenv()
except ImportError:
  pass

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from langchain_core.tools.retriever import create_retriever_tool
except ImportError:
    from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
try:
    from langchain_core.embeddings.fake import FakeEmbeddings
except ImportError:
    from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uvicorn

try:
  from groq import Groq
except Exception:
  Groq = None


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
ALLOWED_UPLOAD_SUFFIXES = {".txt", ".pdf", ".md"}


class ChatRequest(BaseModel):
  question: str = Field(..., min_length=3)
  role: str = Field("Sales Manager", description="User role in the real estate agency")
  experience: str = Field("Mid-level", description="User experience level")


class ChatResponse(BaseModel):
  answer: str
  source: List[str]
  sufficient_context: bool
  reasoning: str


class UploadResponse(BaseModel):
  files_saved: List[str]
  message: str


class KnowledgeFilesResponse(BaseModel):
  files: List[str]


class DecisionAgent:
  def __init__(self, kb_path: Path) -> None:
    self.kb_path = kb_path
    self.llm = None
    self.groq_client = None
    self.llm_provider = None
    self.groq_model = os.getenv(
      "GROQ_MODEL",
      "meta-llama/llama-4-scout-17b-16e-instruct",
    )
    self.retriever = None
    self.retriever_tool = None

    self._setup_llm()
    self._setup_retriever()

  @staticmethod
  def _normalize_text(value) -> str:
    if value is None:
      return ""
    if isinstance(value, str):
      return value
    if isinstance(value, list):
      parts = []
      for item in value:
        if isinstance(item, dict):
          text = item.get("text")
          if isinstance(text, str):
            parts.append(text)
        else:
          parts.append(str(item))
      return "".join(parts)
    return str(value)

  def _setup_llm(self) -> None:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
      self.llm = ChatOpenAI(
        model=os.getenv("GEMINI_MODEL", "gemma-2-27b-it"),
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=0.2
      )
      self.llm_provider = "gemini"
      return

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and Groq is not None:
      self.groq_client = Groq(api_key=groq_api_key)
      self.llm_provider = "groq"
      return

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
      self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
      self.llm_provider = "openai"

  def _load_documents(self):
    txt_loader = DirectoryLoader(
      str(self.kb_path),
      glob="**/*.txt",
      loader_cls=TextLoader,
      loader_kwargs={"encoding": "utf-8"},
      show_progress=False,
    )

    md_loader = DirectoryLoader(
      str(self.kb_path),
      glob="**/*.md",
      loader_cls=TextLoader,
      loader_kwargs={"encoding": "utf-8"},
      show_progress=False,
    )

    pdf_loader = DirectoryLoader(
      str(self.kb_path),
      glob="**/*.pdf",
      loader_cls=PyPDFLoader,
      show_progress=False,
    )

    documents = []

    # Keep startup resilient: one bad file should not bring down the API.
    try:
      documents.extend(txt_loader.load())
    except Exception:
      pass

    try:
      documents.extend(md_loader.load())
    except Exception:
      pass

    try:
      documents.extend(pdf_loader.load())
    except Exception:
      pass

    return documents

  def _setup_retriever(self) -> None:
    self.retriever = None
    self.retriever_tool = None

    if not self.kb_path.exists():
      return

    documents = self._load_documents()
    if not documents:
      return

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(documents)

    if os.getenv("OPENAI_API_KEY"):
      embeddings = OpenAIEmbeddings()
    else:
      # Fallback keeps the project runnable without external API credentials.
      embeddings = FakeEmbeddings(size=1536)

    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    self.retriever_tool = create_retriever_tool(
      self.retriever,
      "sales_playbook_retriever",
      "Search the sales strategy knowledge base for tactical guidance.",
    )

  def save_uploaded_files(self, files: List[UploadFile]) -> List[str]:
    self.kb_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for upload in files:
      original_name = Path(upload.filename or "").name
      if not original_name:
        continue

      suffix = Path(original_name).suffix.lower()
      if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
          status_code=400,
          detail=(
            f"Unsupported file type for {original_name}. "
            "Only .txt, .md, and .pdf are allowed."
          ),
        )

      destination = self.kb_path / original_name
      with destination.open("wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)

      saved_files.append(original_name)

    self._setup_retriever()
    return saved_files

  def _llm_decision(self, query: str, context: str, role: str, experience: str) -> dict:
    prompt = f"""
You are a sales enablement decision agent.
The user is a {role} in a real estate firm, with {experience} level of experience.
Use only the context below to answer the user question. Tailor your language, tone, and guidance to their role and experience level.

Context:
{context}

Question:
{query}

Return strictly valid JSON with this schema:
{{
  "sufficient": boolean,
  "reasoning": "short explanation",
  "answer": "final answer for the user",
  "confidence": "low|medium|high"
}}
"""

    if self.llm_provider == "groq" and self.groq_client is not None:
      completion = self.groq_client.chat.completions.create(
        model=self.groq_model,
        messages=[
          {"role": "system", "content": "You are a precise JSON-only assistant."},
          {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=700,
        top_p=1,
        stream=False,
      )
      message_content = completion.choices[0].message.content if completion.choices else ""
      content = self._normalize_text(message_content)
    elif self.llm is not None:
      result = self.llm.invoke(prompt)
      content = result.content if isinstance(result.content, str) else str(result.content)
    else:
      raise RuntimeError("No LLM provider configured.")

    try:
      return json.loads(content)
    except json.JSONDecodeError:
      start = content.find("{")
      end = content.rfind("}")
      if start != -1 and end != -1 and end > start:
        return json.loads(content[start : end + 1])
      raise

  def ask(self, query: str, role: str = "Sales Manager", experience: str = "Mid-level") -> ChatResponse:
    if self.retriever is None or self.retriever_tool is None:
      return ChatResponse(
        answer=(
          "The knowledge base is empty or unavailable. Add .txt, .md, or .pdf "
          "files to the /knowledge_base folder or upload them from the app."
        ),
        source=["No source"],
        sufficient_context=False,
        reasoning="No retriever could be initialized.",
      )

    documents = self.retriever.invoke(query)
    sources = sorted(
      {
        Path(doc.metadata.get("source", "Unknown source")).name
        for doc in documents
      }
    )

    tool_output = self.retriever_tool.invoke(query)
    tool_output_text = tool_output if isinstance(tool_output, str) else str(tool_output)
    context = "\n\n".join(doc.page_content for doc in documents)

    if (self.llm is not None or self.groq_client is not None) and context.strip():
      try:
        decision = self._llm_decision(query, context, role, experience)
        answer_text = str(decision.get("answer", "")).strip()
        if not answer_text:
          raise ValueError("LLM returned an empty answer.")

        return ChatResponse(
          answer=answer_text,
          source=sources or ["RetrieverTool output"],
          sufficient_context=bool(decision.get("sufficient", False)),
          reasoning=decision.get("reasoning", "Decision generated by LLM."),
        )
      except Exception:
        pass

    if not context.strip():
      return ChatResponse(
        answer=(
          "I could not find enough relevant information in the knowledge base for that question. "
          "Try adding strategy notes specific to this topic."
        ),
        source=["No relevant source"],
        sufficient_context=False,
        reasoning="Retriever returned no relevant chunks.",
      )

    preview = tool_output_text[:450].strip() or context[:450].strip()
    fallback_answer = (
      "I found related guidance in the playbook. "
      f"Here is the closest match: {preview}"
    )

    return ChatResponse(
      answer=fallback_answer,
      source=sources or ["RetrieverTool output"],
      sufficient_context=True,
      reasoning="Heuristic decision: retrieved context exists, using retriever output.",
    )


app = FastAPI(title="Sales Manager AI Playbook API")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

agent = DecisionAgent(KNOWLEDGE_BASE_DIR)


@app.get("/")
def health_check():
  return {"status": "ok", "service": "sales-manager-ai-playbook"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
  return agent.ask(request.question, role=request.role, experience=request.experience)


@app.post("/upload", response_model=UploadResponse)
def upload_files(files: List[UploadFile] = File(...)):
  if not files:
    raise HTTPException(status_code=400, detail="No files uploaded.")

  saved_files = agent.save_uploaded_files(files)
  if not saved_files:
    raise HTTPException(status_code=400, detail="No valid files were uploaded.")

  return UploadResponse(
    files_saved=saved_files,
    message="Files uploaded and retriever refreshed.",
  )


@app.get("/knowledge-files", response_model=KnowledgeFilesResponse)
def knowledge_files():
  if not KNOWLEDGE_BASE_DIR.exists():
    return KnowledgeFilesResponse(files=[])

  files = sorted(
    [
      path.name
      for path in KNOWLEDGE_BASE_DIR.iterdir()
      if path.is_file() and path.suffix.lower() in ALLOWED_UPLOAD_SUFFIXES
    ],
    key=str.lower,
  )

  return KnowledgeFilesResponse(files=files)


if __name__ == "__main__":
  uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
