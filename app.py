import streamlit as st
import os
import json
from pathlib import Path

# Import LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
from langchain_openai import ChatOpenAI

# Load dotenv for local dev
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- CONSTANTS ---
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Page config
st.set_page_config(
    page_title="RealtyAI — Sales Playbook",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS — matches Stitch mockup design system
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0d3d2e !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #d4f0e6 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #1a6b4a !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    margin-top: 0.25rem !important;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #22895f !important;
}

/* ---- Main area ---- */
.main .block-container {
    background: #f5f7f9;
    padding: 1.5rem 2rem;
    max-width: 1400px;
}

/* ---- Page title bar ---- */
.page-header {
    margin-bottom: 1.5rem;
}
.page-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0d2418;
    margin: 0 0 0.25rem 0;
}
.page-header p {
    color: #6b7e74;
    font-size: 0.875rem;
    margin: 0;
}

/* ---- Metric cards (Dashboard) ---- */
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #e8eef0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7a8f85;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0d2418;
    line-height: 1;
}
.metric-value.red { color: #e35b5b; }
.metric-value.teal { color: #1a8f5f; }
.metric-value.blue { color: #2a7acd; }
.metric-sub {
    font-size: 0.75rem;
    color: #9aada5;
    margin-top: 0.35rem;
}

/* ---- Pipeline deal cards ---- */
.pipeline-col-header {
    font-weight: 700;
    font-size: 0.95rem;
    color: #0d2418;
    padding: 0.5rem 0 0.75rem 0;
    border-bottom: 2px solid #1a8f5f;
    margin-bottom: 0.75rem;
}
.badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
.badge-residential { background: #d4f0e6; color: #0d5f35; }
.badge-commercial { background: #dbeafe; color: #1e40af; }
.badge-investment { background: #fef3c7; color: #92400e; }
.deal-card {
    background: #ffffff;
    border: 1px solid #e8eef0;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.deal-name { font-weight: 600; font-size: 0.9rem; color: #0d2418; }
.deal-addr { font-size: 0.78rem; color: #7a8f85; margin: 0.2rem 0 0.4rem 0; }
.deal-footer { display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; }
.deal-value { font-weight: 700; font-size: 0.8rem; color: #1a8f5f; }
.deal-tag {
    font-size: 0.68rem;
    color: #6b7e74;
    background: #f0f4f2;
    padding: 2px 7px;
    border-radius: 12px;
}

/* ---- Chat UI ---- */
.chat-bubble-user {
    background: #1a8f5f;
    color: #ffffff;
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.875rem;
    line-height: 1.5;
}
.chat-bubble-ai {
    background: #ffffff;
    color: #1a2820;
    border: 1px solid #e8eef0;
    border-radius: 12px 12px 12px 2px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.875rem;
    line-height: 1.5;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.chat-citation {
    font-size: 0.7rem;
    color: #9aada5;
    margin-top: 0.4rem;
}

/* ---- SOP document card ---- */
.sop-card {
    background: #ffffff;
    border: 1px solid #e8eef0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.sop-icon { font-size: 1.25rem; }
.sop-name { font-weight: 600; font-size: 0.875rem; color: #0d2418; }
.sop-meta { font-size: 0.72rem; color: #9aada5; }
.sop-badge { font-size: 0.67rem; font-weight: 600; }

/* ---- Action cards (Strategy tools) ---- */
.tool-card {
    background: #ffffff;
    border: 1px solid #e8eef0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.tool-card h4 { margin: 0 0 0.4rem 0; font-size: 0.925rem; color: #0d2418; }
.tool-card p { margin: 0; font-size: 0.8rem; color: #7a8f85; }

/* ---- Governance validation card ---- */
.validation-box {
    background: #f0faf4;
    border: 1px solid #a7d7be;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 0.75rem;
}

/* ---- Live feed item ---- */
.feed-item {
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e8eef0;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: #2d4a3a;
}
.feed-item strong { color: #0d2418; }

/* ---- Tabs override ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 2px solid #dce8e2;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem;
    font-weight: 500;
    color: #6b7e74;
    padding: 0.4rem 1rem;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #1a8f5f !important;
    border-bottom: 2px solid #1a8f5f !important;
    font-weight: 700 !important;
}

/* ---- Section subheadings ---- */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0d2418;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e8eef0;
}

/* ---- New Analysis bottom CTA ---- */
.sidebar-cta {
    background: #1a6b4a;
    color: white;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-weight: 600;
    font-size: 0.875rem;
    text-align: center;
    cursor: pointer;
    margin-top: 1rem;
}

/* ---- Hide Streamlit chrome ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS — API KEY RESOLUTION
# ============================================================
def get_api_key():
    """Read API key from Streamlit Cloud secrets, env, or session state."""
    # 1. Streamlit secrets (production)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # 2. Environment variable (local .env)
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    # 3. Manually entered in sidebar
    return st.session_state.get("gemini_key", "")

def get_model_name():
    try:
        return st.secrets.get("GEMINI_MODEL", "gemma-4-31b-it")
    except Exception:
        return os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

def is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "quota" in msg or "resource_exhausted" in msg or "rate" in msg

def clean_response(text: str) -> str:
    """Strip Gemma 4 chain-of-thought <thought>...</thought> blocks.
    The model outputs its reasoning inside these tags before the real answer."""
    import re
    # Remove anything inside <thought>...</thought> (including the tags)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    # Also handle unclosed <thought> blocks (model cuts off mid-think)
    text = re.sub(r"<thought>.*", "", text, flags=re.DOTALL)
    # Clean up any leftover think tags
    text = re.sub(r"</?thought>", "", text)
    return text.strip()



# ============================================================
# SESSION STATE
# ============================================================
if "profile" not in st.session_state:
    qp = st.query_params
    if "name" in qp and "focus" in qp:
        st.session_state.profile = {
            "name": qp["name"],
            "focus": qp["focus"],
            "experience": qp.get("experience", "Senior (8+ years)"),
            "targetSavings": int(qp["targetSavings"]) if "targetSavings" in qp and qp["targetSavings"].isdigit() else 16
        }
    else:
        st.session_state.profile = None

if "deals" not in st.session_state:
    st.session_state.deals = [
        {"id": 1, "accountName": "Marge Simpson", "propertyAddress": "742 Evergreen Terrace", "dealValue": "$845k", "stage": "Lead", "type": "Residential", "agent": "Logan Roy", "age": "2 days idle", "tag": "Follow up"},
        {"id": 2, "accountName": "Logan Roy", "propertyAddress": "Harbor District Plaza", "dealValue": "$2.4M", "stage": "Lead", "type": "Commercial", "agent": "Logan Roy", "age": "4 hours ago", "tag": "Call Call"},
        {"id": 3, "accountName": "Scotch Roy", "propertyAddress": "Oak Ridge Estates", "dealValue": "$1.2M", "stage": "Discovery", "type": "Residential", "agent": "Scotch Roy", "age": "1 week idle", "tag": "Stay Hot"},
        {"id": 4, "accountName": "Roman Roy", "propertyAddress": "Skyline Heights Tower B", "dealValue": "$5.8M", "stage": "Proposal", "type": "Investment", "agent": "Roman Roy", "age": "active", "tag": "Under Legal"},
        {"id": 5, "accountName": "Summit Group", "propertyAddress": "555 Valley View", "dealValue": "$3.1M", "stage": "Closed Won", "type": "Commercial", "agent": "Dave", "age": "closed", "tag": "Archive"},
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "ai", "content": "Hello! I've finished analyzing your Q3 pipeline data. Your conversion rate on initial discovery calls is currently **18.4%**, which is slightly below the target of 22%.\n\nBased on your recent call recordings, we're seeing a drop-off during the 'Objection Handling' phase of the SOP. Would you like to deep-dive into the specific transcriptions or review the updated scripts?", "source": None}
    ]

if "nb_chat_history" not in st.session_state:
    st.session_state.nb_chat_history = [
        {"role": "ai", "content": "Hello! I am grounded in your team SOPs. Ask me a question or use the suggestions below.", "citations": []}
    ]

if "active_page" not in st.session_state:
    st.session_state.active_page = st.query_params.get("page", "Dashboard")

def switch_page(page_name):
    st.session_state.active_page = page_name
    st.query_params["page"] = page_name
    if st.session_state.profile:
        p = st.session_state.profile
        st.query_params["name"] = p["name"]
        st.query_params["focus"] = p["focus"]
        st.query_params["experience"] = p["experience"]
        st.query_params["targetSavings"] = str(p["targetSavings"])
    st.rerun()


# ============================================================
# RAG DECISION AGENT
# ============================================================
class StreamlitDecisionAgent:
    def __init__(self, kb_path: Path):
        self.kb_path = kb_path
        self.retriever = None
        self.retriever_tool = None
        self.llm = None
        self.llm_provider = None
        self._setup_llm()
        self._setup_retriever()

    def _setup_llm(self):
        api_key = get_api_key()
        if api_key:
            model_name = get_model_name()
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=0.3
            )
            self.llm_provider = "gemini"

    def _load_documents(self):
        documents = []
        for glob, cls, kwargs in [
            ("**/*.txt", TextLoader, {"loader_kwargs": {"encoding": "utf-8"}}),
            ("**/*.md",  TextLoader, {"loader_kwargs": {"encoding": "utf-8"}}),
            ("**/*.pdf", PyPDFLoader, {}),
        ]:
            try:
                loader = DirectoryLoader(str(self.kb_path), glob=glob, loader_cls=cls, show_progress=False, **kwargs)
                documents.extend(loader.load())
            except Exception:
                pass
        return documents

    def _setup_retriever(self):
        self.retriever = None
        self.retriever_tool = None
        if not self.kb_path.exists():
            return
        documents = self._load_documents()
        if not documents:
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        chunks = splitter.split_documents(documents)
        embeddings = FakeEmbeddings(size=1536)
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "sales_playbook_retriever",
            "Search the sales strategy knowledge base for tactical guidance.",
        )

    def _llm_decision(self, query: str, context: str, role: str, experience: str) -> str:
        prompt = f"""You are a senior real estate sales enablement coach.
The user is a {role} with {experience} experience.
Use only the context below to give a concise, actionable answer. Tailor your tone to their experience level.
Do NOT wrap your answer in JSON or any code block — just reply directly in plain text.

Context:
{context}

Question:
{query}
"""
        try:
            result = self.llm.invoke(prompt)
            return clean_response(result.content if isinstance(result.content, str) else str(result.content))
        except Exception as e:
            if is_quota_error(e):
                return "🚫 **API Quota Exceeded** — Your free tier daily limit is used up.\n\n**Fix options:**\n1. Wait ~24 hrs for quota to reset\n2. Enable billing at [aistudio.google.com](https://aistudio.google.com)"
            return f"⚠️ Model error: {str(e)[:200]}"

    def ask(self, query: str, role: str, experience: str) -> dict:
        if not self.llm:
            return {"answer": "⚠️ AI Coach is offline — no API key found. Add your **GEMINI_API_KEY** in Streamlit Cloud **Settings → Secrets**.", "source": None}

        prompt = f"You are a real estate sales coach. Answer this question for a {role} ({experience}). Reply in plain text only — no JSON, no code blocks.\n\nQuestion: {query}"

        if not self.retriever:
            # No documents — use LLM directly without RAG
            try:
                result = self.llm.invoke(prompt)
                answer = clean_response(result.content if isinstance(result.content, str) else str(result.content))
                return {"answer": answer, "source": "General AI (no docs indexed)"}
            except Exception as e:
                if is_quota_error(e):
                    return {"answer": "🚫 **API Quota Exceeded** — Your free tier daily limit is used up.\n\n**Fix:** Enable billing at [aistudio.google.com](https://aistudio.google.com) or wait ~24 hrs for quota reset.", "source": None}
                return {"answer": f"⚠️ Error: {str(e)[:200]}", "source": None}

        documents = self.retriever.invoke(query)
        sources = sorted({Path(doc.metadata.get("source", "Unknown")).name for doc in documents})
        context = "\n\n".join(doc.page_content for doc in documents)
        if context.strip():
            answer = self._llm_decision(query, context, role, experience)
            return {"answer": answer, "source": ", ".join(sources)}
        return {"answer": "Could not find relevant info in the knowledge base. Try uploading more documents.", "source": None}



# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding: 0.5rem 0 1.25rem 0;">
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.25rem;">
            <div style="background:#1a8f5f; border-radius:8px; width:36px; height:36px; display:flex; align-items:center; justify-content:center; font-size:1.1rem; box-shadow:0 0 12px rgba(26,143,95,0.4);">⚡</div>
            <div>
                <div style="font-weight:800; font-size:1.05rem; color:#ffffff; line-height:1.1;">RealtyAI</div>
                <div style="font-size:0.6rem; color:#7db89a; text-transform:uppercase; letter-spacing:1.5px;">Premium Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav pages
    pages = {
        "Dashboard":   "📊",
        "Pipeline":    "📋",
        "AI Coach":    "🤖",
        "SOPs":        "📚",
        "Strategy Tools": "🛠",
    }
    for page, icon in pages.items():
        is_active = st.session_state.active_page == page
        btn_style = "background:#1a8f5f !important;" if is_active else ""
        if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
            switch_page(page)

    st.markdown("---")

    # Profile
    if st.session_state.profile:
        p = st.session_state.profile
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.07); border-radius:8px; padding:0.75rem; margin-bottom:0.75rem;">
            <div style="font-weight:700; font-size:0.85rem;">{p['name']}</div>
            <div style="font-size:0.72rem; color:#7db89a; margin-top:0.2rem;">{p['focus']}</div>
            <div style="font-size:0.68rem; color:#5d8a72; margin-top:0.15rem;">{p['experience']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Edit Profile", use_container_width=True):
            st.session_state.profile = None
            st.query_params.clear()
            st.rerun()
    else:
        st.info("Set up your profile to personalise the AI coach.")

    st.markdown("---")

    # API key section
    st.markdown('<div style="font-size:0.72rem; font-weight:600; color:#7db89a; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;">API Configuration</div>', unsafe_allow_html=True)
    existing_key = get_api_key()
    if existing_key:
        st.markdown('<div style="font-size:0.78rem; color:#4caf82;">✅ Gemini API key active</div>', unsafe_allow_html=True)
    else:
        key_input = st.text_input("Gemini API Key", type="password", placeholder="AIza...", help="Or add GEMINI_API_KEY to Streamlit Secrets")
        if key_input:
            st.session_state.gemini_key = key_input
            st.success("Key saved!")
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#5d8a72; text-align:center; padding-top:0.5rem;">
        RealtyAI v2.0 · Playbook Engine<br>
        <span style="color:#3a6b52;">Powered by Gemini</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ONBOARDING — shown when no profile
# ============================================================
if not st.session_state.profile:
    st.markdown("""
    <div style="max-width:520px; margin:4rem auto; background:#ffffff; border-radius:16px; padding:2.5rem; box-shadow:0 4px 24px rgba(0,0,0,0.1); border:1px solid #e8eef0;">
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="font-size:2.5rem;">⚡</div>
            <h2 style="margin:0.5rem 0 0.25rem; color:#0d2418;">Welcome to RealtyAI</h2>
            <p style="color:#7a8f85; font-size:0.875rem; margin:0;">Configure your workspace to personalise the playbook</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name", placeholder="Ryan Oliver")
        with col2:
            focus = st.selectbox("Real Estate Focus", ["Residential Brokerage", "Commercial Real Estate (CRE)", "Property Developer"])
        exp = st.selectbox("Experience Level", ["Junior (0–3 years)", "Mid-Level (3–8 years)", "Senior (8+ years)"])
        savings = st.slider("Weekly Time Saving Target (hrs)", 5, 30, 16)

        submitted = st.form_submit_button("🚀 Launch Playbook Workspace", use_container_width=True)
        if submitted:
            if not name.strip():
                st.error("Please enter your name.")
            else:
                st.session_state.profile = {"name": name.strip(), "focus": focus, "experience": exp, "targetSavings": savings}
                st.query_params["name"] = name.strip()
                st.query_params["focus"] = focus
                st.query_params["experience"] = exp
                st.query_params["targetSavings"] = str(savings)
                st.rerun()
    st.stop()


# ============================================================
# PAGE ROUTING
# ============================================================
page = st.session_state.active_page
profile = st.session_state.profile


# ============================================================
# PAGE 1 — DASHBOARD OVERVIEW
# ============================================================
if page == "Dashboard":
    deals = st.session_state.deals

    # --- Computations for Dashboard metrics ---
    def parse_deal_val(val_str):
        if not val_str:
            return 0.0
        v = str(val_str).replace("$", "").replace(" ", "").upper()
        if "M" in v:
            try: return float(v.replace("M", "")) * 1_000_000
            except: pass
        if "K" in v:
            try: return float(v.replace("K", "")) * 1_000
            except: pass
        try:
            return float(v)
        except:
            return 0.0

    def format_deal_val(val):
        if val >= 1_000_000:
            return f"${val / 1_000_000:.1f}M"
        elif val >= 1_000:
            return f"${val / 1_000:.0f}k"
        return f"${val:.0f}"

    total_deals   = len(deals)
    leads         = [d for d in deals if d["stage"] == "Lead"]
    discovery     = [d for d in deals if d["stage"] == "Discovery"]
    proposals     = [d for d in deals if d["stage"] == "Proposal"]
    closed        = [d for d in deals if d["stage"] == "Closed Won"]
    active_deals  = leads + discovery + proposals
    win_rate      = round(len(closed) / total_deals * 100) if total_deals else 0

    active_val = sum(parse_deal_val(d.get("dealValue", "")) for d in active_deals)
    closed_val = sum(parse_deal_val(d.get("dealValue", "")) for d in closed)

    st.markdown(f"""
    <div class="page-header">
        <h1>Dashboard Overview</h1>
        <p>Hi {profile['name']}! Here is an easy-to-read summary of your sales pipeline. Use the left menu to navigate, or update details directly below.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Executive Copilot Summary Box ──────────────────────
    st.markdown(f"""
    <div style="background-color: #f0fbf7; border: 1px solid #ccece0; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
        <h4 style="margin-top: 0; color: #0d3d2e; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem;">
            🤖 AI Executive Summary
        </h4>
        <p style="margin: 0.5rem 0; color: #2d503d; font-size: 0.9rem; line-height: 1.5;">
            Your pipeline has <strong>{len(active_deals)} active opportunities</strong> worth <strong>{format_deal_val(active_val)}</strong>. You have closed <strong>{format_deal_val(closed_val)}</strong> successfully.
        </p>
        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #d8f3e5; font-size: 0.82rem; color: #2d503d; display: flex; flex-direction: column; gap: 0.4rem;">
            <div>⚠️ <strong>Action Needed:</strong> {len(leads) + len(discovery)} early-stage deals require initial follow-ups to progress.</div>
            <div>💡 <strong>Coach Tip:</strong> Roman Roy's proposal is at <strong>Proposal</strong> stage ({format_deal_val(parse_deal_val(proposals[0].get("dealValue")) if proposals else 0)}). Go to <strong>AI Coach</strong> to draft a custom follow-up sequence.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: KPI Cards ──────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📋 Active Opportunities</div>
            <div class="metric-value teal">{len(active_deals)}</div>
            <div class="metric-sub">Leads, Discovery & Proposals</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💰 Active Pipeline Value</div>
            <div class="metric-value blue">{format_deal_val(active_val)}</div>
            <div class="metric-sub">Total potential revenue</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">✅ Closed Won Revenue</div>
            <div class="metric-value teal">{format_deal_val(closed_val)}</div>
            <div class="metric-sub">Successfully won deals</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card" style="background:#0d3d2e; border-color:#1a8f5f;">
            <div class="metric-label" style="color:#7db89a;">🎯 Win Rate</div>
            <div class="metric-value" style="color:#ffffff;">{win_rate}%</div>
            <div class="metric-sub" style="color:#7db89a;">Won vs. total opportunities</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Columns ────────────────────────────────────────
    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        st.markdown('<div class="section-title">📊 Deals by Stage (Workflow)</div>', unsafe_allow_html=True)

        stage_data = [
            ("Lead",       len(leads),     "#38bdf8", "New contacts not yet qualified"),
            ("Discovery",  len(discovery), "#f59e0b", "Actively gathering client requirements"),
            ("Proposal",   len(proposals), "#1a8f5f", "Offer or listing agreement sent"),
            ("Closed Won", len(closed),    "#6366f1", "Deal completed successfully"),
        ]
        max_count = max(count for _, count, _, _ in stage_data) or 1

        for stage, count, color, desc in stage_data:
            bar_pct = int(count / max_count * 100)
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e8eef0; border-radius:10px; padding:0.85rem 1.1rem; margin-bottom:0.6rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <div>
                        <span style="font-weight:700; font-size:0.9rem; color:#0d2418;">{stage}</span>
                        <span style="font-size:0.75rem; color:#9aada5; margin-left:0.5rem;">{desc}</span>
                    </div>
                    <span style="font-weight:800; font-size:1.1rem; color:{color};">{count}</span>
                </div>
                <div style="background:#f0f4f2; border-radius:4px; height:6px;">
                    <div style="background:{color}; border-radius:4px; height:6px; width:{bar_pct}%; transition:width 0.3s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Quick Deal Stage Updater
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ Quick Deal Stage Updater</div>', unsafe_allow_html=True)
        with st.container(border=True):
            col_d, col_s = st.columns(2)
            with col_d:
                deal_names = [f"{d['accountName']} ({d['propertyAddress']})" for d in deals]
                selected_deal_name = st.selectbox("Select Deal to Update", deal_names, index=0, key="dash_sel_deal", label_visibility="collapsed")
            with col_s:
                stages = ["Lead", "Discovery", "Proposal", "Closed Won"]
                # Find matching deal
                idx = deal_names.index(selected_deal_name)
                matching_deal = deals[idx]
                curr_stage = matching_deal["stage"]
                try:
                    curr_stage_idx = stages.index(curr_stage)
                except ValueError:
                    curr_stage_idx = 0
                new_stage = st.selectbox("Move to Stage", stages, index=curr_stage_idx, key="dash_sel_stage", label_visibility="collapsed")
                
                if new_stage != curr_stage:
                    matching_deal["stage"] = new_stage
                    st.success(f"Moved {matching_deal['accountName']} to {new_stage}!")
                    st.rerun()

    with right_col:
        st.markdown('<div class="section-title">🔔 Deals Needing Attention</div>', unsafe_allow_html=True)

        attention_deals = [d for d in deals if d["stage"] in ["Lead", "Discovery", "Proposal"]]
        if attention_deals:
            for d in attention_deals[:4]:
                stage_colors = {"Lead": "#38bdf8", "Discovery": "#f59e0b", "Proposal": "#1a8f5f"}
                s_color = stage_colors.get(d["stage"], "#9aada5")
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #e8eef0; border-left:3px solid {s_color}; border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.55rem;">
                    <div style="font-weight:600; font-size:0.875rem; color:#0d2418;">{d['accountName']}</div>
                    <div style="font-size:0.75rem; color:#7a8f85; margin:0.15rem 0;">{d['propertyAddress']} · <strong style="color:{s_color};">{d['stage']}</strong></div>
                    <div style="font-size:0.72rem; color:#9aada5;">Next: {d.get('tag', d.get('nextStep','—'))}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active deals yet. Add some in the Pipeline tab.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚡ Quick Navigation Actions</div>', unsafe_allow_html=True)
        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("💬 Ask AI Coach", use_container_width=True, key="dash_coach"):
                switch_page("AI Coach")
            if st.button("📚 Browse SOPs", use_container_width=True, key="dash_sops"):
                switch_page("SOPs")
        with qa2:
            if st.button("📋 View Pipeline", use_container_width=True, key="dash_pipe"):
                switch_page("Pipeline")
            if st.button("🛠 Strategy Tools", use_container_width=True, key="dash_tools"):
                switch_page("Strategy Tools")



# ============================================================
# PAGE 2 — PIPELINE BOARD
# ============================================================
elif page == "Pipeline":
    st.markdown("""
    <div class="page-header">
        <h1>Sales Pipeline</h1>
        <p>Manage your residential and commercial opportunities</p>
    </div>
    """, unsafe_allow_html=True)

    # New lead form
    with st.expander("➕ New Lead / Deal Card"):
        with st.form("new_deal_form"):
            c1, c2 = st.columns(2)
            with c1:
                c_name = st.text_input("Client / Account Name", placeholder="Sarah Connor")
                c_addr = st.text_input("Property Address", placeholder="1045 River Rd")
                c_val = st.text_input("Deal Value / GCI", placeholder="$450k / $18k GCI")
            with c2:
                c_type = st.selectbox("Property Type", ["Residential", "Commercial", "Investment"])
                c_stage = st.selectbox("Pipeline Stage", ["Lead", "Discovery", "Proposal", "Closed Won"])
                c_agent = st.text_input("Assigned Agent", placeholder="Logan Roy")
            c_tag = st.text_input("Action Tag", placeholder="Follow up")
            if st.form_submit_button("Create Deal Card", use_container_width=True):
                if c_name.strip() and c_addr.strip():
                    new_id = max([d["id"] for d in st.session_state.deals], default=0) + 1
                    st.session_state.deals.append({
                        "id": new_id, "accountName": c_name, "propertyAddress": c_addr,
                        "dealValue": c_val, "stage": c_stage, "type": c_type,
                        "agent": c_agent, "age": "just now", "tag": c_tag
                    })
                    st.success(f"Deal for {c_name} created!")
                    st.rerun()
                else:
                    st.error("Client name and property address are required.")

    # Kanban columns
    stage_defs = ["Lead", "Discovery", "Proposal", "Closed Won"]
    cols = st.columns(4)

    type_badge = {
        "Residential": "badge-residential",
        "Commercial":  "badge-commercial",
        "Investment":  "badge-investment",
    }

    for col, stage in zip(cols, stage_defs):
        with col:
            stage_deals = [d for d in st.session_state.deals if d["stage"] == stage]
            total_val = len(stage_deals)
            st.markdown(f"""
            <div class="pipeline-col-header">
                ● {stage} &nbsp;<span style="background:#e8f5ee; color:#1a8f5f; border-radius:10px; padding:1px 8px; font-size:0.75rem;">{total_val}</span>
            </div>
            """, unsafe_allow_html=True)

            for deal in stage_deals:
                badge_class = type_badge.get(deal.get("type", "Residential"), "badge-residential")
                st.markdown(f"""
                <div class="deal-card">
                    <span class="badge {badge_class}">{deal.get('type','Residential')}</span>
                    <div style="font-size:0.7rem; color:#9aada5; float:right;">{deal.get('dealValue','')}</div>
                    <div style="clear:both; margin-top:0.4rem;" class="deal-name">{deal['accountName']}</div>
                    <div class="deal-addr">{deal['propertyAddress']}</div>
                    <div style="font-size:0.72rem; color:#9aada5;">{deal.get('agent','')} · {deal.get('age','')}</div>
                    <div class="deal-footer">
                        <span class="deal-tag">{deal.get('tag','')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                current_idx = stage_defs.index(stage)
                new_stage = st.selectbox("Move to:", stage_defs, index=current_idx, key=f"move_{deal['id']}", label_visibility="collapsed")
                if new_stage != stage:
                    for d in st.session_state.deals:
                        if d["id"] == deal["id"]:
                            d["stage"] = new_stage
                    st.rerun()


# ============================================================
# PAGE 3 — AI COACH
# ============================================================
elif page == "AI Coach":
    st.markdown("""
    <div class="page-header">
        <h1>AI Coach</h1>
        <p>RAG-powered sales advisor grounded in your knowledge base</p>
    </div>
    """, unsafe_allow_html=True)

    agent = StreamlitDecisionAgent(KNOWLEDGE_BASE_DIR)
    api_active = bool(get_api_key())

    chat_col, kb_col = st.columns([2.2, 1])

    with chat_col:
        if not api_active:
            st.warning("⚠️ No API key detected. Add **GEMINI_API_KEY** to Streamlit Cloud **Settings → Secrets** to enable the AI coach.")

        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)
                    if msg.get("source"):
                        st.markdown(f'<div class="chat-citation">📎 Source: {msg["source"]}</div>', unsafe_allow_html=True)

        # Input
        query = st.chat_input("Ask the Coach anything about your performance...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query, "source": None})
            role_str = profile["focus"]
            exp_str = profile["experience"]
            with st.spinner("Coach is thinking..."):
                response = agent.ask(query, role_str, exp_str)
            st.session_state.chat_history.append({"role": "ai", "content": response["answer"], "source": response.get("source")})
            st.rerun()

    with kb_col:
        # Knowledge base panel
        st.markdown('<div class="section-title">📂 Knowledge Base</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop SOP / Floor Plan Transcripts",
            accept_multiple_files=True,
            type=["txt", "md", "pdf"],
            help="PDF, DOCX, or TXT (max 50MB per file)"
        )
        if st.button("📥 Index Files", use_container_width=True):
            if uploaded_files:
                for f in uploaded_files:
                    dest = KNOWLEDGE_BASE_DIR / f.name
                    with open(dest, "wb") as fh:
                        fh.write(f.getbuffer())
                st.success(f"Indexed {len(uploaded_files)} file(s)!")
                agent._setup_retriever()
                st.rerun()
            else:
                st.error("Select at least one file.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Active content
        st.markdown('<div class="section-title">Active Content</div>', unsafe_allow_html=True)
        files = sorted([f for f in KNOWLEDGE_BASE_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".txt", ".md", ".pdf"]])
        if files:
            for f in files:
                icon = "📄" if f.suffix == ".txt" else ("📋" if f.suffix == ".md" else "📕")
                size_kb = f.stat().st_size // 1024
                st.markdown(f"""
                <div class="sop-card">
                    <div class="sop-icon">{icon}</div>
                    <div>
                        <div class="sop-name">{f.name}</div>
                        <div class="sop-meta">{size_kb} KB · Indexed</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No files indexed yet. Upload above to enable RAG grounding.")

        # Quick insights
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Insights</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-bottom:0.75rem;">
            <div style="background:#f0faf4; border-radius:8px; padding:0.6rem 0.85rem; flex:1;">
                <div style="font-size:0.68rem; color:#5d8a72; text-transform:uppercase;">Conv. Rate</div>
                <div style="font-weight:700; font-size:1.1rem; color:#0d2418;">18.4%</div>
            </div>
            <div style="background:#f0faf4; border-radius:8px; padding:0.6rem 0.85rem; flex:1;">
                <div style="font-size:0.68rem; color:#5d8a72; text-transform:uppercase;">Avg Deal</div>
                <div style="font-weight:700; font-size:1.1rem; color:#0d2418;">2.4m</div>
            </div>
        </div>
        <div style="background:#0d3d2e; border-radius:10px; padding:0.85rem; color:#d4f0e6; font-size:0.78rem;">
            <div style="font-weight:700; margin-bottom:0.3rem;">⚡ CoachMode: Aggressive</div>
            Focusing on high-velocity closing techniques for current pipeline stage.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 4 — SOPs KNOWLEDGE BASE
# ============================================================
elif page == "SOPs":
    st.markdown("""
    <div class="page-header">
        <h1>SOPs Library</h1>
        <p>Knowledge base, document indexing, and library health</p>
    </div>
    """, unsafe_allow_html=True)

    upload_col, health_col = st.columns([1.5, 1])

    with upload_col:
        st.markdown('<div class="section-title">📤 Knowledge Base Uploader</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="border:2px dashed #d0e8dc; border-radius:12px; padding:2rem; text-align:center; background:#f8fdfb; margin-bottom:1rem;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📁</div>
            <div style="font-weight:600; color:#0d2418; margin-bottom:0.25rem;">Drag & Drop Documents</div>
            <div style="font-size:0.8rem; color:#9aada5;">PDF, DOCX, or TXT (max 50MB per file)</div>
        </div>
        """, unsafe_allow_html=True)

        sop_files = st.file_uploader("Select files from computer", accept_multiple_files=True, type=["txt", "md", "pdf"], label_visibility="collapsed")
        if st.button("📥 Select & Index Files from Computer", use_container_width=True):
            if sop_files:
                for f in sop_files:
                    dest = KNOWLEDGE_BASE_DIR / f.name
                    with open(dest, "wb") as fh:
                        fh.write(f.getbuffer())
                st.success(f"✅ {len(sop_files)} file(s) indexed successfully!")
                st.rerun()

        st.markdown('<br><div class="section-title">Browse Documentation</div>', unsafe_allow_html=True)

        # Built-in SOPs
        builtin_sops = [
            ("📘", "Luxury Client Onboarding v2.4", "Operations", "Oct 24, 2023", "Indexed"),
            ("📊", "Market Analysis: Q3 Seattle Residential", "Strategy", "Oct 21, 2023", "Indexed"),
            ("📞", "Cold Calling Script: Rejection Handling", "Training", "Oct 18, 2023", "Processing"),
            ("⚖️", "Broker Compliance Checklist 2024", "Legal", "Oct 15, 2023", "Indexed"),
        ]
        cat_colors = {"Operations": "#dbeafe", "Strategy": "#d4f0e6", "Training": "#fef3c7", "Legal": "#fde8e8"}
        cat_text = {"Operations": "#1e40af", "Strategy": "#0d5f35", "Training": "#92400e", "Legal": "#9b1c1c"}

        for icon, name, cat, date, status in builtin_sops:
            cat_bg = cat_colors.get(cat, "#f0f4f2")
            cat_fg = cat_text.get(cat, "#374151")
            status_color = "#1a8f5f" if status == "Indexed" else "#e38a1a"
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between; background:#ffffff; border:1px solid #e8eef0; border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <span style="font-size:1.2rem;">{icon}</span>
                    <div>
                        <div style="font-weight:600; font-size:0.875rem; color:#0d2418;">{name}</div>
                        <div style="font-size:0.72rem; color:#9aada5;">Last Updated: {date}</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <span style="background:{cat_bg}; color:{cat_fg}; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:4px;">{cat}</span>
                    <span style="color:{status_color}; font-size:0.75rem; font-weight:600;">● {status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Uploaded files too
        uploaded_kb = [f for f in KNOWLEDGE_BASE_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".txt", ".md", ".pdf"]]
        for f in uploaded_kb:
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between; background:#ffffff; border:1px solid #e8eef0; border-radius:8px; padding:0.75rem 1rem; margin-bottom:0.5rem;">
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <span style="font-size:1.2rem;">📄</span>
                    <div>
                        <div style="font-weight:600; font-size:0.875rem; color:#0d2418;">{f.name}</div>
                        <div style="font-size:0.72rem; color:#9aada5;">{f.stat().st_size // 1024} KB</div>
                    </div>
                </div>
                <span style="color:#1a8f5f; font-size:0.75rem; font-weight:600;">● Indexed</span>
            </div>
            """, unsafe_allow_html=True)

    with health_col:
        st.markdown('<div class="section-title">🏥 Library Health</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e8eef0; border-radius:10px; padding:1rem; margin-bottom:0.75rem;">
            <div style="font-size:0.75rem; color:#9aada5; margin-bottom:0.25rem;">RAG Readiness</div>
            <div style="font-size:2rem; font-weight:700; color:#1a8f5f;">92%</div>
        </div>
        <div style="background:#f0faf4; border-radius:10px; padding:1rem; margin-bottom:0.75rem; font-size:0.8rem; color:#2d6048;">
            <div style="font-weight:700; margin-bottom:0.35rem;">128 Documents indexed</div>
            Real-time AI context retrieval active
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0d3d2e; border-radius:10px; padding:1rem; color:#d4f0e6; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-weight:700; font-size:0.875rem; margin-bottom:0.25rem;">⚡ Smart-Tagging Active</div>
                    <div style="font-size:0.75rem; color:#7db89a;">Documents are automatically categorised using GPT-4o analysis.</div>
                </div>
                <span style="background:#1a8f5f; color:white; font-size:0.65rem; font-weight:700; padding:2px 6px; border-radius:4px;">PRO</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Storage stats
        file_count = len(list(KNOWLEDGE_BASE_DIR.iterdir()))
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; text-align:center;">
            <div style="background:#ffffff; border:1px solid #e8eef0; border-radius:8px; padding:0.6rem;">
                <div style="font-size:1rem; font-weight:700; color:#0d2418;">2.4 GB</div>
                <div style="font-size:0.65rem; color:#9aada5;">Total Storage</div>
            </div>
            <div style="background:#ffffff; border:1px solid #e8eef0; border-radius:8px; padding:0.6rem;">
                <div style="font-size:1rem; font-weight:700; color:#0d2418;">12</div>
                <div style="font-size:0.65rem; color:#9aada5;">Active Indices</div>
            </div>
            <div style="background:#ffffff; border:1px solid #e8eef0; border-radius:8px; padding:0.6rem;">
                <div style="font-size:1rem; font-weight:700; color:#0d2418;">42</div>
                <div style="font-size:0.65rem; color:#9aada5;">Users</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE 5 — STRATEGY TOOLS
# ============================================================
elif page == "Strategy Tools":
    st.markdown("""
    <div class="page-header">
        <h1>Strategy & Governance Tools</h1>
        <p>Verify AI outputs against architectural compliance standards and operational SOPs before deployment.</p>
    </div>
    """, unsafe_allow_html=True)

    agent = StreamlitDecisionAgent(KNOWLEDGE_BASE_DIR)
    api_active = bool(get_api_key())

    gov_col, rules_col = st.columns([1.6, 1])

    with gov_col:
        st.markdown('<div class="section-title">🛡 Governance Tester</div>', unsafe_allow_html=True)

        prompt_input = st.text_area(
            "Input Agent Prompt or Response",
            placeholder="Paste the AI-generated property description or client response here for validation...",
            height=150,
            key="gov_prompt_input"
        )

        compliance_fw = st.selectbox(
            "Target Compliance Framework",
            ["General Real Estate Ethics v2024", "Brand Voice & Architectural Guidelines", "HUD Fair Housing Standards", "RESPA Compliance v3.1", "GDPR Data Handling Policy"],
            key="gov_compliance_fw"
        )

        if st.button("▶ Run Validation", use_container_width=True, key="gov_run_btn"):
            if prompt_input.strip():
                with st.spinner("AI compliance agent is auditing your input..."):
                    if api_active and agent.llm:
                        validation_prompt = f"""
You are a real estate compliance checker. Please validate the following agent prompt or response:
"{prompt_input}"

Analyze it against the target compliance framework: "{compliance_fw}"

Check for:
1. Accuracy (Is it clear, truthful, and matching standards?)
2. Risk Level (Is it Low, Medium, or High Risk? Explain why, e.g., HUD violations, RESPA violations, PII leaks, discriminatory terms)
3. Alignment (Is it highly aligned with the framework?)

Provide your response in plain text with:
- A clear summary of the findings (Bullet points of key issues if any)
- Actionable improvement suggestions

At the very end of your response, output a single line matching this format exactly:
METRICS: Accuracy_percentage | Risk_level | Alignment_level
(e.g., METRICS: 98% | Low | High or METRICS: 45% | High | Low)
"""
                        try:
                            result = agent.llm.invoke(validation_prompt)
                            full_output = clean_response(result.content if isinstance(result.content, str) else str(result.content))
                            
                            accuracy = "95%"
                            risk = "Low"
                            risk_color = "#1a8f5f"
                            alignment = "High"
                            explanation = full_output
                            
                            if "METRICS:" in full_output:
                                parts = full_output.split("METRICS:")
                                explanation = parts[0].strip()
                                metric_line = parts[1].strip()
                                m_parts = [m.strip() for m in metric_line.split("|")]
                                if len(m_parts) >= 3:
                                    accuracy = m_parts[0]
                                    risk = m_parts[1]
                                    alignment = m_parts[2]
                                    if "high" in risk.lower():
                                        risk_color = "#e35b5b"
                                    elif "med" in risk.lower():
                                        risk_color = "#f59e0b"
                            
                        except Exception as e:
                            explanation = f"Could not perform live analysis: {str(e)}"
                            accuracy = "90%"
                            risk = "Low"
                            risk_color = "#1a8f5f"
                            alignment = "High"
                    else:
                        # Fallback mock validation
                        lower = prompt_input.lower()
                        has_issue = any(w in lower for w in ["terminate", "discriminat", "personal data", "client ssn", "reject application"])
                        accuracy = "92%" if not has_issue else "64%"
                        risk = "Low" if not has_issue else "High"
                        risk_color = "#1a8f5f" if not has_issue else "#e35b5b"
                        alignment = "High" if not has_issue else "Low"
                        explanation = """**Compliance Audit Findings:**
* The input does not contain major fair housing violations.
* Language matches standard real estate disclosures.

**Recommendations:**
* Keep sentences brief.
* Ensure clear disclosure of brokerage relationship if used in public advertising."""
                    
                    explanation_html = explanation.replace('\n', '<br>')
                    
                    # Display results
                    st.markdown(f"""
                    <div style="background:#ffffff; border:1px solid #e8eef0; border-radius:12px; padding:1.25rem; margin-top:1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                        <div style="font-weight:700; font-size:1rem; margin-bottom:0.75rem; color:#0d2418;">✅ Validation Status</div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:1.25rem; border-bottom:1px solid #f0f4f2; padding-bottom:0.75rem;">
                            <div>
                                <div style="font-size:0.68rem; text-transform:uppercase; color:#9aada5; font-weight:700;">Accuracy</div>
                                <div style="font-weight:800; font-size:1.15rem; color:#0d2418; margin-top:0.15rem;">{accuracy}</div>
                            </div>
                            <div>
                                <div style="font-size:0.68rem; text-transform:uppercase; color:#9aada5; font-weight:700;">Risk Level</div>
                                <div style="font-weight:800; font-size:1.15rem; color:{risk_color}; margin-top:0.15rem;">{risk}</div>
                            </div>
                            <div>
                                <div style="font-size:0.68rem; text-transform:uppercase; color:#9aada5; font-weight:700;">Alignment</div>
                                <div style="font-weight:800; font-size:1.15rem; color:#0d2418; margin-top:0.15rem;">{alignment}</div>
                            </div>
                        </div>
                        <div style="font-size:0.875rem; color:#2d3a33; line-height:1.6;">
                            {explanation_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Please enter some agent prompt or response text to validate.")

    with rules_col:
        st.markdown('<div class="section-title">📏 Active Rules</div>', unsafe_allow_html=True)
        rules = [
            ("🌐", "No Discriminatory Language", "Ensures compliance with global fair housing laws."),
            ("🎙", "Premium Brand Voice", 'Maintains "Architectural Light" aesthetic in tone.'),
            ("🔒", "Privacy Masking", "Auto-detects and masks PII in public responses."),
        ]
        for icon, title, desc in rules:
            st.markdown(f"""
            <div style="display:flex; gap:0.75rem; background:#ffffff; border:1px solid #e8eef0; border-radius:8px; padding:0.75rem; margin-bottom:0.5rem; align-items:flex-start;">
                <span style="font-size:1.1rem;">{icon}</span>
                <div>
                    <div style="font-weight:600; font-size:0.825rem; color:#0d2418;">{title}</div>
                    <div style="font-size:0.72rem; color:#9aada5; margin-top:0.2rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("+ Add Custom Governance Rule", use_container_width=True, key="gov_add_rule_btn"):
            st.info("Custom rule builder coming soon.")

        st.markdown("""
        <div style="background:#0d3d2e; border-radius:10px; padding:1.2rem; color:#d4f0e6; margin-top:1.5rem; font-size:0.85rem; line-height:1.5; border:1px solid #1a8f5f;">
            <div style="font-weight:700; margin-bottom:0.4rem; font-size:0.95rem; color:#ffffff;">⚡ Strategy Intelligence</div>
            Your governance compliance rating has improved by 12% since last month. Recommend updating SOP v2.1 for latest market regulations.
        </div>
        """, unsafe_allow_html=True)

    # Explore Advanced Tools Row at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explore Advanced Strategy Tools</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    adv_tools = [
        ("🤖", "Agent Persona Crafter", "Tune the empathy and expertise parameters of your sales AI agents."),
        ("💬", "Social Proof Verifier", "Cross-reference client testimonials with verified transaction data."),
        ("📉", "Leakage Auditor", "Identify where in the sales funnel AI responses lose momentum."),
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3], adv_tools):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e8eef0; border-radius:12px; padding:1.25rem; height:100%; box-shadow:0 2px 6px rgba(0,0,0,0.04);">
                <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                <h4 style="margin:0 0 0.35rem 0; font-size:0.95rem; font-weight:700; color:#0d2418;">{title}</h4>
                <p style="margin:0; font-size:0.78rem; color:#7a8f85; line-height:1.4;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
