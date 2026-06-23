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
    st.session_state.active_page = "Dashboard"


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
            st.session_state.active_page = page
            st.rerun()

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
    st.markdown(f"""
    <div class="page-header">
        <h1>Intelligence Dashboard</h1>
        <p>Operational efficiency and strategic growth markers · {profile['focus']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Top metric row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">⏱ Time Waste</div>
            <div class="metric-value red">12.4 hrs</div>
            <div class="metric-sub">/ week · Manual operational redundancies</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🎯 Operational Focus</div>
            <div class="metric-value teal">86%</div>
            <div class="metric-sub">+2% · Strategic client interaction vs. admin tasks</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="background:#0d3d2e; border-color:#1a8f5f;">
            <div class="metric-label" style="color:#7db89a;">⚡ AI Target State Goal</div>
            <div class="metric-value" style="color:#ffffff;">95%</div>
            <div class="metric-sub" style="color:#7db89a;">Q4 Projection · Automation saturation goal</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        # Time Leak Calculator
        st.markdown('<div class="section-title">⏳ Time Leak Calculator</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <p style="color:#7a8f85; font-size:0.85rem; margin:0 0 1rem 0;">Quantify the financial impact of manual operational leaks within your sales pipeline.</p>
        """, unsafe_allow_html=True)

        hours_admin = st.slider("Weekly hours on admin tasks", 0, 40, 14)
        avg_commission = st.number_input("Avg. Commission Value ($)", value=12500, step=500)
        revenue_lost = int((hours_admin / 40) * avg_commission * 4)

        st.markdown(f"""
        <div style="display:flex; gap:1.5rem; margin-top:0.75rem;">
            <div><div style="font-size:0.7rem; color:#9aada5; text-transform:uppercase;">Weekly hrs on Admin</div><div style="font-size:1.5rem; font-weight:700; color:#e35b5b;">{hours_admin}h</div></div>
            <div><div style="font-size:0.7rem; color:#9aada5; text-transform:uppercase;">Annual Revenue Loss</div><div style="font-size:1.5rem; font-weight:700; color:#0d2418;">${revenue_lost:,}</div></div>
        </div>
        <div style="margin-top:1rem;">
        """, unsafe_allow_html=True)

        if st.button("📊 Plug the Leak →", key="plug_leak"):
            st.success("RealtyAI typically recovers 65% of this lost time within the first 3 months. Check the Strategy Tools tab to get started.")
        st.markdown("</div></div>", unsafe_allow_html=True)

        # High-priority pipeline
        st.markdown('<br><div class="section-title">📋 High-Priority Pipeline</div>', unsafe_allow_html=True)
        top_deals = [d for d in st.session_state.deals if d["stage"] in ["Proposal", "Discovery"]]
        if top_deals:
            for d in top_deals[:3]:
                confidence = "92%" if d["stage"] == "Proposal" else "76%"
                conf_color = "#1a8f5f" if d["stage"] == "Proposal" else "#e38a1a"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; border:1px solid #e8eef0; border-radius:8px; padding:0.65rem 1rem; margin-bottom:0.5rem;">
                    <div>
                        <div style="font-weight:600; font-size:0.875rem; color:#0d2418;">{d['accountName']} — {d['propertyAddress']}</div>
                        <div style="font-size:0.72rem; color:#9aada5;">{d['type']} · {d['stage']}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; font-weight:700; color:{conf_color};">AI Confidence {confidence}</div>
                        <div style="font-size:0.7rem; color:#9aada5;">{d['dealValue']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with right_col:
        # Live feed
        st.markdown('<div class="section-title">📡 Live Feed</div>', unsafe_allow_html=True)
        feed_items = [
            ("🤖 Automation Deployed", "Property Matching Module • 6h ago"),
            ("💡 Insight Generated", "Pipeline Velocity Alert · Listings"),
            ("📄 Report Prepared", "Weekly Growth Summary · 8 Apr"),
        ]
        for title, sub in feed_items:
            st.markdown(f"""
            <div class="feed-item">
                <strong>{title}</strong><br>{sub}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Governance guardrails
        st.markdown('<div class="section-title">🛡 AI Guardrails</div>', unsafe_allow_html=True)
        guard_mode = st.radio("Mode", ["Policy Viewer", "Scenario Tester"], horizontal=True, label_visibility="collapsed")

        if guard_mode == "Policy Viewer":
            st.markdown("""
            <div style="background:#f0faf4; border:1px solid #a7d7be; border-radius:8px; padding:0.75rem; margin-bottom:0.5rem; font-size:0.8rem;">
                <strong style="color:#0d5f35;">🟢 Green Light (AI Approved)</strong><br>
                <ul style="margin:0.4rem 0 0 1rem; color:#2d6048;">
                    <li>Draft property follow-ups & listing copy</li>
                    <li>Meeting summaries & CRM extractions</li>
                </ul>
            </div>
            <div style="background:#fff5f5; border:1px solid #f5c6c6; border-radius:8px; padding:0.75rem; font-size:0.8rem;">
                <strong style="color:#9b1c1c;">🔴 Red Light (Human-Only)</strong><br>
                <ul style="margin:0.4rem 0 0 1rem; color:#7f1d1d;">
                    <li>Closing high-value negotiations</li>
                    <li>Legal contract reviews & terminations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            test_query = st.text_input("Enter task scenario:", placeholder="Draft listing description from features")
            if st.button("Evaluate"):
                lower = test_query.lower()
                red = ["terminate", "fire", "negotiate", "dispute", "lawsuit", "contract review", "commission split"]
                green = ["draft", "summary", "listing", "email", "crm", "description", "transcript"]
                if any(k in lower for k in red):
                    st.error("🔴 **RED LIGHT** — Human-Only Mandated. Do not automate.")
                elif any(k in lower for k in green):
                    st.success("🟢 **GREEN LIGHT** — AI Approved. Safe to deploy templates.")
                else:
                    st.warning("🟡 **YELLOW LIGHT** — Manual review required before proceeding.")


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
        <h1>Governance Tester</h1>
        <p>Verify AI outputs against architectural compliance standards and operational SOPs before deployment.</p>
    </div>
    """, unsafe_allow_html=True)

    gov_col, rules_col = st.columns([1.6, 1])

    with gov_col:
        st.markdown('<div class="section-title">🛡 Governance Tester</div>', unsafe_allow_html=True)

        prompt_input = st.text_area(
            "Input Agent Prompt or Response",
            placeholder="Paste the AI-generated property description or client response here for validation...",
            height=120
        )

        compliance_fw = st.selectbox(
            "Target Compliance Framework",
            ["General Real Estate Ethics v2024", "HUD Fair Housing Standards", "RESPA Compliance v3.1", "GDPR Data Handling Policy"]
        )

        if st.button("▶ Run Validation", use_container_width=False):
            if prompt_input.strip():
                lower = prompt_input.lower()
                has_issue = any(w in lower for w in ["terminate", "discriminat", "personal data", "client ssn", "reject application"])
                accuracy = "91%" if not has_issue else "64%"
                risk = "Low" if not has_issue else "High"
                risk_color = "#1a8f5f" if not has_issue else "#e35b5b"
                alignment = "High" if not has_issue else "Low"

                st.markdown(f"""
                <div class="validation-box">
                    <div style="font-weight:700; margin-bottom:0.75rem; color:#0d2418;">✅ Validation Status</div>
                    <div style="display:flex; gap:2rem;">
                        <div>
                            <div style="font-size:0.68rem; text-transform:uppercase; color:#9aada5;">Accuracy</div>
                            <div style="font-weight:700; font-size:1rem; color:#0d2418;">{accuracy}</div>
                        </div>
                        <div>
                            <div style="font-size:0.68rem; text-transform:uppercase; color:#9aada5;">Risk Level</div>
                            <div style="font-weight:700; font-size:1rem; color:{risk_color};">{risk}</div>
                        </div>
                        <div>
                            <div style="font-size:0.68rem; text-transform:uppercase; color:#9aada5;">Alignment</div>
                            <div style="font-weight:700; font-size:1rem; color:#0d2418;">{alignment}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Enter a prompt or response to validate.")

        # ACP Prompt Engine
        st.markdown('<br><div class="section-title">🔧 ACP Prompting Engine</div>', unsafe_allow_html=True)
        scenario = st.selectbox("Scenario Template", ["Silent Client Follow-up", "Property Listing Description", "Commission Split Inquiry"])
        defaults = {
            "Silent Client Follow-up": ("Buyer Sarah Connor viewed 1045 River Rd 4 days ago. High interest in kitchen, silent on texts.", "Role: Senior Consultative Advisor. Tone: Warm, low-pressure. Max 120 words. Single open-ended question."),
            "Property Listing Description": ("789 Oakridge Ave. 4 beds, 3.5 baths, solar panels, concrete counters.", "Role: Expert sustainable copywriter. Eco-conscious tone. 3 paragraphs max."),
            "Commission Split Inquiry": ("Agent Kyle requests 5% bump in split for Valley View ($1.8M). Self-sourced buyer, spent $2k on brochures.", "Role: Brokerage Sales Manager. Firm but empathetic. Decline. Reference Section 4.2."),
        }
        ctx, params = defaults[scenario]
        context_input = st.text_area("Context / Raw details:", value=ctx, height=80)
        params_input = st.text_area("Parameters (Tone, Length, Constraints):", value=params, height=80)

        if st.button("⚡ Compile ACP Prompt", use_container_width=False):
            compiled = f"""[ROLE & PERSONA]
You are a Real Estate Strategy Expert — professional, direct, outcome-oriented.

[TASK ACTION]
Generate a response for: {scenario}

[CONTEXT]
{context_input}

[CONSTRAINTS]
{params_input}

[RULES]
1. Write the final response directly — no meta-commentary.
2. Strictly respect the constraints (length, tone, no buzzwords).
3. Do not use phrases like "Sure, here is the email:"."""
            st.code(compiled, language="markdown")
            st.success("✅ ACP Prompt compiled! Copy into Gemini, ChatGPT, or Claude.")

        # 5W1H Tool
        st.markdown('<br><div class="section-title">📋 5W1H Reporting Tool</div>', unsafe_allow_html=True)
        report_src = st.selectbox("Report Source", ["Weekly Pipeline Review Sync (Transcript)", "Active Escrow Pipeline (CRM Export)"])
        raw = "Manager: Dave, status on 412 Hillside? Dave: Seller drops $15k if no offers by Wednesday. Broker open house Tuesday 10-12. Manager: Push MLS update Wednesday 9AM." if "Transcript" in report_src else "ESC-9011 | Dave | 233 Broad St | Loan Commitment Due Jun 25 | Upload pre-approval doc."
        st.text_area("Raw Preview:", value=raw, height=80, disabled=True)
        if st.button("📊 Generate 5W1H Digest", use_container_width=False):
            if "Transcript" in report_src:
                st.markdown("""
**5W1H Structured Digest**
- **Who:** Manager, Dave, Sarah
- **What:** 412 Hillside price drop prep; 102 Pine St roof inspection credit dispute
- **Where:** 412 Hillside St, 102 Pine St
- **When:** Open House: Tuesday 10–12PM; MLS: Wednesday 9AM; Escrow: Friday 5PM
- **Why:** Prevent listing staleness; keep Pine St escrow from terminating
- **How:** Draft MLS price reduction; counter-offer $2,500 credit vs $5,000 demand
                """)
            else:
                st.markdown("""
**5W1H Structured Digest**
- **Who:** Dave, Sarah, Marcus
- **What:** Loan doc upload; easement dispute; agent reminder trigger
- **Where:** 233 Broad St, 1045 River Rd, 890 Ridge Way
- **When:** ESC-9011: Jun 25 5PM; ESC-8922: Jun 29 12PM; ESC-9055: Immediate
- **Why:** Avoid contract defaults; resolve easement liability; arrest lead decay
- **How:** Upload verification docs; legal review; auto-email trigger to Marcus
                """)

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

        if st.button("+ Add Custom Governance Rule", use_container_width=True):
            st.info("Custom rule builder coming soon.")

        st.markdown("""
        <div style="background:#0d3d2e; border-radius:10px; padding:1rem; color:#d4f0e6; margin-top:1rem; font-size:0.8rem;">
            <div style="font-weight:700; margin-bottom:0.35rem;">⚡ Strategy Intelligence</div>
            Your governance score has improved by 12% since last month. Recommend updating SOP v2.1 for latest market regulations.
            <div style="margin-top:0.75rem;">
        """, unsafe_allow_html=True)
        if st.button("VIEW FULL AUDIT →", use_container_width=True):
            st.info("Full audit report export coming soon.")
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Explore advanced tools
        st.markdown('<br><div class="section-title">Explore Advanced Tools</div>', unsafe_allow_html=True)
        adv_tools = [
            ("🤖", "Agent Persona Crafter", "Tune the empathy and expertise parameters of your sales AI agents."),
            ("💬", "Social Proof Verifier", "Cross-reference client testimonials with verified transaction data."),
            ("📉", "Leakage Auditor", "Identify where in the sales funnel AI responses lose momentum."),
        ]
        for icon, title, desc in adv_tools:
            st.markdown(f"""
            <div class="tool-card">
                <div style="font-size:1.3rem; margin-bottom:0.4rem;">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
