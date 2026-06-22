import streamlit as st
import os
import shutil
import json
from pathlib import Path
from typing import List

# Import LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# Load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- CONSTANTS & DIRECTORIES ---
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="RealtyAI - Sales Manager Playbook",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphic styling
st.markdown("""
<style>
    /* Dark glass style containers */
    .metric-card {
        background: rgba(30, 37, 58, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .deal-card {
        background: rgba(25, 30, 50, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .deal-title {
        font-weight: 600;
        color: #ffffff;
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
    }
    .deal-desc {
        font-size: 0.8rem;
        color: #8e9bb4;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .deal-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        padding-top: 0.4rem;
        margin-top: 0.4rem;
    }
    .deal-val {
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .deal-act {
        color: #3dd9b4;
        background: rgba(61, 217, 180, 0.06);
        padding: 1px 5px;
        border-radius: 4px;
        border: 1px solid rgba(61, 217, 180, 0.15);
        font-size: 0.72rem;
    }
</style>
""", unsafe_style_html=True)


# --- SESSION STATE INITIALIZATION ---
if "profile" not in st.session_state:
    st.session_state.profile = None

if "deals" not in st.session_state:
    # Initial seed data matching index.html
    st.session_state.deals = [
        {
            "id": 1,
            "accountName": "Northwind Retail",
            "propertyAddress": "120 Oak St",
            "dealValue": "$18k GCI",
            "stage": "Lead",
            "context": "Buyer toured 120 Oak St. Follow-up email scheduled. Lead source: Zillow Inbound.",
            "nextStep": "MLS Email follow-up"
        },
        {
            "id": 2,
            "accountName": "Blue Harbor Logistics",
            "propertyAddress": "Valley Lot",
            "dealValue": "$24k GCI",
            "stage": "Lead",
            "context": "Requested details on CRE industrial lot. Outbound reply sent.",
            "nextStep": "Call client Monday"
        },
        {
            "id": 3,
            "accountName": "Atlas Components",
            "propertyAddress": "440 River Road",
            "dealValue": "$42k GCI",
            "stage": "Discovery",
            "context": "Toured 440 River Road. Budget verified. Pre-approval letters uploaded.",
            "nextStep": "Send disclosure package"
        },
        {
            "id": 4,
            "accountName": "Pioneer Med",
            "propertyAddress": "10 Pine St",
            "dealValue": "$31k GCI",
            "stage": "Discovery",
            "context": "Consultation call complete. Specific 3-bath property requirements noted.",
            "nextStep": "Schedule weekend tours"
        },
        {
            "id": 5,
            "accountName": "Summit Hospitality",
            "propertyAddress": "555 Valley View",
            "dealValue": "$55k GCI",
            "stage": "Proposal",
            "context": "Broker opinion of value generated. Listing agreement proposal sent.",
            "nextStep": "Negotiate commission split"
        },
        {
            "id": 6,
            "accountName": "Greenline Foods",
            "propertyAddress": "1045 River Rd",
            "dealValue": "$64k GCI",
            "stage": "Closed Won",
            "context": "Represented buyer. Escrow closed, funds wired, keys delivered.",
            "nextStep": "Archive client file"
        }
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "ai", "content": "Hello! Ask a question and I will search your sales playbook for tactical guidance.", "source": None}
    ]

if "nb_chat_history" not in st.session_state:
    st.session_state.nb_chat_history = [
        {"role": "ai", "content": "Hello! I am NotebookLM, grounded in your loaded team SOPs. Ask me a question to simulate query answers or click one of the suggestions below.", "citations": []}
    ]


# --- RAG DECISION AGENT CLASS ---
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
        # Read API key from Streamlit secrets, environment, or user state
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key and "gemini_key" in st.session_state:
            api_key = st.session_state.gemini_key

        if api_key:
            model_name = os.getenv("GEMINI_MODEL", "gemma-2-27b-it")
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                temperature=0.2
            )
            self.llm_provider = "gemini"

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
        for loader in [txt_loader, md_loader, pdf_loader]:
            try:
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

        # Uses FakeEmbeddings since Gemini compatible embeddings aren't supported
        embeddings = FakeEmbeddings(size=1536)
        
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "sales_playbook_retriever",
            "Search the sales strategy knowledge base for tactical guidance.",
        )

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
        try:
            result = self.llm.invoke(prompt)
            content = result.content if isinstance(result.content, str) else str(result.content)
            
            # Simple JSON parse extraction
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start:end+1])
            return json.loads(content)
        except Exception as e:
            return {
                "sufficient": True,
                "reasoning": f"Parsing/LLM Error: {e}",
                "answer": "Error communicating with Gemini model. Please double check your API key.",
                "confidence": "low"
            }

    def ask(self, query: str, role: str, experience: str) -> dict:
        if not self.llm:
            return {
                "answer": "AI Coach is in Offline Mode. Please add your Gemini API Key in the sidebar to retrieve playbook insights.",
                "source": "Local System Indicator",
                "sufficient": False,
                "reasoning": "No LLM loaded."
            }
            
        if not self.retriever:
            return {
                "answer": "The knowledge base index is empty. Please upload some files in the AI Coach tab to enrich the RAG agent.",
                "source": "System RAG Loader",
                "sufficient": False,
                "reasoning": "No files found."
            }

        documents = self.retriever.invoke(query)
        sources = sorted({Path(doc.metadata.get("source", "Unknown source")).name for doc in documents})
        context = "\n\n".join(doc.page_content for doc in documents)

        if context.strip():
            decision = self._llm_decision(query, context, role, experience)
            return {
                "answer": decision.get("answer", "No answer found."),
                "source": ", ".join(sources),
                "sufficient": decision.get("sufficient", False),
                "reasoning": decision.get("reasoning", "Parsed from RAG content.")
            }
            
        return {
            "answer": "I could not find enough relevant information in the knowledge base for that query. Try uploading more strategy files.",
            "source": "Fallback Matcher",
            "sufficient": False,
            "reasoning": "Zero document hits."
        }


# --- PROFILE ONBOARDING DIALOG ---
@st.dialog("Welcome to RealtyAI Onboarding")
def onboarding_modal():
    st.write("Configure your sales workspace to personalize the RAG playbook templates and AI coach responses.")
    name = st.text_input("Manager Name", placeholder="Dave Smith")
    focus = st.selectbox("Real Estate Focus", ["Residential Brokerage", "Commercial Real Estate (CRE)", "Property Developer"])
    exp = st.selectbox("AI Maturity Level / Experience", ["Junior User (Occasional AI)", "Mid Operator (Structured prompting)", "Senior Builder (GPTs & RAG)"])
    savings = st.number_input("Weekly Time Saving Target (Hours)", min_value=5, max_value=30, value=16)
    
    st.info("You can also add your Gemini API Key below to activate the RAG coach chatbot.")
    gemini_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key from Google AI Studio")

    if st.button("Launch Playbook Workspace"):
        if name.strip() == "":
            st.error("Please enter your name.")
            return
        
        st.session_state.profile = {
            "name": name,
            "focus": focus,
            "experience": exp,
            "targetSavings": savings
        }
        if gemini_key:
            st.session_state.gemini_key = gemini_key
        st.rerun()


# --- SIDEBAR WIDGETS ---
with st.sidebar:
    st.markdown('<div style="display: flex; align-items: center; gap: 8px;"><div style="background:#3dd9b4; color:#0f1322; width:32px; height:32px; border-radius:6px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; box-shadow:0 0 10px rgba(61,217,180,0.4)">⚡</div><h2 style="margin:0; font-size:1.25rem;">RealtyAI</h2></div>', unsafe_style_html=True)
    st.markdown('<span style="font-size:0.65rem; color:#3dd9b4; text-transform:uppercase; letter-spacing:1px;">Playbook Engine</span>', unsafe_style_html=True)
    st.write("---")

    if not st.session_state.profile:
        st.write("👋 Welcome! Please configure your profile to personalize the playbook.")
        if st.button("Run Profile Setup"):
            onboarding_modal()
    else:
        # Show Profile Info Card
        st.markdown(f"**Manager:** {st.session_state.profile['name']}")
        st.markdown(f"**Focus:** {st.session_state.profile['focus']}")
        st.markdown(f"**Exp:** {st.session_state.profile['experience']}")
        
        if st.button("Edit Workspace Profile"):
            onboarding_modal()
            
    st.write("---")
    
    # API key override
    st.subheader("API Keys")
    api_placeholder = st.session_state.get("gemini_key", "")
    key_val = st.text_input("Gemini API Key", type="password", value=api_placeholder, placeholder="Grounded RAG Key")
    if key_val and key_val != api_placeholder:
        st.session_state.gemini_key = key_val
        st.success("API key updated!")
        
    st.write("---")
    
    # Weekly progress bar widget
    st.subheader("Weekly Reclaimed Time")
    
    # Simple hour calculation base toggles
    meetings_tog = st.checkbox("MS Teams Recap (Meetings)", value=True, help="Saves ~5h/wk")
    emails_tog = st.checkbox("ACP Prompting (Emails)", value=False, help="Saves ~4h/wk")
    data_tog = st.checkbox("Copilot in Excel (Data Entry)", value=False, help="Saves ~3.5h/wk")
    searching_tog = st.checkbox("NotebookLM SOP (Search)", value=True, help="Saves ~2.5h/wk")
    
    reclaimed = 0.0
    if meetings_tog: reclaimed += 5.0
    if emails_tog: reclaimed += 4.0
    if data_tog: reclaimed += 3.5
    if searching_tog: reclaimed += 2.5
    
    target_hours = st.session_state.profile["targetSavings"] if st.session_state.profile else 16.0
    progress_val = min(reclaimed / target_hours, 1.0)
    
    st.progress(progress_val)
    st.write(f"**Reclaimed:** {reclaimed:.1f}h / {target_hours:.0f}h")
    st.write(f"Progress: {int(progress_val*100)}%")


# --- MAIN TABS ROUTING ---
st.title("Sales Manager AI Playbook")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📋 Pipeline Board", "💬 AI Coach", "📚 SOP Knowledge Base", "🛠 Strategy Tools"])


# --- TAB 1: DASHBOARD ---
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #f87171;">
            <div style="font-size:0.75rem; color:#8e9bb4; font-weight:600; text-transform:uppercase;">Estimated Time Waste</div>
            <h3 style="font-size:1.6rem; color:#f87171; margin-top:0.25rem; font-weight:800;">{max(12.0 - reclaimed, 0.0):.1f}-{max(18.0 - reclaimed, 0.0):.1f} Hours</h3>
            <span style="font-size:0.72rem; color:#8e9bb4;">Leaking Current Week</span>
        </div>
        """, unsafe_style_html=True)
        
    with col2:
        focus_goal = "Maximize GCI"
        focus_sub = "Reduce Days on Market"
        if st.session_state.profile:
            if st.session_state.profile["focus"] == "Commercial Real Estate (CRE)":
                focus_goal = "Broker Deals"
                focus_sub = "Optimize Lease Volume"
            elif st.session_state.profile["focus"] == "Property Developer":
                focus_goal = "Sell Inventory"
                focus_sub = "Onboard Off-Plan Units"
                
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #38bdf8;">
            <div style="font-size:0.75rem; color:#8e9bb4; font-weight:600; text-transform:uppercase;">Operational Focus Goal</div>
            <h3 style="font-size:1.6rem; color:#ffffff; margin-top:0.25rem; font-weight:800;">{focus_goal}</h3>
            <span style="font-size:0.72rem; color:#8e9bb4;">{focus_sub}</span>
        </div>
        """, unsafe_style_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #3dd9b4;">
            <div style="font-size:0.75rem; color:#8e9bb4; font-weight:600; text-transform:uppercase;">AI Target State Goal</div>
            <h3 style="font-size:1.6rem; color:#3dd9b4; margin-top:0.25rem; font-weight:800;">Reclaim {int((reclaimed/target_hours)*100)}%</h3>
            <span style="font-size:0.72rem; color:#8e9bb4;">~{reclaimed:.1f}h of {target_hours:.0f}h Target</span>
        </div>
        """, unsafe_style_html=True)

    # Split: Time leak table & Guardrails
    dash_col1, dash_col2 = st.columns([1.2, 1])
    
    with dash_col1:
        st.subheader("Operational Time Leak Breakdown")
        st.write("Deploy target AI solutions to reclaim hours from recurring organizational drag activities:")
        
        st.markdown(f"""
        | Activity | Weekly Leak | Target AI Solution | Status |
        |---|---|---|---|
        | **Inefficient Meetings** | 4-6 Hours | MS Teams Recap | {'🟢 Deployed' if meetings_tog else '🔴 Stalled'} |
        | **Repetitive Emails** | 3-5 Hours | ACP Prompt Engine | {'🟢 Deployed' if emails_tog else '🔴 Stalled'} |
        | **Manual Data Entry** | 3-4 Hours | Copilot in Excel | {'🟢 Deployed' if data_tog else '🔴 Stalled'} |
        | **Searching SOPs** | 2-3 Hours | NotebookLM SOP Chat | {'🟢 Deployed' if searching_tog else '🔴 Stalled'} |
        """, unsafe_style_html=True)
        
    with dash_col2:
        st.subheader("AI Governance & Usage Guardrails")
        guard_mode = st.radio("Guardrail Mode", ["Policy Viewer", "Scenario Tester"], horizontal=True)
        
        if guard_mode == "Policy Viewer":
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.success("🟢 Green Light (Approved)")
                st.write("""
                - Drafting property follow-ups
                - Meeting summaries
                - CRM metric extractions
                - Generating listing copy
                """)
            with col_g2:
                st.error("🔴 Red Light (Human-Only)")
                st.write("""
                - Closing high-value negotiations
                - Dispute split arbitration
                - Final legal contract reviews
                - Agent terminations
                """)
        else:
            st.write("Evaluate a task for AI suitability instantly:")
            
            # Quick Scenario buttons
            quick_test = st.selectbox("Quick Scenarios", ["-- Select a test task --", "Draft listing description from features", "Terminate an agent for performance", "Extract monthly GCI count", "Resolve broker split commission dispute"])
            
            scenario_text = ""
            if quick_test != "-- Select a test task --":
                scenario_text = quick_test
                
            test_query = st.text_input("Enter your custom task scenario:", value=scenario_text)
            
            if st.button("Evaluate Suitability"):
                lower = test_query.lower()
                red_keywords = ["terminate", "fire", "promote", "hire", "negotiate", "dispute", "lawsuit", "contract review", "commission split", "firing"]
                green_keywords = ["draft", "summary", "transcript", "listing", "email", "crm", "metric", "description", "write follow-up"]
                
                if any(kw in lower for kw in red_keywords):
                    st.error("🔴 **RED LIGHT: Human-Only Mandated**\n\nCRITICAL: This activity involves legally binding negotiations or sensitive HR/personnel management. Do not automate with AI.")
                elif any(kw in lower for kw in green_keywords):
                    st.success("🟢 **GREEN LIGHT: AI Approved Task**\n\nThis task is administrative or drafts low-stakes copy. You can deploy AI templates safely.")
                else:
                    st.warning("🟡 **YELLOW LIGHT: Conditional Policy Check**\n\nSafety parameters could not be verified automatically. Check results thoroughly and remove any PII before using AI.")


# --- TAB 2: PIPELINE BOARD ---
with tab2:
    st.subheader("Deal & Listing Pipeline")
    st.write("Track clients, property listings, and active deals across stages:")
    
    # Deal Wizard Form
    with st.expander("➕ Launch Deal Wizard (Add New Deal Card)"):
        with st.form("new_deal_form"):
            c_name = st.text_input("Client Name", placeholder="Sarah Connor")
            c_addr = st.text_input("Property Address", placeholder="1045 River Rd")
            c_val = st.text_input("Listing Price / GCI Value", placeholder="$450,000 / $18k GCI")
            c_stage = st.selectbox("Pipeline Stage", ["Lead", "Discovery", "Proposal", "Closed Won"])
            c_context = st.text_area("Context / Negotiation Details", placeholder="Buyer liked layout but requested roof credit.")
            c_next = st.text_input("Next Action", placeholder="Send inspector details")
            
            submit_deal = st.form_submit_button("Create Deal Card")
            if submit_deal:
                if c_name.strip() == "" or c_addr.strip() == "":
                    st.error("Please fill in Client Name and Property Address.")
                else:
                    new_id = max([d["id"] for d in st.session_state.deals]) + 1 if st.session_state.deals else 1
                    st.session_state.deals.append({
                        "id": new_id,
                        "accountName": c_name,
                        "propertyAddress": c_addr,
                        "dealValue": c_val,
                        "stage": c_stage,
                        "context": c_context,
                        "nextStep": c_next
                    })
                    st.success(f"Deal for {c_name} created!")
                    st.rerun()

    # Columns display
    col_lead, col_disc, col_prop, col_won = st.columns(4)
    stages = [("Lead", col_lead), ("Discovery", col_disc), ("Proposal", col_prop), ("Closed Won", col_won)]
    
    for stage_name, col in stages:
        with col:
            stage_deals = [d for d in st.session_state.deals if d["stage"] == stage_name]
            st.markdown(f"### {stage_name} ({len(stage_deals)})")
            st.write("---")
            
            for deal in stage_deals:
                # Render deal container card
                with st.container():
                    st.markdown(f"""
                    <div class="deal-card">
                        <div class="deal-title">{deal['accountName']}</div>
                        <div class="deal-desc">{deal['propertyAddress']} - {deal['context']}</div>
                        <div class="deal-meta">
                            <span class="deal-val">{deal['dealValue']}</span>
                            <span class="deal-act">{deal['nextStep']}</span>
                        </div>
                    </div>
                    """, unsafe_style_html=True)
                    
                    # Selectbox to move stage (simulating drag-and-drop actions)
                    idx = ["Lead", "Discovery", "Proposal", "Closed Won"].index(stage_name)
                    new_stage = st.selectbox("Move stage:", ["Lead", "Discovery", "Proposal", "Closed Won"], index=idx, key=f"move_{deal['id']}")
                    
                    if new_stage != stage_name:
                        # Update stage
                        for d in st.session_state.deals:
                            if d["id"] == deal["id"]:
                                d["stage"] = new_stage
                        st.rerun()


# --- TAB 3: AI COACH ---
with tab3:
    st.subheader("AI Playbook Assistant & RAG Index")
    
    coach_col1, coach_col2 = st.columns([2, 1])
    
    # Initialize Agent
    agent = StreamlitDecisionAgent(KNOWLEDGE_BASE_DIR)
    
    with coach_col1:
        st.write("Interact with your RAG Coach grounded in internal SOP documents:")
        
        # Render Chat History
        for msg in st.session_state.chat_history:
            role_label = "👤 User" if msg["role"] == "user" else "🤖 Coach"
            with st.chat_message(msg["role"]):
                st.write(f"**{role_label}**")
                st.write(msg["content"])
                if msg.get("source"):
                    st.caption(f"Source: {msg['source']}")
                    
        # User input query
        query = st.chat_input("Ask a playbook query (e.g. How to handle pricing adjustments?):")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query, "source": None})
            
            # Query backend
            role_str = st.session_state.profile["focus"] if st.session_state.profile else "Sales Manager"
            exp_str = st.session_state.profile["experience"] if st.session_state.profile else "Mid Operator"
            
            response = agent.ask(query, role_str, exp_str)
            
            st.session_state.chat_history.append({
                "role": "ai",
                "content": response["answer"],
                "source": response["source"]
            })
            st.rerun()
            
    with coach_col2:
        st.markdown("### Manage RAG Knowledge Base")
        st.write("Upload strategy documents to index the AI Assistant:")
        
        # File uploader
        uploaded_files = st.file_uploader("Upload .txt, .md, or .pdf files:", accept_multiple_files=True, type=["txt", "md", "pdf"])
        if st.button("Index Files to RAG"):
            if uploaded_files:
                saved_count = 0
                for uploaded_file in uploaded_files:
                    destination = KNOWLEDGE_BASE_DIR / uploaded_file.name
                    with open(destination, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
                
                st.success(f"Indexed {saved_count} file(s) successfully!")
                # Re-index RAG
                agent._setup_retriever()
                st.rerun()
            else:
                st.error("Select files first.")
                
        # Index listing
        st.write("---")
        st.write("**Indexed Files:**")
        files = sorted([f.name for f in KNOWLEDGE_BASE_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".txt", ".md", ".pdf"]])
        if files:
            for f in files:
                st.markdown(f"- 📄 {f}")
        else:
            st.caption("No files indexed yet.")


# --- TAB 4: SOP KNOWLEDGE BASE ---
with tab4:
    st.subheader("SOP Knowledge Base & NotebookLM Simulator")
    
    sop_col1, sop_col2 = st.columns([1.5, 1])
    
    with sop_col1:
        st.write("Select a Standard Operating Procedure (SOP) to review:")
        sop_choice = st.selectbox("Active SOPs:", ["SOP: Lead Handling & Routing", "SOP: Listing Launch Checklist", "SOP: Commission Dispute Resolution"])
        
        if sop_choice == "SOP: Lead Handling & Routing":
            st.markdown("""
            ### SOP: Lead Handling & Routing (Revision 4.1)
            
            #### 1. Lead Response Time SLA
            All inbound web inquiries (Zillow, Realtor.com, personal website landing pages) must be contacted within **15 minutes** of registration during normal business hours (8:00 AM - 7:00 PM). P1-level "Hot Leads" must be called immediately.
            
            #### 2. CRM Logging Protocol
            Every lead contact attempt must be logged in the database CRM under the following guidelines:
            - Record date and time of attempt.
            - Log outcome code (e.g. NA for No Answer, VM for Voicemail, CC for Connected).
            - Schedule follow-up task sequence before closing contact pane.
            
            #### 3. Active Lead Routing Hierarchy
            If an assigned agent fails to contact a client within 2 hours, the lead is automatically reassigned to the secondary on-call listing agent to prevent response time degradation. Dispute claims regarding lead stealing will be routed to the Regional Sales Manager.
            """)
        elif sop_choice == "SOP: Listing Launch Checklist":
            st.markdown("""
            ### SOP: Listing Launch Checklist (Revision 2.0)
            
            #### 1. Marketing Preparations (Day -7 to Day -1)
            Standard listing onboarding workflow checklist:
            1. Schedule HDR photography and 3D walkthrough scanning (must complete by Day -4).
            2. Acquire and verify listing title deed and survey documents.
            3. Draft MLS property description using standard approved formatting rules.
            
            #### 2. Go-Live Procedures (Day 0)
            Listing goes live on MLS by 9:00 AM EST on Thursday. Submit syndication feeds to Zillow, Trulia, and Realtor.com. Install yard signs, lockbox, and flyer cabinets at the property.
            
            #### 3. Post-Launch Actions (Day +1 to Day +3)
            Schedule public Open House for Saturday and Sunday (1:00 PM - 4:00 PM). Launch targeted social media marketing campaigns (minimum budget $100 per listing).
            """)
        else:
            st.markdown("""
            ### SOP: Commission Dispute Resolution (Revision 1.2)
            
            #### 1. Definition of Procuring Cause
            Disputes regarding split commissions are governed by the principle of "Procuring Cause"—the agent who initiated the chain of events that directly led to the final contract signing is entitled to the primary split.
            
            #### 2. Informal Review Process
            Prior to submitting a formal broker dispute, agents must attempt an informal mutual review with the sales manager. The manager acts as a neutral negotiator to establish splitting terms (e.g. 50/50, 75/25 split arrangements).
            
            #### 3. Formal Arbitration Workflow
            If informal review fails, agents must submit a formal written request for arbitration within 10 business days of contract closing. The executive review board will render a final, non-appealable decision within 5 business days of submission.
            """)
            
    with sop_col2:
        st.markdown("### NotebookLM Simulator")
        st.caption("Grounded in Active SOPs")
        
        # Render NotebookLM Chat
        for msg in st.session_state.nb_chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("citations"):
                    st.caption(f"Citations: {', '.join(msg['citations'])}")
                    
        # Suggestions buttons
        st.write("Suggested Queries:")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            q1 = st.button("What is lead SLA limit?")
        with col_s2:
            q2 = st.button("First 24h listing steps?")
            
        nb_query = st.text_input("Ask NotebookLM Simulator:")
        
        if q1: nb_query = "What is the response time limit for a hot web lead?"
        if q2: nb_query = "What are the marketing steps for the first 24 hours of a listing?"
        
        if st.button("Query NotebookLM") or nb_query:
            if nb_query:
                st.session_state.nb_chat_history.append({"role": "user", "content": nb_query, "citations": []})
                
                lower = nb_query.lower()
                answer = "I couldn't find a direct answer to that query in the active SOPs."
                citations = []
                
                if "response time" in lower or "sla" in lower or "limit" in lower or "hot" in lower:
                    answer = "According to **SOP: Lead Handling & Routing (Section 1)**, all inbound web inquiries must be contacted within **15 minutes** of registration. Hot Leads require immediate phone call response."
                    citations = ["SOP: Lead Handling & Routing §1"]
                elif "24 hours" in lower or "listing" in lower or "checklist" in lower or "marketing" in lower:
                    answer = "Based on **SOP: Listing Launch Checklist (Section 2)**, within the first 24 hours of going live (Day 0), you must: 1. Confirm syndication to portals. 2. Install physical yard signs and lockboxes. 3. Setup flyer cabinets."
                    citations = ["SOP: Listing Launch Checklist §2"]
                elif "dispute" in lower or "claim" in lower or "split" in lower or "cause" in lower:
                    answer = "According to **SOP: Commission Dispute Resolution (Section 2)**, agents must attempt an informal review with the manager to reach a split agreement (e.g. 50/50). If unresolved, they must submit written arbitration request within 10 days."
                    citations = ["SOP: Commission Dispute Resolution §2"]
                    
                st.session_state.nb_chat_history.append({
                    "role": "ai",
                    "content": answer,
                    "citations": citations
                })
                st.rerun()


# --- TAB 5: STRATEGY & PROMPTING TOOLS ---
with tab5:
    st.subheader("Strategy & Custom Prompt compiler")
    
    tool_col1, tool_col2 = st.columns(2)
    
    with tool_col1:
        st.markdown("### Action, Context, Parameters (ACP) Prompting Engine")
        st.write("Compile prompts that output clean real estate templates without generic AI filler:")
        
        scenario = st.selectbox("Scenario Template", ["Silent Client Follow-up", "Property Listing Description", "Commission Split Inquiry Response"])
        
        default_context = ""
        default_params = ""
        
        if scenario == "Silent Client Follow-up":
            default_context = "Buyer named Sarah Connor viewed 1045 River Rd 4 days ago. Expressed high interest in chef kitchen and quiet backyard, but has been silent on texts."
            default_params = "Role: Senior Consultative Real Estate Advisor.\nTone: Warm, professional, low-pressure.\nConstraints: Max 120 words. No corporate jargon. Ask a single open-ended question about disclosures."
        elif scenario == "Property Listing Description":
            default_context = "Listing: 789 Oakridge Ave. 4 beds, 3.5 baths, modern style. Solar panels offset utility cost, concrete counters, native landscaping."
            default_params = "Role: Expert sustainable copywriter.\nTone: Eco-conscious, architectural.\nConstraints: Limit to 3 short paragraphs. Include utility offset bullet points."
        else:
            default_context = "Agent Kyle requests 5% bump in split override for Valley View transaction ($1.8M sales price) claiming self-sourced buyer and spent $2k on custom brochures."
            default_params = "Role: Brokerage Sales Manager.\nTone: Firm but empathetic.\nConstraints: Decline request. Remind of Section 4.2 Uniform Split policy. Detail Tier Volume pathway."
            
        context_input = st.text_area("Context / Raw details:", value=default_context)
        params_input = st.text_area("Parameters (Tone, Length, Constraints):", value=default_params)
        
        if st.button("Compile ACP Prompt"):
            compiled_prompt = f"""[ROLE & PERSONA]
You are a Real Estate Strategy Expert. Adopt a professional, direct, and outcome-oriented communication style.

[TASK ACTION]
Generate a response appropriate for the following scenario: {scenario}.

[CONTEXT & RAW DATA]
{context_input}

[CONSTRAINTS & PARAMETERS]
{params_input}

[EXECUTION RULESET]
1. Write the final response draft directly.
2. Ensure you strictly adhere to the negative constraints (no buzzwords, max length, tone).
3. Do not include introductory text like "Sure, here is the email:" or placeholder symbols."""
            
            st.code(compiled_prompt, language="markdown")
            st.success("ACP Prompt Compiled! Copy the block above into ChatGPT, Gemini, or Claude.")
            
    with tool_col2:
        st.markdown("### 5W1H Reporting Tool (Who, What, Where, When, Why, How)")
        st.write("Summarize noisy meeting transcripts or CRM logs into high-density actionable checklists:")
        
        report_src = st.selectbox("Report Source Data:", ["Weekly Pipeline Review Sync (Transcript)", "Active Escrow Pipeline (CRM Export)"])
        
        raw_text = ""
        if report_src == "Weekly Pipeline Review Sync (Transcript)":
            raw_text = """Manager: Let's do a fast alignment. Dave, what is the status of the listing at 412 Hillside?
Dave: The seller is willing to drop the price by $15,000 if we don't get offers by Wednesday night. I'm hosting an broker open house on Tuesday morning between 10 AM and 12 PM to drum up agent interest.
Manager: Okay, write up the MLS adjustment draft today so we can push it live Wednesday at 9 AM if needed. Sarah, what about escrow on the 102 Pine St deal?
Sarah: The buyers are objecting to the roof inspection report. They want a $5,000 credit or they threat to terminate. We have until Friday at 5 PM to submit our response amendment to Escrow.
Manager: Write a counter-amendment offering a $2,500 credit and coordinate with listing contractors for an independent quote by Thursday morning."""
        else:
            raw_text = """- RECORD ID: #ESC-9011 | Agent: Dave | Listing: 233 Broad St | Status: Awaiting Loan Commitment | Deadline: June 25, 5:00 PM | Action: Upload buyer pre-approval verification document.
- RECORD ID: #ESC-8922 | Agent: Sarah | Listing: 1045 River Rd | Phase: Attorney Review | Status: Sewer line easement dispute | Deadline: June 29, 12:00 PM | Action: Coordinate escrow disclosure review with legal team.
- RECORD ID: #ESC-9055 | Agent: Marcus | Listing: 890 Ridge Way | Status: Lead Response Lagging | Total Leads: 12 | Call rate: 25% | Action: Trigger auto-reminder templates to Marcus."""

        st.text_area("Raw Text Preview:", value=raw_text, height=120, disabled=True)
        
        if st.button("Generate 5W1H Structured Digest"):
            if report_src == "Weekly Pipeline Review Sync (Transcript)":
                st.markdown("""
                #### 5W1H Structured Digest
                
                - **Who (Stakeholders):** Manager, Dave (Listing Agent), Sarah (Transaction Agent).
                - **What (Issues & Actions):** 412 Hillside price reduction preparation. 102 Pine St inspections roofing dispute ($5k buyer credit demand vs $2.5k target offer).
                - **Where (Locations):** 412 Hillside, 102 Pine St.
                - **When (Deadlines):** Open House: Tuesday 10-12 PM. Price reduction MLS live: Wednesday 9:00 AM. Contractor quote: Thursday Morning. Escrow Response Amendment: Friday 5:00 PM.
                - **Why (Rationales):** Drive agent interest to avoid listings staleness. Keep Pine St escrow from contract termination.
                - **How (Methodologies):** Draft price drop on MLS; construct counter-amendment offering $2,500 credit rather than $5,000.
                """)
            else:
                st.markdown("""
                #### 5W1H Structured Digest
                
                - **Who (Stakeholders):** Dave, Sarah, Marcus (Owner Agents).
                - **What (Issues & Actions):** ESC-9011: Loan pre-approval document upload. ESC-8922: Sewer line easement review. ESC-9055: Trigger agent reminders for lagging follow-up.
                - **Where (Locations):** 233 Broad St, 1045 River Rd, 890 Ridge Way.
                - **When (Deadlines):** ESC-9011: June 25, 5:00 PM. ESC-8922: June 29, 12:00 PM. ESC-9055: Immediate.
                - **Why (Rationales):** Avoid contract defaults in escrow. Mitigate sewer easement liability. Arrest lead conversion rate decay.
                - **How (Methodologies):** Upload loan verification docs, route legal boundary files to counsel, fire auto-email triggers to agent.
                """)
