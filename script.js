const API_BASE_URL = "http://127.0.0.1:8000";

// --- DOM ELEMENTS SELECTION ---
// Onboarding Elements
const onboardingOverlay = document.getElementById("onboarding-overlay");
const onboardingForm = document.getElementById("onboarding-form");
const userNameInput = document.getElementById("user-name");
const workspaceEnvSelect = document.getElementById("workspace-env");
const userExpSelect = document.getElementById("user-exp");
const targetSavingsInput = document.getElementById("target-savings");
const btnResetProfile = document.getElementById("btn-reset-profile");

// Sidebar & Profile Elements
const profileName = document.getElementById("profile-name");
const profileRole = document.getElementById("profile-role");
const userAvatar = document.getElementById("user-avatar");
const brandBadgeText = document.getElementById("brand-badge-text");

// Tab Navigation Elements
const navItems = document.querySelectorAll(".nav-item");
const tabPanes = document.querySelectorAll(".tab-pane");
const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");
const headerDate = document.getElementById("header-date");

// Dashboard Elements
const wasteHoursText = document.getElementById("metric-waste-hours");
const wasteSubText = document.getElementById("metric-waste-sub");
const dashboardFocusGoal = document.getElementById("dashboard-focus-goal");
const dashboardFocusSub = document.getElementById("dashboard-focus-sub");
const dashboardTargetSavings = document.getElementById("dashboard-target-savings");
const dashboardTargetSub = document.getElementById("dashboard-target-sub");

const leakToggles = document.querySelectorAll(".leak-toggle");
const btnDeployAll = document.getElementById("btn-deploy-all");
const sidebarReclaimedHours = document.getElementById("sidebar-reclaimed-hours");
const sidebarProgressBar = document.getElementById("sidebar-progress-bar");

const guardrailModePolicy = document.getElementById("guardrail-mode-policy");
const guardrailModeTester = document.getElementById("guardrail-mode-tester");
const policyView = document.getElementById("policy-view");
const testerView = document.getElementById("tester-view");
const scenarioInput = document.getElementById("scenario-input");
const btnEvaluate = document.getElementById("btn-evaluate");
const testerResult = document.getElementById("tester-result");
const testerResultCard = document.getElementById("tester-result-card");
const testerResultBadge = document.getElementById("tester-result-badge");
const testerResultTitle = document.getElementById("tester-result-title");
const testerResultDesc = document.getElementById("tester-result-desc");
const scenarioChips = document.querySelectorAll(".scenario-chip");

// Pipeline & Deal Wizard Elements
const openDealWizardBtn = document.getElementById("openDealWizardBtn");
const dealWizard = document.getElementById("dealWizard");
const wizardStep = document.getElementById("wizardStep");
const wizardQuestion = document.getElementById("wizardQuestion");
const wizardFieldContainer = document.getElementById("wizardFieldContainer");
const wizardCancelBtn = document.getElementById("wizardCancelBtn");
const wizardBackBtn = document.getElementById("wizardBackBtn");
const wizardNextBtn = document.getElementById("wizardNextBtn");

// RAG Chat Elements
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const uploadForm = document.getElementById("uploadForm");
const ragFilesInput = document.getElementById("ragFiles");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const kbFilesList = document.getElementById("kbFilesList");
const refreshFilesBtn = document.getElementById("refreshFilesBtn");

// SOP Knowledge Base Elements
const sopListItems = document.querySelectorAll(".sop-list-item");
const sopTitleEl = document.getElementById("sop-title");
const sopRevisionEl = document.getElementById("sop-revision");
const sopContentBody = document.getElementById("sop-content-body");
const sopSearchInput = document.getElementById("sop-sidebar-search");

// NotebookLM Simulator Elements
const nbChatArea = document.getElementById("nb-chat-area");
const nbQueryInput = document.getElementById("notebook-query-input");
const nbSubmitBtn = document.getElementById("btn-submit-notebook-query");
const nbSuggestBtns = document.querySelectorAll(".nb-suggest-btn");

// Strategy Tools Elements
const commScenarioSelect = document.getElementById("comm-scenario");
const commContextInput = document.getElementById("comm-context");
const commParametersInput = document.getElementById("comm-parameters");
const generatePromptBtn = document.getElementById("btn-generate-prompt");
const promptOutputText = document.getElementById("prompt-output-text");
const copyPromptBtn = document.getElementById("btn-copy-prompt");

const sourceSelect = document.getElementById("reporting-source-select");
const rawContentEl = document.getElementById("reporting-raw-content");
const generateSummaryBtn = document.getElementById("btn-generate-summary");
const summaryResultContainer = document.getElementById("summary-result-container");
const summaryResultText = document.getElementById("summary-result-text");
const copySummaryBtn = document.getElementById("btn-copy-summary");

const kpiCheckboxes = document.querySelectorAll(".kpi-checkbox");
const kpiPercentageText = document.getElementById("kpi-percentage");

// Clipboard Toast
const toast = document.getElementById("toast");


// --- ONBOARDING & PROFILE STATE MANAGEMENT ---
let userProfile = {
  name: "",
  focus: "",
  experience: "",
  targetSavings: 16
};

function initProfile() {
  const savedProfile = localStorage.getItem("realtyai_profile");
  if (savedProfile) {
    userProfile = JSON.parse(savedProfile);
    applyProfile();
  } else {
    onboardingOverlay.classList.remove("hidden");
  }
}

function saveProfile(name, focus, experience, targetSavings) {
  userProfile = { name, focus, experience, targetSavings: parseFloat(targetSavings) || 16 };
  localStorage.setItem("realtyai_profile", JSON.stringify(userProfile));
  applyProfile();
  onboardingOverlay.classList.add("hidden");
  showToast("Workspace initialized!");
}

function applyProfile() {
  // Update avatar initials
  if (userProfile.name) {
    const initials = userProfile.name.split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase();
    userAvatar.textContent = initials || "SM";
  }
  
  // Update sidebar footer
  profileName.textContent = userProfile.name || "Sales Manager";
  profileRole.textContent = userProfile.focus || "Workspace Owner";
  brandBadgeText.textContent = userProfile.experience || "Playbook Engine";
  
  // Update dashboard greetings & header titles
  dashboardFocusGoal.textContent = userProfile.focus === "Commercial Real Estate (CRE)" ? "Broker Deals" : (userProfile.focus === "Property Developer" ? "Sell Inventory" : "Maximize GCI");
  dashboardFocusSub.textContent = userProfile.focus === "Commercial Real Estate (CRE)" ? "Optimize Lease Volume" : (userProfile.focus === "Property Developer" ? "Onboard Off-Plan Units" : "Reduce Days on Market");
  
  dashboardTargetSavings.textContent = `Reclaim ${Math.round((userProfile.targetSavings / 40) * 100)}%`;
  dashboardTargetSub.textContent = `~${userProfile.targetSavings} Hours of Work Week`;
  
  updateReclaimedTime();
}

onboardingForm.addEventListener("submit", (e) => {
  e.preventDefault();
  saveProfile(
    userNameInput.value.trim(),
    workspaceEnvSelect.value,
    userExpSelect.value,
    targetSavingsInput.value
  );
});

btnResetProfile.addEventListener("click", () => {
  localStorage.removeItem("realtyai_profile");
  userNameInput.value = userProfile.name;
  workspaceEnvSelect.value = userProfile.focus || "Residential Brokerage";
  userExpSelect.value = userProfile.experience || "Mid Operator";
  targetSavingsInput.value = userProfile.targetSavings || 16;
  onboardingOverlay.classList.remove("hidden");
});


// --- TAB ROUTING SYSTEM ---
const tabMeta = {
  dashboard: {
    title: "Dashboard Overview",
    subtitle: "Track your current condition, time waste metrics, and active AI guardrails."
  },
  pipeline: {
    title: "Deal & Listing Pipeline",
    subtitle: "Track buyers, sellers, and active listings using the team Kanban workflow."
  },
  assistant: {
    title: "AI Coach Assistant",
    subtitle: "Query the RAG knowledge base for personalized coaching based on SOPs."
  },
  knowledge: {
    title: "SOP Knowledge Base",
    subtitle: "Review team operational procedures and simulate NotebookLM queries."
  },
  tools: {
    title: "Strategy & Prompting Tools",
    subtitle: "Access the ACP prompt compiler and compile 5W1H meeting summaries."
  }
};

navItems.forEach(item => {
  item.addEventListener("click", () => {
    const targetTab = item.getAttribute("data-tab");
    
    navItems.forEach(nav => nav.classList.remove("active"));
    item.classList.add("active");
    
    tabPanes.forEach(pane => pane.classList.remove("active"));
    const targetPane = document.getElementById(`tab-${targetTab}`);
    if (targetPane) targetPane.classList.add("active");
    
    if (tabMeta[targetTab]) {
      pageTitle.textContent = tabMeta[targetTab].title;
      pageSubtitle.textContent = tabMeta[targetTab].subtitle;
    }
  });
});

// Set date in header
const now = new Date();
const dateOptions = { year: "numeric", month: "long", day: "numeric" };
headerDate.textContent = now.toLocaleDateString("en-US", dateOptions);


// --- TAB 1: DASHBOARD TIME LEAK CALCULATOR ---
const originalWasteMin = 12;
const originalWasteMax = 18;

function updateReclaimedTime() {
  let totalReclaimed = 0.0;
  
  leakToggles.forEach(toggle => {
    const row = toggle.closest("tr");
    if (toggle.checked) {
      totalReclaimed += parseFloat(toggle.getAttribute("data-hours"));
      row.classList.add("active-experiment");
    } else {
      row.classList.remove("active-experiment");
    }
  });

  const targetHours = userProfile.targetSavings || 16.0;
  sidebarReclaimedHours.textContent = `${totalReclaimed.toFixed(1)}h / ${targetHours.toFixed(0)}h`;
  const percent = Math.min((totalReclaimed / targetHours) * 100, 100);
  sidebarProgressBar.style.width = `${percent}%`;

  if (totalReclaimed === 0) {
    wasteHoursText.textContent = "12-18 Hours";
    wasteHoursText.className = "metric-value text-red";
    wasteSubText.textContent = "Leaking Current Week";
    wasteSubText.className = "metric-sub text-red";
  } else {
    const currentMinLeft = Math.max(originalWasteMin - totalReclaimed, 0);
    const currentMaxLeft = Math.max(originalWasteMax - totalReclaimed, 0);
    
    if (currentMaxLeft === 0) {
      wasteHoursText.textContent = "0 Hours Leaking!";
      wasteHoursText.className = "metric-value text-emerald";
      wasteSubText.textContent = "Operational Buffer Fully Reclaimed";
      wasteSubText.className = "metric-sub text-emerald";
      showToast("Maximum Time Reclaimed! Buffer secured.");
    } else {
      wasteHoursText.textContent = `${currentMinLeft.toFixed(1)}-${currentMaxLeft.toFixed(1)} Hours`;
      wasteHoursText.className = "metric-value text-amber";
      wasteSubText.textContent = `${totalReclaimed.toFixed(1)}h Reclaimed Active`;
      wasteSubText.className = "metric-sub text-amber";
    }
  }
}

leakToggles.forEach(toggle => {
  toggle.addEventListener("change", updateReclaimedTime);
});

btnDeployAll.addEventListener("click", () => {
  const allChecked = Array.from(leakToggles).every(t => t.checked);
  leakToggles.forEach(toggle => {
    toggle.checked = !allChecked;
  });
  btnDeployAll.textContent = !allChecked ? "Reset All" : "Deploy All";
  updateReclaimedTime();
});


// --- TAB 1: AI GOVERNANCE GUARDRAILS ---
guardrailModePolicy.addEventListener("click", () => {
  guardrailModePolicy.classList.add("active");
  guardrailModeTester.classList.remove("active");
  policyView.classList.remove("d-none");
  testerView.classList.add("d-none");
});

guardrailModeTester.addEventListener("click", () => {
  guardrailModeTester.classList.add("active");
  guardrailModePolicy.classList.remove("active");
  testerView.classList.remove("d-none");
  policyView.classList.add("d-none");
});

scenarioChips.forEach(chip => {
  chip.addEventListener("click", () => {
    const text = chip.getAttribute("data-text");
    scenarioInput.value = text;
    evaluateScenario(text);
  });
});

btnEvaluate.addEventListener("click", () => {
  const text = scenarioInput.value.trim();
  if (text) evaluateScenario(text);
});

scenarioInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    const text = scenarioInput.value.trim();
    if (text) evaluateScenario(text);
  }
});

function evaluateScenario(text) {
  const lower = text.toLowerCase();
  
  const redKeywords = [
    "terminate", "fire", "promote", "hire", "negotiate", "negotiation", 
    "dispute", "lawsuit", "contract review", "sign contract", "commission dispute",
    "firing", "firing agent", "court", "legal compliance", "close deal"
  ];

  const greenKeywords = [
    "draft", "summary", "transcript", "listing", "email", "crm", 
    "metric", "description", "write follow-up", "follow-up", "follow up", 
    "outlines", "agenda", "copywriting", "post", "social media"
  ];

  testerResult.classList.remove("d-none");
  
  let suitability = "yellow";
  for (const kw of redKeywords) {
    if (lower.includes(kw)) {
      suitability = "red";
      break;
    }
  }

  if (suitability !== "red") {
    for (const kw of greenKeywords) {
      if (lower.includes(kw)) {
        suitability = "green";
        break;
      }
    }
  }

  if (suitability === "green") {
    testerResultCard.className = "result-status-card green-light";
    testerResultBadge.textContent = "GREEN LIGHT";
    testerResultTitle.textContent = "AI Approved Task";
    testerResultDesc.textContent = "This task is administrative, informational, or drafts low-stakes copy. You can deploy AI tools safely to expedite this task (e.g. drafting emails, transcripts).";
  } else if (suitability === "red") {
    testerResultCard.className = "result-status-card red-light";
    testerResultBadge.textContent = "RED LIGHT";
    testerResultTitle.textContent = "Human-Only Mandated";
    testerResultDesc.textContent = "CRITICAL: This activity involves legally binding negotiations, commission arbitration, or sensitive HR/personnel management. AI should not automate this task; a human manager is strictly required.";
  } else {
    testerResultCard.className = "result-status-card red-light";
    testerResultCard.style.borderColor = "#fbbf24";
    testerResultCard.style.backgroundColor = "rgba(251, 191, 36, 0.05)";
    testerResultBadge.textContent = "YELLOW LIGHT";
    testerResultBadge.style.backgroundColor = "#fbbf24";
    testerResultBadge.style.color = "#04060b";
    testerResultTitle.textContent = "Conditional Policy Check";
    testerResultDesc.textContent = "Unable to verify task safety parameters automatically. Ensure client PII is omitted and check results thoroughly before distributing communications.";
  }
  
  testerResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
}


// --- TAB 2: PIPELINE BOARD (KANBAN & WIZARD) ---
const WIZARD_STEPS = [
  {
    key: "accountName",
    question: "Who is the client or buyer name?",
    type: "text",
    placeholder: "Example: Sarah Connor",
  },
  {
    key: "propertyAddress",
    question: "What is the property listing address?",
    type: "text",
    placeholder: "Example: 1045 River Rd",
  },
  {
    key: "dealValue",
    question: "What is the listing price or GCI?",
    type: "text",
    placeholder: "Example: $450,000 / $18k GCI",
  },
  {
    key: "stage",
    question: "Which pipeline stage is the deal in right now?",
    type: "select",
    options: ["Lead", "Discovery", "Proposal", "Closed Won"],
  },
  {
    key: "context",
    question: "What is the current context we should capture on the tile?",
    type: "textarea",
    placeholder: "Example: Buyer viewed property, concerns about sewer line easement.",
  },
  {
    key: "nextStep",
    question: "What is the next logical action for this deal?",
    type: "text",
    placeholder: "Example: Coordinate disclosures review with legal team.",
  },
];

let wizardState = {
  currentStep: 0,
  answers: {},
};

function escapeCssIdentifier(value) {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}

function buildWizardField(step) {
  let field;
  if (step.type === "textarea") {
    field = document.createElement("textarea");
    field.className = "wizard-textarea";
    field.rows = 4;
  } else if (step.type === "select") {
    field = document.createElement("select");
    field.className = "wizard-select";
    (step.options || []).forEach(optionText => {
      const option = document.createElement("option");
      option.value = optionText;
      option.textContent = optionText;
      field.appendChild(option);
    });
  } else {
    field = document.createElement("input");
    field.className = "wizard-input";
    field.type = "text";
  }

  field.id = "wizardInput";
  if (step.placeholder) {
    field.placeholder = step.placeholder;
  }

  const savedValue = wizardState.answers[step.key];
  if (typeof savedValue === "string") {
    field.value = savedValue;
  }

  return field;
}

function getCurrentWizardInput() {
  return document.getElementById("wizardInput");
}

function showWizardStep() {
  if (!dealWizard || !wizardStep || !wizardQuestion || !wizardFieldContainer || !wizardBackBtn || !wizardNextBtn) return;

  const stepIndex = wizardState.currentStep;
  const step = WIZARD_STEPS[stepIndex];

  wizardStep.textContent = `Question ${stepIndex + 1} of ${WIZARD_STEPS.length}`;
  wizardQuestion.textContent = step.question;

  wizardFieldContainer.innerHTML = "";
  const inputElement = buildWizardField(step);
  wizardFieldContainer.appendChild(inputElement);

  wizardBackBtn.disabled = stepIndex === 0;
  wizardNextBtn.textContent = stepIndex === WIZARD_STEPS.length - 1 ? "Create Tile" : "Next";
  inputElement.focus();
}

function openWizard() {
  if (!dealWizard) return;
  wizardState = { currentStep: 0, answers: {} };
  dealWizard.classList.remove("hidden");
  dealWizard.setAttribute("aria-hidden", "false");
  showWizardStep();
}

function closeWizard() {
  if (!dealWizard) return;
  dealWizard.classList.add("hidden");
  dealWizard.setAttribute("aria-hidden", "true");
}

function captureCurrentStepAnswer() {
  const currentStep = WIZARD_STEPS[wizardState.currentStep];
  const inputElement = getCurrentWizardInput();

  if (!currentStep || !inputElement) return false;

  const value = (inputElement.value || "").trim();
  if (!value) {
    inputElement.focus();
    showToast(`Please fill out: ${currentStep.question}`);
    return false;
  }

  wizardState.answers[currentStep.key] = value;
  return true;
}

function createDealTileFromWizard() {
  const { accountName, propertyAddress, dealValue, stage, context, nextStep } = wizardState.answers;
  
  const stageSelector = `[data-stage="${escapeCssIdentifier(stage)}"] .kanban-cards`;
  const stageColumn = document.querySelector(stageSelector);
  
  if (!stageColumn) {
    showToast(`Could not find a matching stage for "${stage}".`);
    return;
  }

  const card = document.createElement("div");
  card.className = "deal-card";

  const title = document.createElement("h4");
  title.textContent = accountName;

  const description = document.createElement("p");
  description.textContent = `${propertyAddress} - ${context}`;

  const metaRow = document.createElement("div");
  metaRow.className = "deal-card-meta";

  const valEl = document.createElement("span");
  valEl.className = "deal-value";
  valEl.textContent = dealValue;

  const nextEl = document.createElement("span");
  nextEl.className = "deal-next-action";
  nextEl.textContent = nextStep;

  metaRow.appendChild(valEl);
  metaRow.appendChild(nextEl);

  card.appendChild(title);
  card.appendChild(description);
  card.appendChild(metaRow);

  stageColumn.prepend(card);
  refreshColumnCounts();
  showToast(`Added new card for ${accountName} in ${stage}!`);
}

function handleWizardNext() {
  if (!captureCurrentStepAnswer()) return;

  const isLastStep = wizardState.currentStep === WIZARD_STEPS.length - 1;
  if (isLastStep) {
    createDealTileFromWizard();
    closeWizard();
    return;
  }

  wizardState.currentStep += 1;
  showWizardStep();
}

function handleWizardBack() {
  const inputElement = getCurrentWizardInput();
  const currentStep = WIZARD_STEPS[wizardState.currentStep];
  if (inputElement && currentStep) {
    wizardState.answers[currentStep.key] = (inputElement.value || "").trim();
  }

  if (wizardState.currentStep === 0) return;

  wizardState.currentStep -= 1;
  showWizardStep();
}

function refreshColumnCounts() {
  document.querySelectorAll(".kanban-column").forEach(column => {
    const countEl = column.querySelector("[data-count]");
    const cardCount = column.querySelectorAll(".deal-card").length;
    if (countEl) countEl.textContent = String(cardCount);
  });
}

function initKanban() {
  document.querySelectorAll(".kanban-column").forEach(column => {
    const stageName = column.querySelector("header h3")?.textContent?.trim();
    if (stageName) {
      column.dataset.stage = stageName;
    }
  });

  if (typeof Sortable === "undefined") return;

  document.querySelectorAll(".kanban-cards").forEach(column => {
    new Sortable(column, {
      group: "pipeline",
      animation: 200,
      ghostClass: "sortable-ghost",
      chosenClass: "sortable-chosen",
      onAdd: refreshColumnCounts,
      onRemove: refreshColumnCounts,
      onSort: refreshColumnCounts,
    });
  });
}

if (openDealWizardBtn) openDealWizardBtn.addEventListener("click", openWizard);
if (wizardCancelBtn) wizardCancelBtn.addEventListener("click", closeWizard);
if (wizardBackBtn) wizardBackBtn.addEventListener("click", handleWizardBack);
if (wizardNextBtn) wizardNextBtn.addEventListener("click", handleWizardNext);


// --- TAB 3: AI PLAYBOOK ASSISTANT (RAG COMMUNICATIONS) ---
function addChatMessage(text, type = "ai", source = null) {
  if (!chatMessages) return;

  const bubble = document.createElement("div");
  bubble.className = `message ${type === "user" ? "user-message" : "ai-message"}`;

  const content = document.createElement("p");
  content.textContent = text;
  bubble.appendChild(content);

  if (source) {
    const sourceLine = document.createElement("div");
    sourceLine.className = "source-line";
    sourceLine.textContent = `Source: ${source}`;
    bubble.appendChild(sourceLine);
  }

  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createThinkingMessage() {
  if (!chatMessages) return null;
  const bubble = document.createElement("div");
  bubble.className = "message ai-message thinking";
  bubble.textContent = "Thinking";
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

async function sendChatMessage(question) {
  const thinkingBubble = createThinkingMessage();
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        role: userProfile.focus || "Sales Manager",
        experience: userProfile.experience || "Mid Operator"
      }),
    });

    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    
    const data = await response.json();
    if (thinkingBubble) thinkingBubble.remove();

    const answerText = typeof data.answer === "string" ? data.answer.trim() : "";
    const sourceText = Array.isArray(data.source) ? data.source.join(", ") : data.source || "No source provided";
    addChatMessage(answerText || "No answer generated.", "ai", sourceText);
  } catch (error) {
    if (thinkingBubble) thinkingBubble.remove();
    addChatMessage(`I ran into an issue contacting the server: ${error.message}. Make sure FastAPI is running.`, "ai");
  }
}

async function uploadRagFiles(files) {
  const formData = new FormData();
  files.forEach(file => {
    formData.append("files", file);
  });

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorText = `Upload failed (${response.status})`;
    try {
      const errorPayload = await response.json();
      if (errorPayload?.detail) {
        errorText = Array.isArray(errorPayload.detail) ? errorPayload.detail.join(", ") : String(errorPayload.detail);
      }
    } catch (_e) {}
    throw new Error(errorText);
  }
  return response.json();
}

function renderKnowledgeFiles(files) {
  if (!kbFilesList) return;
  kbFilesList.innerHTML = "";

  if (!files.length) {
    const emptyItem = document.createElement("li");
    emptyItem.className = "kb-file-item kb-file-empty";
    emptyItem.textContent = "No files added yet.";
    kbFilesList.appendChild(emptyItem);
    return;
  }

  files.forEach(fileName => {
    const fileItem = document.createElement("li");
    fileItem.className = "kb-file-item";
    
    const nameSpan = document.createElement("span");
    nameSpan.textContent = fileName;
    fileItem.appendChild(nameSpan);
    
    kbFilesList.appendChild(fileItem);
  });
}

async function loadKnowledgeFiles() {
  if (!kbFilesList) return;
  try {
    const response = await fetch(`${API_BASE_URL}/knowledge-files`);
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const payload = await response.json();
    const files = Array.isArray(payload.files) ? payload.files : [];
    renderKnowledgeFiles(files);
  } catch (error) {
    renderKnowledgeFiles([]);
    const errorItem = document.createElement("li");
    errorItem.className = "kb-file-item kb-file-empty";
    errorItem.textContent = `Could not load files: ${error.message}`;
    kbFilesList.innerHTML = "";
    kbFilesList.appendChild(errorItem);
  }
}

if (chatForm) {
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;

    addChatMessage(question, "user");
    chatInput.value = "";
    sendBtn.disabled = true;
    await sendChatMessage(question);
    sendBtn.disabled = false;
    chatInput.focus();
  });
}

if (uploadForm) {
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const selectedFiles = Array.from(ragFilesInput.files || []);
    if (!selectedFiles.length) {
      uploadStatus.textContent = "Select one or more files first.";
      return;
    }

    uploadBtn.disabled = true;
    uploadStatus.textContent = "Uploading and indexing files...";

    try {
      const payload = await uploadRagFiles(selectedFiles);
      const fileCount = Array.isArray(payload.files_saved) ? payload.files_saved.length : 0;
      uploadStatus.textContent = `Uploaded ${fileCount} file(s). RAG index refreshed.`;
      ragFilesInput.value = "";
      await loadKnowledgeFiles();
      addChatMessage(`Knowledge base updated with ${fileCount} file(s). You can now ask questions based on the new content.`, "ai");
    } catch (error) {
      uploadStatus.textContent = `Upload error: ${error.message}`;
    } finally {
      uploadBtn.disabled = false;
    }
  });
}

if (refreshFilesBtn) refreshFilesBtn.addEventListener("click", loadKnowledgeFiles);


// --- TAB 4: SOP KNOWLEDGE BASE ---
const sopDocuments = {
  "lead-routing": {
    title: "SOP: Lead Handling & Routing",
    revision: "Revision 4.1",
    html: `
      <h5>1. Lead Response Time SLA</h5>
      <p>All inbound web inquiries (Zillow, Realtor.com, personal website landing pages) must be contacted within <strong>15 minutes</strong> of registration during normal business hours (8:00 AM - 7:00 PM). P1-level "Hot Leads" must be called immediately.</p>
      
      <h5>2. CRM Logging Protocol</h5>
      <p>Every lead contact attempt must be logged in the database CRM under the following guidelines:</p>
      <ul>
        <li>Record date and time of attempt.</li>
        <li>Log outcome code (e.g. NA for No Answer, VM for Voicemail, CC for Connected).</li>
        <li>Schedule follow-up task sequence before closing contact pane.</li>
      </ul>

      <h5>3. Active Lead Routing Hierarchy</h5>
      <p>If an assigned agent fails to contact a client within 2 hours, the lead is automatically reassigned to the secondary on-call listing agent to prevent response time degradation. Dispute claims regarding lead stealing will be routed to the Regional Sales Manager.</p>
    `
  },
  "listing-launch": {
    title: "SOP: Listing Launch Checklist",
    revision: "Revision 2.0",
    html: `
      <h5>1. Marketing Preparations (Day -7 to Day -1)</h5>
      <p>Standard listing onboarding workflow checklist:</p>
      <ol>
        <li>Schedule HDR photography and 3D walkthrough scanning (must complete by Day -4).</li>
        <li>Acquire and verify listing title deed and survey documents.</li>
        <li>Draft MLS property description using standard approved formatting rules.</li>
      </ol>

      <h5>2. Go-Live Procedures (Day 0)</h5>
      <p>Listing goes live on MLS by 9:00 AM EST on Thursday. Submit syndication feeds to Zillow, Trulia, and Realtor.com. Install yard signs, lockbox, and flyers cabinet at the property.</p>

      <h5>3. Post-Launch Actions (Day +1 to Day +3)</h5>
      <p>Schedule public Open House for Saturday and Sunday (1:00 PM - 4:00 PM). Launch targeted social media marketing campaigns (minimum budget $100 per listing).</p>
    `
  },
  "dispute-res": {
    title: "SOP: Commission Dispute Resolution",
    revision: "Revision 1.2",
    html: `
      <h5>1. Definition of Procuring Cause</h5>
      <p>Disputes regarding split commissions are governed by the principle of "Procuring Cause"—the agent who initiated the chain of events that directly led to the final contract signing is entitled to the primary split.</p>

      <h5>2. Informal Review Process</h5>
      <p>Prior to submitting a formal broker dispute, agents must attempt an informal mutual review with the sales manager. The manager acts as a neutral negotiator to establish splitting terms (e.g. 50/50, 75/25 split arrangements).</p>

      <h5>3. Formal Arbitration Workflow</h5>
      <p>If informal review fails, agents must submit a formal written request for arbitration within 10 business days of contract closing. The executive review board will render a final, non-appealable decision within 5 business days of submission.</p>
    `
  }
};

function renderActiveSop(key) {
  const doc = sopDocuments[key];
  if (doc) {
    sopTitleEl.textContent = doc.title;
    sopRevisionEl.textContent = doc.revision;
    sopContentBody.innerHTML = doc.html;
  }
}

sopListItems.forEach(item => {
  item.addEventListener("click", () => {
    sopListItems.forEach(i => i.classList.remove("active"));
    item.classList.add("active");
    renderActiveSop(item.getAttribute("data-sop"));
  });
});

if (sopSearchInput) {
  sopSearchInput.addEventListener("input", () => {
    const query = sopSearchInput.value.toLowerCase().trim();
    sopListItems.forEach(item => {
      const title = item.querySelector(".doc-title").textContent.toLowerCase();
      item.style.display = title.includes(query) ? "flex" : "none";
    });
  });
}


// --- TAB 4: NOTEBOOKLM SIMULATOR ---
const simulatedNotebookResponses = {
  "response-time": {
    answer: "According to **SOP: Lead Handling & Routing (Section 1)**, all inbound web inquiries must be contacted within **15 minutes** of registration during business hours (8:00 AM - 7:00 PM). Hot Leads require immediate phone call response.",
    citations: ["SOP: Lead Handling & Routing §1"]
  },
  "first-24": {
    answer: "Based on **SOP: Listing Launch Checklist (Section 2)**, within the first 24 hours of going live (Day 0), you must: 1. Confirm syndication to Zillow/Realtor.com. 2. Install physical yard signs and lockboxes. 3. Setup flyer cabinets at the property.",
    citations: ["SOP: Listing Launch Checklist §2"]
  },
  "lead-dispute": {
    answer: "According to **SOP: Commission Dispute Resolution (Section 2)**, if two agents claim the same lead, they must first attempt an informal mutual review with the sales manager to reach an split agreement (e.g., 50/50 split). If unresolved, they must submit a formal written request for arbitration within 10 days of contract closing.",
    citations: ["SOP: Commission Dispute Resolution §2", "SOP: Lead Handling & Routing §3"]
  },
  "fallback": {
    answer: "I couldn't find a direct answer to that question in the loaded SOP documents. Please verify your SOP files or contact the brokerage compliance officer.",
    citations: []
  }
};

function addNotebookChatMessage(text, isUser = false, citations = []) {
  const msg = document.createElement("div");
  msg.className = `chat-message ${isUser ? "user-message" : "bot-message"}`;
  
  let html = `<p>${text}</p>`;
  if (!isUser && citations.length > 0) {
    html += `<div class="bot-message-citations">`;
    citations.forEach(c => {
      html += `<span class="citation-pill">${c}</span>`;
    });
    html += `</div>`;
  }
  
  msg.innerHTML = html;
  nbChatArea.appendChild(msg);
  nbChatArea.scrollTop = nbChatArea.scrollHeight;
}

function handleNotebookQuery(query) {
  if (query.trim() === "") return;
  
  addNotebookChatMessage(query, true);
  nbQueryInput.value = "";

  const typingIndicator = document.createElement("div");
  typingIndicator.className = "chat-message bot-message typing-indicator";
  typingIndicator.innerHTML = `
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  `;
  nbChatArea.appendChild(typingIndicator);
  nbChatArea.scrollTop = nbChatArea.scrollHeight;

  setTimeout(() => {
    typingIndicator.remove();
    const lower = query.toLowerCase();
    let responseKey = "fallback";

    if (lower.includes("response time") || lower.includes("limit") || lower.includes("hot web lead") || lower.includes("how quickly")) {
      responseKey = "response-time";
    } else if (lower.includes("24 hours") || lower.includes("listing launch") || lower.includes("marketing steps") || lower.includes("first 24")) {
      responseKey = "first-24";
    } else if (lower.includes("dispute") || lower.includes("claim the same") || lower.includes("resolve") || lower.includes("procuring")) {
      responseKey = "lead-dispute";
    }

    const res = simulatedNotebookResponses[responseKey];
    addNotebookChatMessage(res.answer, false, res.citations);
  }, 1000);
}

if (nbSubmitBtn) {
  nbSubmitBtn.addEventListener("click", () => handleNotebookQuery(nbQueryInput.value));
}
if (nbQueryInput) {
  nbQueryInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleNotebookQuery(nbQueryInput.value);
  });
}
nbSuggestBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    handleNotebookQuery(btn.getAttribute("data-query"));
  });
});


// --- TAB 5: STRATEGY & PROMPTING TOOLS ---
const commTemplates = {
  "silent-client": {
    context: "Buyer named Sarah viewed 1045 River Rd 4 days ago. Expressed high interest in the updated chef kitchen and private wooded backyard, but has been silent on text/email since. Needs to follow up and prompt for next steps.",
    parameters: "Role: Senior Real Estate Consultant with 15 years experience.\nTone: Consultative, professional, warm, and low-pressure.\nConstraints: Max 120 words. Absolutely no generic marketing buzzwords like 'nestled', 'boasts', 'stunning', or 'prestigious'. Ask a singular open-ended closing question about disclosures."
  },
  "listing-desc": {
    context: "New Listing: 789 Oakridge Ave. 4 bedrooms, 3.5 bathrooms, modern design. Key Selling Points: double-pane low-E windows, custom solar array offset, open concrete counters in kitchen, native-plant landscaping.",
    parameters: "Role: Expert Real Estate copywriter specializing in sustainable properties.\nTone: Modern, eco-conscious, architecturally appreciative.\nConstraints: Limit to 3 short paragraphs. Include a feature list highlighting utility cost offsets. Do not use filler sentences."
  },
  "comm-inquiry": {
    context: "Agent Kyle requests a 5% bump in commission split override for the transaction at 555 Valley View Rd ($1.8M sales price), claiming he spent twice as much on private media mailers and self-sourced the buyer.",
    parameters: "Role: Regional Brokerage Sales Manager.\nTone: Grounded in team policy, empathetic but firm.\nConstraints: Do not grant split change. Remind agent of Uniform Compensation policy Section 4.2. Outline pathway to next tier of commissions via monthly volume."
  }
};

function loadCommDefaults() {
  const selected = commScenarioSelect.value;
  if (commTemplates[selected]) {
    commContextInput.value = commTemplates[selected].context;
    commParametersInput.value = commTemplates[selected].parameters;
  }
}

commScenarioSelect.addEventListener("change", loadCommDefaults);

generatePromptBtn.addEventListener("click", () => {
  const context = commContextInput.value.trim();
  const parameters = commParametersInput.value.trim();

  if (!context || !parameters) {
    showToast("Please fill in both Context and Parameters.");
    return;
  }

  const selectedScenario = commScenarioSelect.options[commScenarioSelect.selectedIndex].text;
  const formattedPrompt = `[ROLE & PERSONA]
You are a Real Estate Strategy Expert. Adopt a professional, direct, and outcome-oriented communication style.

[TASK ACTION]
Generate a response appropriate for the following scenario: ${selectedScenario}.

[CONTEXT & RAW DATA]
${context}

[CONSTRAINTS & PARAMETERS]
${parameters}

[EXECUTION RULESET]
1. Write the final response draft directly.
2. Ensure you strictly adhere to the negative constraints (no buzzwords, max length, tone).
3. Do not include introductory text like "Sure, here is the email:" or placeholder symbols.`;

  promptOutputText.textContent = formattedPrompt;
  promptOutputText.style.fontFamily = "monospace";
  promptOutputText.style.color = "#38bdf8";
  showToast("Prompt template generated!");
});

copyPromptBtn.addEventListener("click", () => {
  const text = promptOutputText.textContent;
  if (text && !text.startsWith("Select a scenario")) {
    navigator.clipboard.writeText(text).then(() => showToast("Prompt copied to clipboard!"));
  }
});


// 5W1H Reporting Tool logic
const sourceData = {
  transcript: `[Transcript: Monday Morning Sales Pipeline Sync]
Manager: Let's do a fast alignment. Dave, what is the status of the listing at 412 Hillside?
Dave: The seller is willing to drop the price by $15,000 if we don't get offers by Wednesday night. I'm hosting an broker open house on Tuesday morning between 10 AM and 12 PM to drum up agent interest.
Manager: Okay, write up the MLS adjustment draft today so we can push it live Wednesday at 9 AM if needed. Sarah, what about escrow on the 102 Pine St deal?
Sarah: The buyers are objecting to the roof inspection report. They want a $5,000 credit or they threat to terminate. We have until Friday at 5 PM to submit our official response amendment to Escrow.
Manager: Write a counter-amendment offering a $2,500 credit and coordinate with listing contractors for an independent quote by Thursday morning.`,
  
  "crm-data": `[CRM Export - Escrow Pipeline & Agent Touchpoints - June 22]
- RECORD ID: #ESC-9011 | Agent: Dave | Listing: 233 Broad St | Phase: Under Contract | Status: Awaiting Loan Commitment | Deadline: June 25, 5:00 PM | Action: Upload buyer pre-approval verification document.
- RECORD ID: #ESC-8922 | Agent: Sarah | Listing: 1045 River Rd | Phase: Attorney Review | Status: Contract dispute regarding sewer line easement | Deadline: June 29, 12:00 PM | Action: Coordinate escrow disclosure review with legal team.
- RECORD ID: #ESC-9055 | Agent: Marcus | Listing: 890 Ridge Way | Phase: Newly Active | Status: Lead Response Lagging | Total Leads: 12 | Call rate: 25% | Action: Trigger auto-reminder templates to Marcus for follow-up call sequence.`
};

function loadRawSource() {
  const selected = sourceSelect.value;
  if (sourceData[selected]) {
    rawContentEl.textContent = sourceData[selected];
  }
}

sourceSelect.addEventListener("change", loadRawSource);

generateSummaryBtn.addEventListener("click", () => {
  const selected = sourceSelect.value;
  let htmlOutput = "";

  if (selected === "transcript") {
    htmlOutput = `
      <h5>Who (Stakeholders)</h5>
      <ul>
        <li><strong>Manager:</strong> Directing workflow and approving final negotiation credits.</li>
        <li><strong>Dave (Listing Agent):</strong> Managing 412 Hillside listing and Tuesday broker sync.</li>
        <li><strong>Sarah (Transaction Agent):</strong> Handling negotiations for 102 Pine St buyers.</li>
      </ul>

      <h5>What (Actions & Issues)</h5>
      <ul>
        <li>Price reduction preparation ($15k reduction) for 412 Hillside.</li>
        <li>Escrow inspection amendment dispute regarding roof repairs ($5k credit request vs $2.5k target offer).</li>
      </ul>

      <h5>Where (Properties / Locations)</h5>
      <ul>
        <li>412 Hillside (Listing price reduction / Open House).</li>
        <li>102 Pine St (Escrow roof repair dispute).</li>
      </ul>

      <h5>When (Deadlines)</h5>
      <ul>
        <li><strong>Tuesday 10:00 AM - 12:00 PM:</strong> Broker Open House (Hillside).</li>
        <li><strong>Wednesday 9:00 AM:</strong> Price drop MLS adjustment goes live (Hillside).</li>
        <li><strong>Thursday Morning:</strong> Contractor independent repair quote submission (Pine St).</li>
        <li><strong>Friday 5:00 PM:</strong> Legal deadline to submit amendment response (Pine St).</li>
      </ul>

      <h5>Why (Rationales)</h5>
      <ul>
        <li>To drum up agent interest through direct broker walkthrough and secure contract price adjustments prior to listing stale-dates.</li>
        <li>To preserve escrow on Pine St from contract termination threats.</li>
      </ul>

      <h5>How (Methodologies)</h5>
      <ul>
        <li>Draft price reduction on MLS back-end; construct legal counter-amendment offering $2,500 credit rather than $5,000.</li>
      </ul>
    `;
  } else {
    htmlOutput = `
      <h5>Who (Stakeholders)</h5>
      <ul>
        <li><strong>Dave:</strong> Coordinating ESC-9011 buyer loan compliance.</li>
        <li><strong>Sarah:</strong> Managing attorney review and seller documentation on ESC-8922.</li>
        <li><strong>Marcus:</strong> Managing lead responses on active Ridge Way listing.</li>
      </ul>

      <h5>What (Actions & Issues)</h5>
      <ul>
        <li><strong>ESC-9011:</strong> Upload buyer pre-approval verification documentation.</li>
        <li><strong>ESC-8922:</strong> Resolve attorney review Sewer easement dispute.</li>
        <li><strong>ESC-9055:</strong> Lead response lagging alert. 12 active leads but only 25% call contact rate.</li>
      </ul>

      <h5>Where (Properties / Locations)</h5>
      <ul>
        <li>233 Broad St (ESC-9011)</li>
        <li>1045 River Rd (ESC-8922)</li>
        <li>890 Ridge Way (ESC-9055)</li>
      </ul>

      <h5>When (Deadlines)</h5>
      <ul>
        <li><strong>June 25, 5:00 PM:</strong> Loan verification documentation upload deadline.</li>
        <li><strong>June 29, 12:00 PM:</strong> Sewer easement legal resolution deadline.</li>
        <li><strong>Immediate:</strong> Trigger auto-reminder notifications.</li>
      </ul>

      <h5>Why (Rationales)</h5>
      <ul>
        <li>To avoid contract default states in escrow, mitigate legal property line disclosure liabilities, and arrest lead response degradation.</li>
      </ul>

      <h5>How (Methodologies)</h5>
      <ul>
        <li>Deploy automated reminder integrations via CRM email alerts and route escrow reviews to regional legal counsel.</li>
      </ul>
    `;
  }

  summaryResultText.innerHTML = htmlOutput;
  summaryResultContainer.classList.remove("d-none");
  summaryResultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  showToast("5W1H summary created!");
});

copySummaryBtn.addEventListener("click", () => {
  const text = summaryResultText.innerText;
  if (text) {
    navigator.clipboard.writeText(text).then(() => showToast("Summary copied to clipboard!"));
  }
});


// KPI Checklist
function updateKpiScore() {
  const checkedCount = Array.from(kpiCheckboxes).filter(cb => cb.checked).length;
  const total = kpiCheckboxes.length;
  const pct = Math.round((checkedCount / total) * 100);
  kpiPercentageText.textContent = `${pct}%`;
  
  if (pct === 100) {
    kpiPercentageText.style.color = "var(--accent-teal)";
  } else if (pct > 0) {
    kpiPercentageText.style.color = "var(--color-amber)";
  } else {
    kpiPercentageText.style.color = "#FFFFFF";
  }
}

kpiCheckboxes.forEach(cb => cb.addEventListener("change", updateKpiScore));


// --- UTILITY TOAST ---
function showToast(message) {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.remove("d-none");
  
  setTimeout(() => {
    toast.classList.add("d-none");
  }, 2500);
}


// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
  initProfile();
  initKanban();
  refreshColumnCounts();
  loadKnowledgeFiles();
  
  // Set defaults for tools
  loadCommDefaults();
  loadRawSource();
  renderActiveSop("lead-routing");
});
