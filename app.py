# app.py — Ad's up · GA4 Chat (UI épurée, sans sidebar) — Cloud-ready + fix lancement MCP
import os
import sys
import io
import json
import asyncio
import shutil
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd

from google import genai
from google.genai import types as gt
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# ======================
# 🔐 Secrets / Env (Cloud & local)
# ======================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_MODEL   = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
GOOGLE_PROJECT_ID = st.secrets.get("GOOGLE_PROJECT_ID", os.getenv("GOOGLE_PROJECT_ID", "adsupdata"))

# Écrire le JSON du service account dans /tmp (Cloud) ou utiliser un chemin local en fallback
adc_json = st.secrets.get("GA_ADC_JSON", os.getenv("GA_ADC_JSON", ""))
if adc_json:
    GOOGLE_APPLICATION_CREDENTIALS = "/tmp/ga_adc.json"
    with open(GOOGLE_APPLICATION_CREDENTIALS, "w", encoding="utf-8") as f:
        f.write(adc_json)
else:
    # Fallback local pour le dev (chemin Windows de ton poste)
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", r"C:\Users\lione\.gemini\mcp.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_PROJECT_ID"] = GOOGLE_PROJECT_ID


# ======================
# 🎨 Style global (pas de sidebar) + suppression bandes blanches
# ======================
st.set_page_config(page_title="Ad's up — GA4 Chat", page_icon="📊", layout="wide")
st.markdown("""
<style>
/* === Supprimer les bandes blanches === */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stHeader"]::before { display:none !important; }
div[role="banner"]{ box-shadow:none !important; border-bottom:0 !important; }

/* Barre claire de progression (spinner) */
div[data-testid="stStatusWidget"] div[role="progressbar"]{ display:none !important; }

/* Liseré au-dessus de l'input chat */
div[data-testid="stChatInput"] > div:first-child{ border-top:0 !important; }
div[data-testid="stChatInput"]{ background: transparent !important; }

/* HR / séparateurs résiduels */
hr{ border:none !important; height:0 !important; }

/* Tuning visuel */
.block-container { padding-top: 1.1rem !important; }
.hero {
  border: 1px solid #e5e7eb; border-radius: 16px; padding: 18px 18px;
  background: linear-gradient(135deg, #0ea5e911 0%, #0f172a 35%, #111827 100%);
}
.brand {
  font-size: 26px; font-weight: 900; letter-spacing: -0.02em;
  background: linear-gradient(90deg,#0ea5e9 0%, #22c55e 50%, #a855f7 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.chips span{
  display:inline-block; padding:6px 12px; border-radius:999px;
  background:#0b1220; border:1px solid #1f2937; font-size:12px; color:#e5e7eb; margin-right:6px;
}
.glass {
  background: rgba(15,23,42,0.45);
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 16px; padding: 12px 14px 6px 14px; margin-top: 12px;
  box-shadow: 0 10px 30px rgba(2,8,23,0.25);
}
.kpi-card {
  border:1px solid #1f2937; border-radius:14px; padding:14px; background:#0b1220;
}
.kpi-title { font-size:12px; color:#94a3b8; margin-bottom:6px; }
.kpi-value { font-size:22px; font-weight:700; color:#e5e7eb; }
.stChatMessage { margin-bottom: 8px !important; }
.top-actions { text-align:right; }
.top-actions button { transform: translateY(4px); }
</style>
""", unsafe_allow_html=True)


# ======================
# Helpers texte
# ======================
def extract_text(resp) -> str:
    if getattr(resp, "text", None):
        return resp.text
    out = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            if hasattr(part, "text") and part.text:
                out.append(part.text)
    return "\n".join(out).strip()

def extract_tool_call(resp):
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            if hasattr(part, "function_call") and part.function_call:
                return part.function_call
    return None


# ======================
# Helpers contexte (mémoire modèle)
# ======================
def as_user(text: str) -> gt.Content:
    return gt.Content(role="user", parts=[gt.Part(text=text)])

def as_model_text(text: str) -> gt.Content:
    return gt.Content(role="model", parts=[gt.Part(text=text)])

def as_model_call(call) -> gt.Content:
    return gt.Content(role="model", parts=[gt.Part(function_call=call)])

def as_tool_resp(name: str, resp) -> gt.Content:
    if not isinstance(resp, dict):
        resp = {"raw": str(resp)}
    return gt.Content(role="tool", parts=[gt.Part(function_response={"name": name, "response": resp})])

def trim_history(msgs: List[gt.Content], keep_last: int = 30):
    if len(msgs) > keep_last:
        del msgs[:-keep_last]


# ======================
# GA4 utils (affichage)
# ======================
def ga4_rows_to_df(report: Dict[str, Any]) -> Optional[pd.DataFrame]:
    try:
        dim_headers = [d["name"] for d in report.get("dimensionHeaders", [])]
        met_headers = [m["name"] for m in report.get("metricHeaders", [])]
        rows = report.get("rows", [])
        if not rows:
            return None
        table = []
        for r in rows:
            dims = r.get("dimensionValues", [])
            mets = r.get("metricValues", [])
            row = {}
            for i, d in enumerate(dim_headers):
                row[d] = dims[i]["value"] if i < len(dims) else None
            for j, m in enumerate(met_headers):
                row[m] = mets[j]["value"] if j < len(mets) else None
            table.append(row)
        df = pd.DataFrame(table)
        for m in met_headers:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="ignore")
        return df
    except Exception:
        return None

def ga4_totals(report: Dict[str, Any]) -> Dict[str, Any]:
    totals = report.get("totals") or report.get("metricTotals") or []
    if not totals:
        return {}
    met_headers = [m["name"] for m in report.get("metricHeaders", [])]
    mv = totals[0].get("metricValues", []) if isinstance(totals, list) else []
    out = {}
    for i, m in enumerate(met_headers):
        if i < len(mv):
            out[m] = mv[i].get("value")
    return out


# ======================
# 🚀 MCP côté Cloud : préférer le binaire, fallback vers python -m
# ======================
def _build_server_params() -> StdioServerParameters:
    # 1) Essayer le binaire "analytics-mcp" (console script installé via requirements)
    if shutil.which("analytics-mcp"):
        return StdioServerParameters(
            command="analytics-mcp",
            args=[],
            env={
                "GOOGLE_APPLICATION_CREDENTIALS": os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
                "GOOGLE_PROJECT_ID": os.environ.get("GOOGLE_PROJECT_ID", ""),
            },
        )
    # 2) Fallback : tenter "python -m analytics_mcp"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "analytics_mcp"],
        env={
            "GOOGLE_APPLICATION_CREDENTIALS": os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
            "GOOGLE_PROJECT_ID": os.environ.get("GOOGLE_PROJECT_ID", ""),
        },
    )

SERVER_PARAMS = _build_server_params()

def _assert_mcp_available():
    if not shutil.which("analytics-mcp"):
        st.info("ℹ️ Lancement via `python -m analytics_mcp` (binaire `analytics-mcp` introuvable dans PATH).")

_assert_mcp_available()


# ======================
# MCP : Tools (utilise SERVER_PARAMS)
# ======================
async def list_mcp_tools() -> List[gt.Tool]:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tl = await session.list_tools()
            tools = [
                gt.Tool(function_declarations=[{
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": {"type": "object", "properties": {}},
                }])
                for t in tl.tools
            ]
            return tools


# ======================
# Un tour de chat (identique, avec capture payload)
# ======================
async def run_one_turn(user_q: str, tools: List[gt.Tool], client: genai.Client) -> str:
    # 1) Ajouter le message user au contexte modèle
    st.session_state.messages.append(as_user(user_q))

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 2) Premier appel (avec tout l'historique)
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=st.session_state.messages,
                config=gt.GenerateContentConfig(temperature=0, tools=tools),
            )

            call = extract_tool_call(resp)
            if not call:
                text = extract_text(resp) or "(pas de réponse)"
                st.session_state.messages.append(as_model_text(text))
                trim_history(st.session_state.messages)
                return text

            # 3) Mémoriser l'appel outil
            st.session_state.messages.append(as_model_call(call))

            # Exécuter l’outil
            args = dict(call.args or {})
            result = await session.call_tool(call.name, args)
            raw = result.content[0].text if result.content else "{}"
            try:
                tool_json = json.loads(raw)
            except json.JSONDecodeError:
                tool_json = {"raw": raw}

            # Stocker la réponse outil (pour l'historique + affichage)
            st.session_state.last_tool_payload = {"name": call.name, "response": tool_json}
            st.session_state.messages.append(as_tool_resp(call.name, tool_json))

            # 4) Deuxième appel : réponse finale avec tout l'historique
            resp2 = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=st.session_state.messages,
                config=gt.GenerateContentConfig(temperature=0),
            )
            final_text = extract_text(resp2) or "(aucun texte)"
            st.session_state.messages.append(as_model_text(final_text))
            trim_history(st.session_state.messages)
            return final_text


# ======================
# HEADER compact + bouton reset (pas de sidebar)
# ======================
with st.container():
    c1, c2 = st.columns([0.75, 0.25])
    with c1:
        st.markdown(
    """
    <div class="hero">
      <div class="brand">GA4 Analyzer - Ad's up Consulting</div>
      <div style="color:#e5e7eb; margin-top:4px; font-weight:700;">
        Plateforme copilote GA4 propulsée par Gemini : insights instantanés
      </div>
      <div class="chips" style="margin-top:8px;">
        <span>Gemini 2.5</span><span>GA4 Tools</span><span>Mémoire multi-tours</span>
      </div>
    </div>
    """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown('<div class="top-actions">', unsafe_allow_html=True)
        if st.button("🔄 Réinitialiser"):
            st.session_state.history = []
            st.session_state.messages = []
            st.session_state.last_tool_payload = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ======================
# Vérifs de base
# ======================
if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY manquante (Secrets).")
    st.stop()
if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    st.error(f"⚠️ Clé GA4 introuvable : {GOOGLE_APPLICATION_CREDENTIALS}")
    st.stop()

# État (mémoire UI + modèle)
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_tool_payload" not in st.session_state:
    st.session_state.last_tool_payload = None

# Client Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Charger / mettre en cache la liste des outils MCP une fois
@st.cache_resource(show_spinner=True)
def _load_tools():
    return asyncio.run(list_mcp_tools())

tools = _load_tools()

# ======================
# Zone de chat dans une carte “glass”
# ======================
st.markdown('<div class="glass">', unsafe_allow_html=True)

user_q = st.chat_input("Pose ta question")
if user_q:
    st.session_state.history.append(("user", user_q))
    with st.spinner("Gemini réfléchit…"):
        try:
            answer = asyncio.run(run_one_turn(user_q, tools, client))
        except Exception as e:
            answer = f"❌ Erreur: {type(e).__name__}: {e}"
    st.session_state.history.append(("assistant", answer))

# Bulles
for role, msg in st.session_state.history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)

st.markdown('</div>', unsafe_allow_html=True)

# (Section "Résultats GA4 (KPIs, tableau, exports)" supprimée comme demandé)
