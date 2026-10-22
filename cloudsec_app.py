"""
cloudsec_app.py -- the CloudSec web app: one page per job, switched from the sidebar.

    Agent Gate       agent_console.py       ask for an AWS change; screen, harden, predict, assess, execute
    Live Dashboard   cloudsec_dashboard.py  the real-time pipeline's scored events, alerts and explanations

Started by cloudsec.py (run.cmd / run.sh) on http://localhost:8503. Each page also runs on its own
(streamlit run agent_console.py / cloudsec_dashboard.py).
"""
import streamlit as st

st.set_page_config(page_title="CloudSec", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stSidebar"]{background:#0d0e15;border-right:1px solid #242835}
  [data-testid="stSidebarCollapseButton"],[data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarHeader"]{display:none}
  [data-testid="stSidebarNav"]::before{content:"🛡️  CloudSec";display:block;padding:18px 18px 12px;
      font-weight:700;font-size:16px;letter-spacing:.02em;color:#e7e9f0}
  [data-testid="stSidebarNavLink"]{border-radius:8px;margin:2px 10px}
  [data-testid="stSidebarNavLink"][aria-current="page"]{background:#1c2033}
</style>
""", unsafe_allow_html=True)

AGENT_GATE = st.Page("agent_console.py", title="Agent Gate", icon=":material/shield:", default=True)
DASHBOARD = st.Page("cloudsec_dashboard.py", title="Live Dashboard", icon=":material/monitoring:")

st.navigation([AGENT_GATE, DASHBOARD]).run()
