import streamlit as st
from ui.layout import sidebar_navigation, display_chat_ui
from ui.chat_handler import handle_query, clear_cache_handler

APP_TITLE = "Semantic RAG Chat"
SIDEBAR_TITLE = "Navigation"
CHAT_TAB = "💬 Chat Assistant"
STATS_TAB = "📊 Query Stats"

st.set_page_config(page_title=APP_TITLE, layout="wide")
tab = sidebar_navigation()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if tab == CHAT_TAB:
    # Clear cache button
    if st.sidebar.button("🧹 Clear Cache", use_container_width=True):
        clear_cache_handler()
        st.sidebar.success("Cache memory cleared ✅")

    # Display conversation
    query = display_chat_ui(st.session_state.chat_history)

    if query:
        response = handle_query(query)
        st.session_state.chat_history.append((query, response))
        st.rerun()

elif tab == STATS_TAB:
    st.title("📊 User Query Statistics (Coming Soon)")
    st.markdown("This section will show usage insights, top queries, cache hits, etc.")
