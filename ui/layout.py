import streamlit as st
# from ui.config import CHAT_TAB, STATS_TAB
APP_TITLE = "Semantic RAG Chat"
SIDEBAR_TITLE = "Navigation"
CHAT_TAB = "💬 Chat Assistant"
STATS_TAB = "📊 Query Stats"

def sidebar_navigation():
    return st.sidebar.radio("Go to", [CHAT_TAB, STATS_TAB])

def display_chat_ui(chat_history):
    """
    Render chat messages in Streamlit interface.
    """
    st.title("Semantic Chat Assistant")
    
    for user, bot in chat_history:
        with st.chat_message("user"):
            st.markdown(user)
        with st.chat_message("assistant"):
            st.markdown(bot)

    # Chat input
    user_input = st.chat_input("Ask something about Azure...")
    return user_input
