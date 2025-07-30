from langchain_community.chat_message_histories import ChatMessageHistory

def get_session_history(session_state, session_id):
    if session_id not in session_state.store:
        session_state.store[session_id] = ChatMessageHistory()
    return session_state.store[session_id]
