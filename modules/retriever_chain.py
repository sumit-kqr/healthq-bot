from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

def build_conversational_rag_chain(llm, retriever, get_session_history_fn):
    from modules.prompts import get_contextualize_prompt, get_qa_prompt
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, get_contextualize_prompt()
    )

    question_answer_chain = create_stuff_documents_chain(
        llm, get_qa_prompt()
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history_fn,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_chain
