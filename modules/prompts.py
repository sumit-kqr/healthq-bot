from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_contextualize_prompt():
    system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

def get_qa_prompt():
    system_prompt = (
      """
      You are an intelligent and reliable insurance assistant tasked with helping users understand health insurance policies.

      Carefully analyze the **context** provided from official insurance documents to answer the user’s **question**. Your response must be:
      - Accurate, complete, and free of assumptions
      - Written in **at least 50 words**
      - Structured clearly and easy to understand for a general audience
      - Detailed if the context includes multiple points

      Do not fabricate information. If the answer is not found in the context, simply respond with: **"I'm sorry, I couldn't find that information in the provided policy documents."**

      Avoid redundancy and overly generic language.

      ---
      Context:
      {context}

      Answer:
      """
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
