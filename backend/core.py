import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def get_vectorstore():
    """Initialize and return the Pinecone vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "dochelp"), embedding=embeddings
    )


def run_llm(
    query: str, chat_history: List[Dict[str, Any]] = [], model: str = "gpt-4o-mini"
):
    """Main function to run LLM with retrieval-augmented generation."""
    docsearch = get_vectorstore()
    chat = ChatOpenAI(model=model, verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm_simple(
    query: str, chat_history: List[Dict[str, Any]] = [], model: str = "gpt-4o-mini"
):
    """Alternative implementation using LCEL chain for simple RAG."""
    docsearch = get_vectorstore()
    chat = ChatOpenAI(model=model, verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    rag_chain = (
        {
            "context": docsearch.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()

    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
        answer=rag_chain
    )

    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result
