"""Answer a question from retrieved chunks.

Two modes:
  * extractive (default, no API key needed): show the most relevant passages with sources
  * LLM mode: a LangChain chain (prompt | llm | parser) writes an answer grounded in the passages
"""
import os

PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def format_context(hits):
    return "\n\n".join(f"[{i}] ({h['source']}, p.{h['page']}) {h['text']}" for i, h in enumerate(hits, 1))


def get_llm():
    """Return a LangChain chat model if an API key is configured, else None."""
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return None


def answer_with_llm(question, hits, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": format_context(hits), "question": question})


def answer_extractive(hits):
    if not hits:
        return "No relevant passages found."
    lines = ["Most relevant passages:"]
    for i, h in enumerate(hits, 1):
        lines.append(f"\n[{i}] {h['source']} (page {h['page']}, similarity {h['score']:.2f})\n{h['text']}")
    return "\n".join(lines)


def answer(question, hits, llm=None):
    return answer_with_llm(question, hits, llm) if llm is not None else answer_extractive(hits)
