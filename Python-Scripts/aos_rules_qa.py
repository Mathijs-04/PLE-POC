import argparse
import os
from typing import List

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


DEFAULT_DATA_PATH = os.path.join("Data", "aos_core_rules_text_clean.md")
DEFAULT_INDEX_DIR = os.path.join("Data", "aos_aos_core_rules_faiss_index")


EXAMPLE_SYSTEM_PROMPT = """
You are a rules explainer for Warhammer Age of Sigmar.

You answer questions ONLY using the provided rules text from the core rules. DO NOT use any other information or context. If the question is not related to the rules, say so.

Guidelines:
- If the rules text clearly answers the question, quote or closely paraphrase the relevant passages.
- If the answer depends on definitions, sequences, or edge cases, walk through them step by step.
- If the context does NOT contain enough information to answer with confidence, say you are unsure
  and clearly state what is missing rather than inventing rules.
- Do NOT reference page numbers unless they are explicitly present in the provided context.
- Be concise, but do not omit important conditions or exceptions.

Assume the user is familiar with basic tabletop gaming, but not necessarily all Age of Sigmar jargon.
Explain specialised terms briefly when they are important to the answer.
""".strip()


def build_index(
    data_path: str = DEFAULT_DATA_PATH,
    index_dir: str = DEFAULT_INDEX_DIR,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> None:
    """
    Build (or rebuild) a FAISS index from the cleaned AoS core rules markdown.
    The resulting index is stored on disk in `index_dir`.
    """
    if not os.path.isfile(data_path):
        raise SystemExit(f"Rules file not found: {data_path}")

    print(f"Loading rules from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n#", "\n\n", "\n", " "],
    )
    docs = splitter.create_documents([full_text])

    print(f"Split into {len(docs)} chunks for embedding.")

    embeddings = OpenAIEmbeddings(
        # You can change this to another embedding model if desired
        model="text-embedding-3-small",
    )

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # Ensure directory exists
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"Saved FAISS index to: {index_dir}")


def load_index(index_dir: str = DEFAULT_INDEX_DIR) -> FAISS:
    """
    Load a previously built FAISS index from disk.
    """
    if not os.path.isdir(index_dir):
        raise SystemExit(
            f"FAISS index not found at '{index_dir}'. "
            f"Build it first with: python aos_rules_qa.py --build-index"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def retrieve_context(
    vectorstore: FAISS,
    question: str,
    k: int = 4,
) -> List[str]:
    """
    Retrieve the top-k most relevant text chunks for the user's question.
    Returns a list of raw text snippets.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # In newer LangChain versions, retrievers are LCEL runnables;
    # use .invoke(...) instead of .get_relevant_documents(...)
    docs = retriever.invoke(question)
    return [d.page_content for d in docs]


def answer_question(
    question: str,
    system_prompt: str,
    vectorstore: FAISS,
    model_name: str = "gpt-5-nano",
    k: int = 4,
) -> str:
    """
    Retrieve relevant rules text and ask the OpenAI chat model
    to answer the user's question based only on that context.
    """
    context_snippets = retrieve_context(vectorstore, question, k=k)

    joined_context = "\n\n---\n\n".join(context_snippets)

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
    )

    messages = [
        (
            "system",
            system_prompt,
        ),
        (
            "system",
            "Here is the relevant rules context from the Warhammer Age of Sigmar core rules. "
            "Only use information that appears here when answering.\n\n"
            f"{joined_context}",
        ),
        ("user", question),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def main() -> int:
    load_dotenv()  # for OPENAI_API_KEY

    parser = argparse.ArgumentParser(
        description=(
            "Ask Warhammer Age of Sigmar core rules questions using "
            "LangChain embeddings + FAISS + OpenAI (gpt-5-nano)."
        )
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
        help=(
            "Build (or rebuild) the FAISS index from the cleaned AoS core rules "
            f"markdown. Default input: {DEFAULT_DATA_PATH}"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the cleaned AoS core rules markdown file.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help="Directory to store / load the FAISS index.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-nano",
        help="Chat model to use (e.g. gpt-5-nano).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=EXAMPLE_SYSTEM_PROMPT,
        help=(
            "Custom system prompt text for the assistant. "
            "If omitted, a default AoS rules-explainer prompt is used."
        ),
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="Optional one-off question. If omitted, an interactive REPL is started.",
    )

    args = parser.parse_args()

    if args.build_index:
        build_index(data_path=args.data_path, index_dir=args.index_dir)
        # If the user only wanted to build the index, we can exit early.
        if not args.question:
            return 0

    # Load FAISS index for question answering
    vectorstore = load_index(index_dir=args.index_dir)

    system_prompt = args.system_prompt
    model_name = args.model

    if args.question:
        # Single question mode
        answer = answer_question(
            question=args.question,
            system_prompt=system_prompt,
            vectorstore=vectorstore,
            model_name=model_name,
        )
        print("\n=== Answer ===\n")
        print(answer)
        return 0

    # Interactive CLI loop
    print("AoS Rules Q&A (LangChain + FAISS + OpenAI)")
    print("Type 'exit', 'quit', or Ctrl+C to stop.")

    while True:
        try:
            user_q = input("\nYour question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = answer_question(
            question=user_q,
            system_prompt=system_prompt,
            vectorstore=vectorstore,
            model_name=model_name,
        )
        print("\n--- Answer ---\n")
        print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

