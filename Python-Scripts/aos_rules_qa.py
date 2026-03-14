import argparse
import os
import re
from typing import List

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Default data/index locations per game system
DEFAULT_AOS_DATA_PATH = os.path.join("Data", "aos_core_rules_text_clean.md")
DEFAULT_AOS_INDEX_DIR = os.path.join("Data", "aos_core_rules_faiss_index")

DEFAULT_WH40K_DATA_PATH = os.path.join("Data", "wh40k_core_rules_text_clean.md")
DEFAULT_WH40K_INDEX_DIR = os.path.join("Data", "wh40k_core_rules_faiss_index")


EXAMPLE_SYSTEM_PROMPT = """
You are a rules explainer for Warhammer tabletop games.

You answer questions ONLY using the provided rules text from the selected game's core rules. DO NOT use any other information or context. If the question is not related to the rules, say so.

Guidelines:
- If the rules text clearly answers the question, quote or closely paraphrase the relevant passages.
- If the answer depends on definitions, sequences, or edge cases, walk through them step by step.
- If the context does NOT contain enough information to answer with confidence, say you are unsure
  and clearly state what is missing rather than inventing rules.
- Do NOT reference page numbers unless they are explicitly present in the provided context.
- Be concise, but do not omit important conditions or exceptions.

When the user names a specific unit, ability, or keyword (for example, a unit name in a battle profile table):
- Prioritise any context snippets that mention that exact name (case-insensitive).
- Pay particular attention to short "notes" style sentences such as "This unit cannot be reinforced."
- If such a sentence is present in the context, treat it as authoritative for the question.

Assume the user is familiar with basic tabletop gaming, but not necessarily all Age of Sigmar jargon.
Explain specialised terms briefly when they are important to the answer.

Always structure your final answer in exactly this format (no extra sections, no bullet lists outside these fields):

Short Answer: <one-line answer, e.g. "Yes, it can.", "No, it cannot.", or "I’m not sure based on the provided rules.">

Detailed Answer: <a longer explanation of why the Short Answer is true, including relevant conditions, edge cases, and short quotations or paraphrases from the rules context.>

Source: <a brief description of where in the provided context this comes from, e.g. referencing unit names, section headings, or key phrases, but NOT page numbers unless they explicitly appear in the text.>
""".strip()


def build_index(
    data_path: str,
    index_dir: str,
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


def load_index(index_dir: str) -> FAISS:
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


def _extract_candidate_phrases(question: str) -> List[str]:
    """
    Heuristic extraction of proper-noun-like phrases from the user's question,
    e.g. "Freeguild Command Corps".
    """
    # Look for 2+ consecutive capitalised words.
    pattern = re.compile(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+)")
    phrases = [m.group(1).strip() for m in pattern.finditer(question)]

    # Deduplicate while preserving order and drop very short phrases.
    seen = set()
    result: List[str] = []
    for p in phrases:
        key = p.lower()
        if len(p) < 4 or key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def _find_keyword_snippets(
    full_text: str,
    phrases: List[str],
    max_snippets: int = 2,
    window: int = 400,
) -> List[str]:
    """
    Lightweight keyword-based lookup into the full rules text.

    For each candidate phrase, find the first occurrence (case-insensitive)
    and return a small window of surrounding text as a snippet.

    This is designed to improve recall for specific unit/ability lookups
    without significantly increasing token usage.
    """
    text_lower = full_text.lower()
    snippets: List[str] = []

    for phrase in phrases:
        if len(snippets) >= max_snippets:
            break
        needle = phrase.lower()
        idx = text_lower.find(needle)
        if idx == -1:
            continue

        start = max(0, idx - window // 2)
        end = min(len(full_text), idx + len(phrase) + window // 2)
        snippet = full_text[start:end].strip()
        if not snippet:
            continue

        snippets.append(snippet)

    return snippets


def answer_question(
    question: str,
    system_prompt: str,
    vectorstore: FAISS,
    game_label: str,
    # model_name: str = "gpt-4o-mini",
    model_name: str = "gpt-5.4",
    k: int = 4,
    full_rules_text: str | None = None,
) -> str:
    """
    Retrieve relevant rules text and ask the OpenAI chat model
    to answer the user's question based only on that context.
    """
    # Slightly higher k for better recall, but still modest to avoid
    # large context windows by default.
    context_snippets = retrieve_context(vectorstore, question, k=max(k, 6))

    # Optional keyword-based lookup for specific units/abilities to
    # augment the embedding-based retrieval without many extra tokens.
    keyword_snippets: List[str] = []
    if full_rules_text:
        phrases = _extract_candidate_phrases(question)
        if phrases:
            keyword_snippets = _find_keyword_snippets(full_rules_text, phrases)

    all_snippets: List[str] = context_snippets + keyword_snippets
    joined_context = "\n\n---\n\n".join(all_snippets)

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.2,
    )

    messages = [
        (
            "system",
            f"{system_prompt}\n\nYou are currently answering rules questions for {game_label}.",
        ),
        (
            "system",
            "Here is the relevant rules context from the selected game's core rules. "
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
            "Ask Warhammer core rules questions (AoS or 40K) using "
            "LangChain embeddings + FAISS + OpenAI (gpt-5-nano)."
        )
    )

    parser.add_argument(
        "--game",
        type=str,
        choices=["aos", "wh40k"],
        required=True,
        help=(
            "Which ruleset to use: 'aos' for Warhammer Age of Sigmar "
            "or 'wh40k' for Warhammer 40,000."
        ),
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
        help=(
            "Build (or rebuild) the FAISS index from the cleaned core rules "
            "markdown for the selected game."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help=(
            "Override the default cleaned core rules markdown path for the selected game. "
            "If omitted, a sensible default per game is used."
        ),
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        help=(
            "Override the directory to store / load the FAISS index. "
            "If omitted, a sensible default per game is used."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="gpt-4o-mini",
        # help="Chat model to use (e.g. gpt-4o-mini).",
        default="gpt-5.4",
        help="Chat model to use (e.g. gpt-5.4).",
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

    # Resolve data/index paths based on selected game and optional overrides.
    if args.game == "aos":
        default_data = DEFAULT_AOS_DATA_PATH
        default_index = DEFAULT_AOS_INDEX_DIR
        game_label = "Warhammer Age of Sigmar"
    else:
        default_data = DEFAULT_WH40K_DATA_PATH
        default_index = DEFAULT_WH40K_INDEX_DIR
        game_label = "Warhammer 40,000"

    data_path = args.data_path or default_data
    index_dir = args.index_dir or default_index

    # If requested, (re)build the index from the specified markdown.
    if args.build_index:
        build_index(data_path=data_path, index_dir=index_dir)
        # If the user only wanted to build the index, we can exit early.
        if not args.question:
            return 0

    # Load FAISS index for question answering
    vectorstore = load_index(index_dir=index_dir)

    # Load the full rules text once so we can do lightweight keyword-based
    # lookups to augment retrieval without extra network/token cost.
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            full_rules_text = f.read()
    except OSError:
        full_rules_text = None

    system_prompt = args.system_prompt
    model_name = args.model

    if args.question:
        # Single question mode
        answer = answer_question(
            question=args.question,
            system_prompt=system_prompt,
            vectorstore=vectorstore,
            game_label=game_label,
            model_name=model_name,
            full_rules_text=full_rules_text,
        )
        print("\n=== Answer ===\n")
        print(answer)
        return 0

    # Interactive CLI loop
    print(f"{game_label} Rules Q&A (LangChain + FAISS + OpenAI)")
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
            game_label=game_label,
            model_name=model_name,
            full_rules_text=full_rules_text,
        )
        print("\n--- Answer ---\n")
        print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

