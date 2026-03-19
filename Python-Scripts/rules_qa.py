import argparse
import os
import re
import string
import difflib
from typing import Iterable, List, Sequence, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Default data/index locations per game system
DEFAULT_AOS_DATA_DIR = os.path.join("Data", "AOS-Datafiles")
DEFAULT_AOS_INDEX_DIR = os.path.join("Data", "aos_rules_faiss_index")

DEFAULT_WH40K_DATA_DIR = os.path.join("Data", "40K-Datafiles")
DEFAULT_WH40K_INDEX_DIR = os.path.join("Data", "wh40k_rules_faiss_index")


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

**Short Answer:** <one-line answer, e.g. "Yes, it can.", "No, it cannot.", or "I’m not sure based on the provided rules.">

**Detailed Answer:** <a longer explanation of why the Short Answer is true, including relevant conditions, edge cases, and short quotations or paraphrases from the rules context.>

**Source:** <a brief description of where in the provided context this comes from, e.g. referencing unit names, section headings, or key phrases, but NOT page numbers unless they explicitly appear in the text.>
""".strip()


def _iter_markdown_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Rules directory not found: {data_dir}")

    paths: List[str] = []
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(".md"):
            paths.append(os.path.join(data_dir, name))
    if not paths:
        raise SystemExit(f"No markdown files found in: {data_dir}")
    return paths


def _infer_doc_kind_and_faction(md_path: str, game: str) -> Tuple[str, str | None]:
    """
    Returns (doc_kind, faction).
    doc_kind is "core_rules" or "faction_rules" (or "supplement" for misc).
    faction is None for core rules, else a best-effort faction name.
    """
    base = os.path.basename(md_path)
    stem = os.path.splitext(base)[0].strip()
    stem_lower = stem.lower()

    if game == "aos":
        if "core" in stem_lower and "rule" in stem_lower:
            return "core_rules", None
    else:
        if "core" in stem_lower and "rule" in stem_lower:
            return "core_rules", None

    # Some AoS files are non-faction supplements (e.g. Lores, Path to Glory)
    if game == "aos" and stem_lower in {"lores", "regiments of renown"}:
        return "supplement", None
    if game == "aos" and stem_lower.startswith("path to glory"):
        return "supplement", None

    return "faction_rules", stem


def _load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _split_markdown_to_documents(
    md_text: str,
    *,
    source_path: str,
    game: str,
    doc_kind: str,
    faction: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split a markdown file into chunks while preserving structure via metadata.

    Strategy:
    - First split by markdown headers to keep sections coherent.
    - Then further chunk using a recursive character splitter.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
    )
    header_docs = header_splitter.split_text(md_text)

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    out: List[Document] = []
    for hd in header_docs:
        # Carry over header-derived metadata (h1/h2/h3/h4 keys)
        meta = dict(hd.metadata or {})
        meta.update(
            {
                "source": source_path,
                "game": game,
                "doc_kind": doc_kind,
            }
        )
        if faction:
            meta["faction"] = faction

        # Now chunk this section further if needed.
        for part in recursive.split_text(hd.page_content):
            content = part.strip()
            if not content:
                continue
            out.append(Document(page_content=content, metadata=meta))

    return out


def load_rules_corpus_text(data_dir: str) -> str:
    """
    Load and concatenate all markdown files in the data directory.
    Used only for lightweight keyword snippets (not embeddings).
    """
    parts: List[str] = []
    for p in _iter_markdown_files(data_dir):
        try:
            parts.append(_load_md(p))
        except OSError:
            continue
    return "\n\n---\n\n".join([t for t in parts if t])


def load_rules_sources(data_dir: str) -> List[Tuple[str, str]]:
    """
    Load all markdown files from the data directory as (path, text).
    Used for robust structure-aware keyword lookups.
    """
    sources: List[Tuple[str, str]] = []
    for p in _iter_markdown_files(data_dir):
        try:
            sources.append((p, _load_md(p)))
        except OSError:
            continue
    return sources


def build_index(
    data_dir: str,
    index_dir: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    game: str = "aos",
) -> None:
    """
    Build (or rebuild) a FAISS index from the structured markdown rules files.
    The resulting index is stored on disk in `index_dir`.
    """
    md_paths = _iter_markdown_files(data_dir)
    print(f"Loading rules from {len(md_paths)} markdown files in: {data_dir}")

    docs: List[Document] = []
    for p in md_paths:
        md_text = _load_md(p)
        doc_kind, faction = _infer_doc_kind_and_faction(p, game=game)
        docs.extend(
            _split_markdown_to_documents(
                md_text,
                source_path=p,
                game=game,
                doc_kind=doc_kind,
                faction=faction,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    print(f"Split into {len(docs)} chunks for embedding across all files.")

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
            f"Build it first with: python rules_qa.py --game aos --build-index"
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
    # Use MMR for better diversity/recall across structured docs.
    # This helps when a query matches both "mentions" and the authoritative entry.
    docs = vectorstore.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=max(30, k * 6),
        lambda_mult=0.5,
    )
    snippets: List[str] = []
    for d in docs:
        meta = d.metadata or {}
        # Keep the context self-identifying to reduce cross-faction confusion.
        source_bits: List[str] = []
        if meta.get("doc_kind"):
            source_bits.append(str(meta["doc_kind"]))
        if meta.get("faction"):
            source_bits.append(str(meta["faction"]))
        if meta.get("h2"):
            source_bits.append(str(meta["h2"]))
        elif meta.get("h1"):
            source_bits.append(str(meta["h1"]))

        label = " | ".join(source_bits).strip()
        if label:
            snippets.append(f"[{label}]\n{d.page_content}")
        else:
            snippets.append(d.page_content)
    return snippets


def _normalize_name(s: str) -> str:
    s = s.strip().lower()
    # keep alnum + spaces only
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_HEADING_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "without",
    "unit",
    "units",
    "model",
    "models",
    "squad",
    "team",
    "detachment",
}


def _tokenize_for_match(s: str) -> List[str]:
    norm = _normalize_name(s)
    if not norm:
        return []
    toks = [t for t in norm.split(" ") if t and t not in _HEADING_STOPWORDS]
    return toks


def _token_overlap_score(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def _extract_candidate_phrases(question: str) -> List[str]:
    """
    Heuristic extraction of proper-noun-like phrases from the user's question,
    e.g. "Freeguild Command Corps".
    """
    phrases: List[str] = []
    q_lower = question.lower()

    # Prefer anything the user explicitly puts in quotes (often ability names).
    quoted = re.compile(r"[\"'“”]([^\"'“”]{3,80})[\"'“”]")
    for m in quoted.finditer(question):
        phrases.append(m.group(1).strip())

    # If the user asks about points, try to capture the unit name even if it's lowercase,
    # e.g. "How many points is a unit clanrats".
    if "point" in q_lower or "points" in q_lower or "pts" in q_lower:
        after_unit = re.compile(
            r"\bunit\s+([A-Za-z][A-Za-z'’\-]+(?:\s+[A-Za-z][A-Za-z'’\-]+){0,4})",
            re.IGNORECASE,
        )
        for m in after_unit.finditer(question):
            phrases.append(m.group(1).strip())

    # Look for 2+ consecutive capitalised words (typical unit names).
    multi_cap = re.compile(r"([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+)")
    for m in multi_cap.finditer(question):
        phrases.append(m.group(1).strip())

    # Also allow a single TitleCase token (e.g. Mirrorshield, DeepStrike).
    single_cap = re.compile(r"\b([A-Z][A-Za-z]{3,})\b")
    for m in single_cap.finditer(question):
        phrases.append(m.group(1).strip())

    # Finally, if the user explicitly mentions "unit <name>" but it's lowercase (common),
    # capture that too (kept tight to avoid adding too many generic tokens).
    if "unit " in q_lower:
        after_unit_anycase = re.compile(
            r"\bunit\s+([a-z][a-z'’\-]{3,}(?:\s+[a-z][a-z'’\-]{3,}){0,4})\b",
            re.IGNORECASE,
        )
        for m in after_unit_anycase.finditer(question):
            phrases.append(m.group(1).strip())

    # Also capture a trailing noun phrase like "points is clanrats" (no 'unit' keyword).
    points_tail = re.compile(
        r"\bpoints?\b[^A-Za-z0-9]{0,10}(?:is|are|for)\s+([A-Za-z][A-Za-z'’\-]+(?:\s+[A-Za-z][A-Za-z'’\-]+){0,4})",
        re.IGNORECASE,
    )
    for m in points_tail.finditer(question):
        phrases.append(m.group(1).strip())

    # Deduplicate while preserving order and drop very short phrases.
    seen = set()
    result: List[str] = []
    for p in phrases:
        key = _normalize_name(p)
        if len(p) < 4 or key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result


def _extract_markdown_section(md_text: str, start_pos: int, start_level: int) -> str:
    """
    Return markdown from the heading at start_pos until the next heading
    of same or higher level.
    """
    # Find next heading after start_pos
    heading_re = re.compile(r"^(#{1,6})\s+.+$", re.MULTILINE)
    for m in heading_re.finditer(md_text, start_pos + 1):
        level = len(m.group(1))
        if level <= start_level:
            return md_text[start_pos : m.start()].strip()
    return md_text[start_pos:].strip()


def _find_heading_sections_in_sources(
    sources: Sequence[Tuple[str, str]],
    phrase: str,
    *,
    prefer_points: bool,
    max_sections: int,
) -> List[str]:
    """
    Structure-aware lookup: find headings like '### Clanrats' and return the section.
    """
    needle_norm = _normalize_name(phrase)
    needle_tokens = _tokenize_for_match(phrase)
    if not needle_norm and not needle_tokens:
        return []

    results: List[tuple[int, str]] = []
    heading_re = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.MULTILINE)

    for path, text in sources:
        file_stem = os.path.splitext(os.path.basename(path))[0]
        file_tokens = set(_tokenize_for_match(file_stem))

        for m in heading_re.finditer(text):
            level = len(m.group(1))
            heading = m.group(2).strip()
            heading_norm = _normalize_name(heading)
            heading_tokens = _tokenize_for_match(heading)

            # Exact match is best, but also allow fuzzy match:
            # - high token overlap (subset/superset)
            # - high string similarity (handles small variations)
            exact = heading_norm == needle_norm and bool(heading_norm)
            tok_overlap = _token_overlap_score(needle_tokens, heading_tokens)
            seq_sim = difflib.SequenceMatcher(None, needle_norm, heading_norm).ratio() if needle_norm and heading_norm else 0.0

            if not exact and tok_overlap < 0.45 and seq_sim < 0.72:
                continue

            section = _extract_markdown_section(text, m.start(), level)
            if not section:
                continue

            score = 0
            if exact:
                score += 25
            # Token overlap strongly indicates it's the same unit (e.g. "Black Templar Sword Brethren" vs "Sword Brethren Squad")
            score += int(tok_overlap * 20)
            score += int(seq_sim * 10)
            if prefer_points and "**Points:**".lower() in section.lower():
                score += 12

            # If the user's phrase contains faction-ish tokens and those appear in the filename,
            # boost headings from that file. Example: "Black Templar ..." -> Black Templars.md
            if file_tokens and needle_tokens:
                faction_overlap = len(set(needle_tokens) & file_tokens)
                score += min(8, faction_overlap * 3)

            label = os.path.basename(path)
            results.append((score, f"[{label} | heading]\n{section}"))

    results.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in results[:max_sections]]


def _find_best_windows_in_sources(
    sources: Sequence[Tuple[str, str]],
    phrase: str,
    *,
    prefer_points: bool,
    window: int,
    max_windows: int,
) -> List[str]:
    """
    Fallback keyword search across sources using multiple occurrences and scoring.
    """
    out: List[tuple[int, str]] = []
    phrase_clean = phrase.strip().strip(string.punctuation)
    if len(phrase_clean) < 3:
        return []

    for path, text in sources:
        matches = list(re.finditer(re.escape(phrase_clean), text, flags=re.IGNORECASE))
        if not matches:
            continue

        for m in matches[:50]:
            idx = m.start()
            start = max(0, idx - window // 2)
            end = min(len(text), idx + len(phrase_clean) + window // 2)
            snippet = text[start:end].strip()
            if not snippet:
                continue

            s_lower = snippet.lower()
            score = 0
            if prefer_points and "**points:**" in s_lower:
                score += 8
            # If snippet contains a nearby heading with the phrase, boost hard.
            if re.search(rf"^###\s+{re.escape(phrase_clean)}\s*$", snippet, flags=re.IGNORECASE | re.MULTILINE):
                score += 12
            if re.search(rf"^##\s+{re.escape(phrase_clean)}\s*$", snippet, flags=re.IGNORECASE | re.MULTILINE):
                score += 6

            # small boost for being in the same file name (often faction file)
            if _normalize_name(os.path.splitext(os.path.basename(path))[0]).find(_normalize_name(phrase_clean)) != -1:
                score += 2

            label = os.path.basename(path)
            out.append((score, f"[{label} | match]\n{snippet}"))

    out.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in out[:max_windows]]


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
    # Backwards-compatible wrapper: if we only have a single concatenated string,
    # we can't do structure-aware per-file extraction. Prefer callers to pass sources.
    # We still try to bias toward headings/points, but this is less reliable.
    snippets: List[str] = []
    for phrase in phrases:
        if len(snippets) >= max_snippets:
            break
        matches = list(re.finditer(re.escape(phrase.strip()), full_text, flags=re.IGNORECASE))
        if not matches:
            continue
        # Prefer occurrences close to a markdown heading of the phrase.
        best_score = -1
        best_snip = ""
        for m in matches[:50]:
            idx = m.start()
            start = max(0, idx - window // 2)
            end = min(len(full_text), idx + len(phrase) + window // 2)
            snippet = full_text[start:end].strip()
            if not snippet:
                continue
            s_lower = snippet.lower()
            score = 0
            if "**points:**" in s_lower:
                score += 6
            if f"### {phrase.lower()}" in s_lower:
                score += 10
            if f"## {phrase.lower()}" in s_lower:
                score += 4
            if score > best_score:
                best_score = score
                best_snip = snippet
        if best_snip:
            snippets.append(best_snip)
    return snippets


def find_keyword_snippets(
    sources: Sequence[Tuple[str, str]],
    question: str,
    phrases: List[str],
    *,
    max_snippets: int = 3,
    window: int = 600,
) -> List[str]:
    """
    Robust, structure-aware keyword lookup across the corpus.
    """
    q_lower = question.lower()
    prefer_points = ("point" in q_lower) or ("points" in q_lower) or ("pts" in q_lower)

    out: List[str] = []
    for phrase in phrases:
        if len(out) >= max_snippets:
            break

        # 1) Prefer exact heading sections (unit/ability definitions).
        sections = _find_heading_sections_in_sources(
            sources, phrase, prefer_points=prefer_points, max_sections=1
        )
        if sections:
            out.extend(sections)
            continue

        # 2) Fallback to scored windows across sources.
        windows = _find_best_windows_in_sources(
            sources,
            phrase,
            prefer_points=prefer_points,
            window=window,
            max_windows=1,
        )
        if windows:
            out.extend(windows)

    return out[:max_snippets]


def answer_question(
    question: str,
    system_prompt: str,
    vectorstore: FAISS,
    game_label: str,
    # model_name: str = "gpt-4o-mini",
    model_name: str = "gpt-5.4",
    k: int = 4,
    full_rules_text: str | None = None,
    full_rules_sources: Sequence[Tuple[str, str]] | None = None,
) -> str:
    """
    Retrieve relevant rules text and ask the OpenAI chat model
    to answer the user's question based only on that context.
    """
    # Slightly higher k for better recall, but still modest to avoid
    # large context windows by default.
    context_snippets = retrieve_context(vectorstore, question, k=max(k, 10))

    # Optional keyword-based lookup for specific units/abilities to
    # augment the embedding-based retrieval without many extra tokens.
    keyword_snippets: List[str] = []
    phrases = _extract_candidate_phrases(question)
    if full_rules_sources and phrases:
        keyword_snippets = find_keyword_snippets(
            full_rules_sources,
            question=question,
            phrases=phrases,
            max_snippets=3,
        )
    elif full_rules_text and phrases:
        keyword_snippets = _find_keyword_snippets(full_rules_text, phrases)

    # If we found strong structure-aware matches (especially points blocks),
    # put them FIRST so they don't get drowned out by other retrieved context.
    prefer_points = "point" in question.lower() or "points" in question.lower() or "pts" in question.lower()
    if prefer_points and any("**Points:**" in s for s in keyword_snippets):
        # We already have the authoritative unit block; keep embedding context small.
        context_snippets = context_snippets[:3]

    all_snippets: List[str] = keyword_snippets + context_snippets
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
            "Ask Warhammer rules questions (AoS or 40K) using "
            "LangChain embeddings + FAISS + OpenAI."
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
            "Build (or rebuild) the FAISS index from the structured markdown "
            "rules files for the selected game."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help=(
            "Override the default rules markdown directory for the selected game. "
            "If omitted, a sensible default per game is used (Data/AOS-Datafiles or Data/40K-Datafiles)."
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
        default_data = DEFAULT_AOS_DATA_DIR
        default_index = DEFAULT_AOS_INDEX_DIR
        game_label = "Warhammer Age of Sigmar"
    else:
        default_data = DEFAULT_WH40K_DATA_DIR
        default_index = DEFAULT_WH40K_INDEX_DIR
        game_label = "Warhammer 40,000"

    data_dir = args.data_dir or default_data
    index_dir = args.index_dir or default_index

    # If requested, (re)build the index from the specified markdown.
    if args.build_index:
        build_index(data_dir=data_dir, index_dir=index_dir, game=args.game)
        # If the user only wanted to build the index, we can exit early.
        if not args.question:
            return 0

    # Load FAISS index for question answering
    vectorstore = load_index(index_dir=index_dir)

    # Load the full rules corpus once so we can do lightweight keyword-based
    # lookups to augment retrieval without extra network/token cost.
    full_rules_text = load_rules_corpus_text(data_dir=data_dir) or None

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

