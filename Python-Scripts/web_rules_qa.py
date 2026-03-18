import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rules_qa import (
    EXAMPLE_SYSTEM_PROMPT,
    DEFAULT_AOS_DATA_DIR,
    DEFAULT_AOS_INDEX_DIR,
    DEFAULT_WH40K_DATA_DIR,
    DEFAULT_WH40K_INDEX_DIR,
    answer_question,
    load_index,
    load_rules_corpus_text,
    load_rules_sources,
)


load_dotenv()  # for OPENAI_API_KEY and friends

app = FastAPI(title="Warhammer Rules Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    game: Literal["aos", "wh40k"]
    question: str


class AskResponse(BaseModel):
    answer: str


_VECTORSTORES = {}
_FULL_TEXT: dict[str, str] = {}
_SOURCES = {}


@app.on_event("startup")
def startup() -> None:
    """
    Load FAISS indexes and full rules text once on startup so that
    each API call is fast and cheap.
    """
    # Load FAISS indexes from the existing Data directories.
    _VECTORSTORES["aos"] = load_index(index_dir=DEFAULT_AOS_INDEX_DIR)
    _VECTORSTORES["wh40k"] = load_index(index_dir=DEFAULT_WH40K_INDEX_DIR)

    # Load full corpus text for lightweight keyword snippets.
    _FULL_TEXT["aos"] = load_rules_corpus_text(DEFAULT_AOS_DATA_DIR) or ""
    _FULL_TEXT["wh40k"] = load_rules_corpus_text(DEFAULT_WH40K_DATA_DIR) or ""

    # Load per-file sources for robust structure-aware keyword lookup.
    _SOURCES["aos"] = load_rules_sources(DEFAULT_AOS_DATA_DIR)
    _SOURCES["wh40k"] = load_rules_sources(DEFAULT_WH40K_DATA_DIR)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    game = req.game
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if game not in _VECTORSTORES:
        raise HTTPException(status_code=400, detail=f"Unsupported game: {game}")

    vectorstore = _VECTORSTORES[game]
    full_text = _FULL_TEXT.get(game) or ""

    game_label = "Warhammer Age of Sigmar" if game == "aos" else "Warhammer 40,000"

    answer = answer_question(
        question=question,
        system_prompt=EXAMPLE_SYSTEM_PROMPT,
        vectorstore=vectorstore,
        game_label=game_label,
        model_name="gpt-5.4",
        full_rules_text=full_text,
        full_rules_sources=_SOURCES.get(game),
    )

    return AskResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

