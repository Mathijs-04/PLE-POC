import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI


def main():
    # Load variables from .env into environment
    load_dotenv()  # expects OPENAI_API_KEY in .env

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file as OPENAI_API_KEY=your_key_here."
        )

    parser = argparse.ArgumentParser(
        description="Send a simple question to OpenAI gpt-4o-mini."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="The question/prompt to send to the model.",
    )
    args = parser.parse_args()

    question = args.question
    if not question:
        # Fallback to interactive input if nothing passed on the command line
        question = input("Enter your question: ").strip()
        if not question:
            raise SystemExit("No question provided.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=question,
    )

    # The Responses API returns a structured object; extract the text content.
    # This assumes a simple text response.
    try:
        output_text = response.output[0].content[0].text
    except (AttributeError, IndexError, KeyError):
        output_text = str(response)

    print("\n=== Model response ===\n")
    print(output_text)


if __name__ == "__main__":
    main()