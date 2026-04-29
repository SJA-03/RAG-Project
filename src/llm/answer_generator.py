import os


DEFAULT_MODEL = "gpt-5.4-mini"


def generate_answer(query: str, contexts: list[dict]) -> str:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if not contexts:
        raise ValueError("Contexts must not be empty")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError("OpenAI Python SDK is not installed. Install dependencies with `pip install -r requirements.txt`.") from error

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(query, contexts)

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    if getattr(response, "output_text", None):
        return response.output_text.strip()

    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue

        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "output_text" and getattr(content, "text", None):
                return content.text.strip()

    raise ValueError("OpenAI API returned no text output")


def _build_prompt(query: str, contexts: list[dict]) -> str:
    formatted_contexts = []
    for index, context in enumerate(contexts, start=1):
        chunk_id = context.get("chunk_id", "unknown")
        text = str(context.get("text", "")).strip()
        formatted_contexts.append(f"[Context {index}] chunk_id={chunk_id}\n{text}")

    context_block = "\n\n".join(formatted_contexts)

    return (
        "You are a retrieval-grounded academic assistant.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved contexts:\n{context_block}\n\n"
        "Instructions:\n"
        "1. Answer using only the retrieved contexts above.\n"
        "2. If the contexts are insufficient, say you do not know based on the provided context.\n"
        "3. Do not add unsupported facts.\n"
        "4. Provide a concise answer grounded in the context.\n"
    )
