import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
TEMPERATURE = 0.1
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a precise 3GPP Telecom Standards assistant.
Answer ONLY using the provided context from 3GPP specification documents.
Rules:
- Output the final answer only. No thinking, no reasoning steps, no analysis.
- If the context does not contain the answer, respond exactly:
  "I don't have enough information in the provided 3GPP documents to answer this question."
- Always cite the source document (e.g., "According to 23501-k20...").
- Never fabricate technical specifications, numbers, or procedures.
- Be concise and technically accurate."""


def clean_answer(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return f"Query: {query}\nContext: [No relevant context found]"
    context_str = "\n\n".join(
        f"[Source {i} - {chunk['metadata'].get('source', 'Unknown')}]:\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks, 1)
    )
    return f"Context from 3GPP Documents:\n{context_str}\n\nQuestion: {query}"


class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )

    def generate(self, query: str, context_chunks: list[dict]) -> dict:
        prompt = build_prompt(query, context_chunks)
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        return {
            "answer": clean_answer(response.choices[0].message.content),
            "model": MODEL,
            "sources": [c["metadata"].get("source", "unknown") for c in context_chunks]
        }