"""RAG engine: prompt building and evidence formatting."""
from typing import List


def build_rag_prompt(question: str, evidence: List[dict]) -> str:
    """Build prompt with retrieval evidence."""
    prompt = f"Answer concisely based on the image: {question}"
    if evidence:
        refs = "\n".join(
            f"Similar case (sim={e['sim']:.2f}): Q=\"{e['question']}\" A=\"{e['answer']}\""
            for e in evidence[:3]
        )
        prompt += f"\n\nReference similar cases:\n{refs}"
    return prompt


def format_evidence(sim: float, question: str, answer: str) -> str:
    """Format a single evidence entry."""
    return f"Similar case (sim={sim:.2f}): Q=\"{question}\" A=\"{answer}\""
