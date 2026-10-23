"""Grounded retrieval and extractive answer generation."""
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from .ingest import build_vector_store

STOP_WORDS = {"how", "many", "what", "is", "the", "a", "an", "of", "for", "does", "do", "company", "get", "give", "given", "per", "year", "days", "rate", "number", "amount", "period", "duration", "current", "in", "india", "are", "to", "and", "from", "date", "their", "employee", "employees", "leave", "manager", "up"}

@dataclass
class QAResult:
    answer: str
    sources: list[dict[str, Any]]


def _words(text: str) -> set[str]:
    words = set()
    for token in re.findall(r"[A-Za-z]{2,}", text):
        word = token.lower()
        if word == "sr":
            word = "senior"
        if word not in STOP_WORDS:
            words.add(word)
    return words


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\bSr\.", "Sr<ABBR>", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    parts = re.split(r"(?<=[.!?])\s+|\s*[•€]\s*|\s*&\s*", cleaned)
    return [s.strip(" •-&€").replace("Sr<ABBR>", "Sr.") for s in parts if len(s.strip()) > 8]


def _query_acronyms(question: str) -> set[str]:
    """Find likely abbreviations for multiword topics, such as maternity leave -> ML."""
    ordered_words = [
        word.upper()
        for word in re.findall(r"[A-Za-z]+", question)
        if word.lower() not in (STOP_WORDS - {"leave", "manager"}) and len(word) > 2
    ]
    acronyms = {
        "".join(word[0] for word in pair)
        for pair in zip(ordered_words, ordered_words[1:])
        if 2 <= len(pair) <= 5
    }
    # Preserve abbreviations explicitly supplied in a question, such as (PL).
    acronyms.update(re.findall(r"\b[A-Z]{2,5}\b", question))
    return acronyms


def _clean_text(text: str) -> str:
    """Repair common text-extraction artifacts without changing policy facts."""
    text = text.replace("/T_he", "The ").replace("/t_he", "the ")
    text = text.replace("/f_i", "fi").replace("/f_", "f")
    text = text.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\(", " (", text)
    text = re.sub(r"^[>•\s]+", "", text)
    text = re.sub(r"\(\s*-\s*", "(", text)
    text = re.sub(r"\s+\(", " (", text)
    text = re.sub(r"(\d)\s*(days?|weeks?|months?|years?)\b", r"\1 \2", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_from_documents(payload: dict[str, Any]) -> QAResult:
    """Select only retrieved evidence and cite every passage used."""
    question = payload["question"]
    documents = payload.get("documents", [])
    query_terms = _words(question)
    query_acronyms = _query_acronyms(question)
    scored = []
    for doc, distance in documents:
        content_terms = _words(doc.page_content)
        overlap = query_terms & content_terms
        if overlap:
            scored.append((doc, distance, overlap))
    if not scored:
        return QAResult("Not found", [])

    best_doc, best_distance, best_overlap = scored[0]
    if len(best_overlap) / max(1, len(query_terms)) < 0.65:
        return QAResult("Not found", [])

    asks_for_amount = bool(re.search(r"how many|how much|amount|number|rate|period|duration", question, re.I))
    evidence = []
    seen = set()
    for doc, distance, overlap in scored:
        for sentence in _sentences(doc.page_content):
            terms = _words(sentence)
            coverage = len(query_terms & terms) / max(1, len(query_terms))
            has_topic_abbreviation = any(
                re.search(rf"\b{re.escape(abbreviation)}\b", sentence)
                for abbreviation in query_acronyms
            )
            has_measured_value = bool(re.search(
                r"\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|%|percent|rupees|rs\.?|₹)",
                sentence, re.I,
            ))
            is_relevant = coverage >= 0.65 or (asks_for_amount and has_topic_abbreviation and has_measured_value)
            if is_relevant:
                normalized = sentence.casefold()
                if normalized not in seen:
                    seen.add(normalized)
                    # Direct topic mentions sort before adjacent abbreviation/value details.
                    priority = coverage + (0.01 if has_topic_abbreviation and has_measured_value else 0)
                    if has_measured_value and re.search(r"\b(?:total|to-\s*tal)\b", sentence, re.I):
                        priority += 1.0
                    evidence.append((priority, doc, sentence, coverage, has_topic_abbreviation, has_measured_value))

    evidence.sort(key=lambda item: item[0], reverse=True)
    if not evidence:
        return QAResult("Not found", [])

    amount_evidence = [item for item in evidence if item[5] and (item[3] >= 0.65 or item[4])]
    if asks_for_amount and amount_evidence:
        # Abbreviations derived from the query connect table rows where the heading appears
        # separately from the numeric entitlement text.
        answer = _clean_text(amount_evidence[0][2])
        chosen = amount_evidence[:1]
    elif asks_for_amount:
        chosen = evidence[:3]
        topic = _clean_text(chosen[0][2]).rstrip(".")
        kind = "number of days" if "how many" in question.lower() else "definite amount or duration"
        answer = f'The manual mentions "{topic}" but does not specify a {kind}.'
    else:
        chosen = evidence[:3]
        answer = _clean_text(" ".join(item[2] for item in chosen))

    refs = []
    for _, doc, _, _, _, _ in chosen:
        md = doc.metadata
        ref = {"policy": md.get("policy", ""), "annexure": md.get("annexure", ""),
               "page_number": md.get("page_number", md.get("page", 0) + 1)}
        if ref not in refs:
            refs.append(ref)
    return QAResult(answer, refs)


class HRPolicyQA:
    """LangChain retrieval chain backed by the persistent Chroma collection."""
    def __init__(self):
        self.vector_store = build_vector_store()
        self.chain = RunnableLambda(self._retrieve).with_config(run_name="hr_policy_retrieval") | RunnableLambda(answer_from_documents)

    def _retrieve(self, inputs):
        question = inputs["question"]
        vector_results = self.vector_store.similarity_search_with_score(question, k=6)

        # Add indexed chunks with strong exact-term overlap. This helps with short policy
        # labels and acronyms that a semantic top-k search can rank below nearby content.
        query_terms = _words(question)
        lexical_results = []
        seen = set()
        collection_data = self.vector_store._collection.get(include=["documents", "metadatas"])
        for text, metadata in zip(collection_data["documents"], collection_data["metadatas"]):
            overlap = query_terms & _words(text)
            coverage = len(overlap) / max(1, len(query_terms))
            if coverage >= 0.65:
                key = (metadata.get("page_number"), text)
                seen.add(key)
                lexical_results.append((coverage, Document(page_content=text, metadata=metadata)))
        lexical_results.sort(key=lambda item: item[0], reverse=True)

        combined = [(doc, 0.0) for _, doc in lexical_results]
        for doc, distance in vector_results:
            key = (doc.metadata.get("page_number"), doc.page_content)
            if key not in seen:
                combined.append((doc, distance))
                seen.add(key)
        return {"question": question, "documents": combined}

    def ask(self, question: str) -> QAResult:
        return self.chain.invoke({"question": question})
