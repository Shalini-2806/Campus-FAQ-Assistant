"""FAQ data validation and local question matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = {"question", "answer"}
STOP_WORDS = {
    "a", "an", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "of", "on", "or",
    "please", "tell", "the", "to", "was", "we", "what", "when", "where",
    "which", "who", "why", "with", "would", "you",
}
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
STEMMER = PorterStemmer()


class FAQValidationError(ValueError):
    """Raised when an FAQ file does not have a usable schema or content."""


def preprocess(text: str) -> list[str]:
    """Normalize text into useful English word stems without downloading corpora."""
    words = TOKEN_PATTERN.findall(str(text).lower())
    return [
        STEMMER.stem(word)
        for word in words
        if word not in STOP_WORDS
    ]


def load_faqs(source: str | Path | IO[Any]) -> pd.DataFrame:
    """Load and validate FAQ rows from a CSV path or file-like object."""
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        frame = pd.read_csv(source, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise FAQValidationError(f"Invalid CSV: {error}") from error

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise FAQValidationError(
            f"Missing required column(s): {', '.join(sorted(missing))}."
        )

    if "alternate_questions" not in frame.columns:
        frame["alternate_questions"] = ""
    frame = frame[["question", "answer", "alternate_questions"]].copy()
    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: str(value).strip())

    frame = frame[
        frame["question"].ne("") & frame["answer"].ne("")
    ].reset_index(drop=True)
    if frame.empty:
        raise FAQValidationError("The CSV must contain at least one non-empty FAQ.")

    duplicates = frame["question"].str.casefold().duplicated()
    if duplicates.any():
        raise FAQValidationError("Each FAQ question must be unique.")

    return frame


@dataclass(frozen=True)
class MatchResult:
    answer: str
    question: str
    score: float
    matched: bool


class FAQMatcher:
    """Rank FAQs by a blend of word and character TF-IDF similarity."""

    def __init__(self, faqs: list[dict[str, str]]) -> None:
        if not faqs:
            raise FAQValidationError("At least one FAQ is required.")
        self.faqs = faqs
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=preprocess,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
        )

        self._question_indices: list[int] = []
        training_texts: list[str] = []
        for index, faq in enumerate(faqs):
            variations = [faq["question"]]
            variations.extend(
                item.strip()
                for item in faq.get("alternate_questions", "").split("|")
                if item.strip()
            )
            for variation in variations:
                self._question_indices.append(index)
                training_texts.append(variation)

        if not any(preprocess(text) for text in training_texts):
            raise FAQValidationError(
                "FAQ questions must include at least one searchable word."
            )

        self.word_vectors = self.word_vectorizer.fit_transform(training_texts)
        self.char_vectors = self.char_vectorizer.fit_transform(training_texts)

    def match(self, user_question: str, threshold: float = 0.28) -> MatchResult:
        """Return the closest answer, or a safe fallback below the threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if not isinstance(user_question, str) or not user_question.strip():
            return MatchResult(
                "Please enter a question so I can find the closest FAQ.",
                "",
                0.0,
                False,
            )

        word_query = self.word_vectorizer.transform([user_question])
        char_query = self.char_vectorizer.transform([user_question])
        word_scores = cosine_similarity(word_query, self.word_vectors)[0]
        char_scores = cosine_similarity(char_query, self.char_vectors)[0]
        if not word_scores.any():
            return MatchResult(
                "I couldn't find a confident match in the FAQs. "
                "Try rephrasing your question or contact the college office.",
                "",
                0.0,
                False,
            )
        variant_scores = 0.8 * word_scores + 0.2 * char_scores

        faq_scores = [0.0] * len(self.faqs)
        for variant_index, faq_index in enumerate(self._question_indices):
            faq_scores[faq_index] = max(
                faq_scores[faq_index], float(variant_scores[variant_index])
            )

        best_index = max(range(len(faq_scores)), key=faq_scores.__getitem__)
        best_score = faq_scores[best_index]
        faq = self.faqs[best_index]
        matched = best_score >= threshold
        if matched:
            answer = faq["answer"]
        else:
            answer = (
                "I couldn't find a confident match in the FAQs. "
                "Try rephrasing your question or contact the college office."
            )

        return MatchResult(answer, faq["question"], best_score, matched)
