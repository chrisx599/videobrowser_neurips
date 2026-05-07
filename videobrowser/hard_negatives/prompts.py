from __future__ import annotations
import re
from typing import Optional, Literal

Verdict = Literal["YES", "NO", "UNCERTAIN"]
Relevance = Literal["HIGH", "MID", "LOW"]


_LAYER1_TEMPLATE = """You are evaluating a YouTube video against a benchmark question.

Question:
{question}

The exact correct answer is: {answer}

Candidate video metadata:
- Title: {title}
- Description: {description}

Transcript excerpt (auto-captioned, may be partial):
{transcript_block}

Make TWO independent judgements.

(1) VERDICT — does this video contain THIS SPECIFIC answer?
- YES: the video almost certainly states / shows the EXACT answer ({answer}). Same value, same name, same string.
- NO: the video does not appear to contain that exact answer. (A video on the same topic with a different value is NO, not YES.)
- UNCERTAIN: cannot tell from metadata + transcript alone.

(2) RELEVANCE — how topically related is the video to the question, regardless of whether it contains the answer?
- HIGH: same narrow subject — would be a plausible candidate a reader could mistake for the source.
- MID: same broad topic / domain, but a different specific subject.
- LOW: unrelated, off-topic, or an obvious mismatch (e.g., random viral video, music clip, unrelated category).

Respond in this exact format:
VERDICT: <YES|NO|UNCERTAIN>
RELEVANCE: <HIGH|MID|LOW>
Reason: <one short sentence>
"""


_LAYER2_TEMPLATE = """You are evaluating a YouTube video against a benchmark question.
You will see {n_frames} sparse frames sampled across the full video AND the transcript below.

Question:
{question}

The exact correct answer is: {answer}

Candidate video metadata:
- Title: {title}
- Description: {description}

Transcript excerpt:
{transcript_block}

Use BOTH the visual frames and the text. Make TWO independent judgements.

(1) VERDICT — does this video contain THIS SPECIFIC answer?
- YES: visual or transcript evidence shows the EXACT answer ({answer}) — same value, same name, same string.
- NO: the answer is not visibly present in the frames or stated in the transcript. (Same topic with a different value is NO.)
- UNCERTAIN: ambiguous evidence.

(2) RELEVANCE — how topically related is the video to the question?
- HIGH: same narrow subject — would be a plausible candidate a reader could mistake for the source.
- MID: same broad topic / domain, but a different specific subject.
- LOW: unrelated or off-topic (random viral video, music clip, wrong category, etc.).

Respond in this exact format:
VERDICT: <YES|NO|UNCERTAIN>
RELEVANCE: <HIGH|MID|LOW>
Reason: <one short sentence>
"""


def _transcript_block(excerpt: Optional[str]) -> str:
    if excerpt is None or not excerpt.strip():
        return "(transcript not available)"
    return excerpt.strip()


def build_layer1_prompt(
    *,
    question: str,
    answer: str,
    title: str,
    description: str,
    transcript_excerpt: Optional[str],
) -> str:
    return _LAYER1_TEMPLATE.format(
        question=question,
        answer=answer,
        title=title,
        description=description,
        transcript_block=_transcript_block(transcript_excerpt),
    )


def build_layer2_text_prompt(
    *,
    question: str,
    answer: str,
    title: str,
    description: str,
    transcript_excerpt: Optional[str],
    n_frames: int = 16,
) -> str:
    return _LAYER2_TEMPLATE.format(
        question=question,
        answer=answer,
        title=title,
        description=description,
        transcript_block=_transcript_block(transcript_excerpt),
        n_frames=n_frames,
    )


_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(YES|NO|UNCERTAIN)\b", re.IGNORECASE)
_VERDICT_RE_FALLBACK = re.compile(r"\b(YES|NO|UNCERTAIN)\b", re.IGNORECASE)
_RELEVANCE_RE = re.compile(r"RELEVANCE\s*:\s*(HIGH|MID|LOW)\b", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[Verdict, str]:
    """Extract verdict + short reason from a model response.

    Falls back to UNCERTAIN on malformed responses.
    """
    if not text:
        return "UNCERTAIN", ""
    m = _VERDICT_RE.search(text) or _VERDICT_RE_FALLBACK.search(text)
    verdict: Verdict = m.group(1).upper() if m else "UNCERTAIN"
    reason = ""
    lower = text.lower()
    idx = lower.find("reason:")
    if idx >= 0:
        reason = text[idx + len("reason:"):].strip().splitlines()[0].strip()
    else:
        post = text[m.end():].strip() if m else text.strip()
        reason = post.split("\n", 1)[0].lstrip(":—- ").strip()
    return verdict, reason[:200]


def parse_relevance(text: str) -> Optional[Relevance]:
    """Extract HIGH/MID/LOW relevance from a model response, or None if absent."""
    if not text:
        return None
    m = _RELEVANCE_RE.search(text)
    if m is None:
        return None
    return m.group(1).upper()  # type: ignore[return-value]
