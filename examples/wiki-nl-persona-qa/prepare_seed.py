"""Build the seed for the persona-grounded Dutch Wikipedia QA pipeline.

Samples clean Dutch Wikipedia articles and pairs each one with a random
Belgian-Dutch persona, a random question type and a random question/answer
length. All of that is what stops half a million generated questions from
sounding like the same person asking the same kind of question, at the same
length, about every article.

```sh
uv run examples/wiki-nl-persona-qa/prepare_seed.py \
    --num-samples 50000 --num-proc 16 \
    --out examples/wiki-nl-persona-qa/outputs/seed
```
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from spacy.language import Language


QUESTION_TYPES = {
    "feit": (
        "een vraag naar een concreet feit: een getal, datum, naam, plaats of"
        " hoeveelheid uit het artikel"
    ),
    "definitie": (
        "een vraag naar wat iets is of betekent, waarop het antwoord een"
        " omschrijving is"
    ),
    "uitleg": (
        "een waarom- of hoe-vraag, waarop het antwoord een mechanisme, reden"
        " of werkwijze uitlegt"
    ),
    "chronologie": (
        "een vraag naar de volgorde of het verloop van gebeurtenissen in de"
        " tijd"
    ),
    "vergelijking": (
        "een vraag die twee zaken uit het artikel tegenover elkaar zet, of"
        " vraagt waarin iets verschilt van iets anders"
    ),
    "gevolg": (
        "een vraag naar het gevolg, het effect of het belang van iets dat in"
        " het artikel beschreven wordt"
    ),
}
"""Question types to spread over the seed, as the Dutch instruction the prompt
renders into `{question_type}`. The value is written into the dataset, not the
key, so the prompt needs one placeholder rather than a lookup table."""

QUESTION_LENGTHS = {
    "kort": "een zin, direct en zonder inleiding of aanloop",
    "uitgebreid": (
        "een paar zinnen: schets eerst kort een aanleiding waarom deze"
        " persoon dit zou willen weten (bijvoorbeeld uit interesse, voor het"
        " werk, voor een project of hobby, omdat een kind of kleinkind ernaar"
        " vroeg, of iets vergelijkbaars), en stel dan de eigenlijke vraag."
        " Verzin een aanleiding die bij de persoon past, zonder hun beroep,"
        " naam, woonplaats of leeftijd letterlijk te noemen"
    ),
}
"""Question lengths to spread over the seed, rendered the same way as
[`QUESTION_TYPES`] into `{question_length}`."""

ANSWER_LENGTHS = {
    "kort": (
        "één alinea die meteen met de kern begint; geen losse feitjes die de"
        " vraag niet beantwoorden"
    ),
    "uitgebreid": (
        "twee tot vier alinea's: geef de kern meteen in de eerste alinea en"
        " werk daarna de achtergrond, oorzaken of het verloop verder uit,"
        " zodat het antwoord de vraag helemaal doelgericht beantwoordt"
        " met de context die relevant is voor de vraagsteller"
    ),
}
"""Answer lengths, rendered into `{answer_length}` in the answer prompt. Drawn
per row with weights that favour "uitgebreid" when `question_type` is one of
[`DEEP_QUESTION_TYPES`], so answer depth tracks what the question actually
needs instead of varying independently of it."""

DEEP_QUESTION_TYPES = {"uitleg", "chronologie", "vergelijking", "gevolg"}
"""Question types whose answer more often needs more than one paragraph."""


def keep_articles(texts: Sequence[str], *, min_words: int) -> list[bool]:
    """Apply the quality filter to a batch of articles.

    Naive checking of how many sentences (`. `) >= 3 and a word count above
    min_words. There is no upper bound here: [`pack_sections`] guarantees
    every chunk it produces already respects max_words by construction.

    Args:
        texts: Rendered article or chunk texts.
        min_words: Shortest text to keep, exclusive.

    Returns:
        One keep/drop flag per text, in input order.
    """
    return [
        text.count(". ") + int(text.endswith(".")) >= 3
        and len(text.split()) > min_words
        for text in texts
    ]


SECTION_MARKER = re.compile(r"(?m)^(?=## )")
"""Matches the start of a markdown line opening a top-level (`## `) section,
so [`split_sections`] splits on main sections without also breaking off
subsections (`### `, ...) into their own chunk."""


def split_sections(text: str) -> list[str]:
    """Split a markdown article on its top-level (`## `) section headings.

    Args:
        text: Rendered article markdown.

    Returns:
        The article's sections in order, each a heading plus its body. Text
        before the first heading is kept as its own leading section.

    Examples:
        >>> split_sections("Inleiding.\\n\\n## Kop\\nBody.")
        ['Inleiding.\\n\\n', '## Kop\\nBody.']
    """
    return [
        section for section in SECTION_MARKER.split(text) if section.strip()
    ]


def keep_section_sizes(
    texts: Sequence[str], *, max_section_chars: int
) -> list[bool]:
    """Drop an article if any of its sections is absurdly long.

    Some finewiki articles have a `## ` section running into the millions of
    characters: leftover markup, a raw table dump, or some other non-prose
    artifact rather than an actual long section, and well past what
    [`truncate_to_sentences`]'s spaCy pipeline can even accept (its
    `nlp.max_length` default is 1,000,000 characters). max_section_chars is
    set far above any real section on purpose, so this only catches that
    kind of malformed content, not genuinely long prose, which
    [`pack_sections`] already handles by truncating. The whole article is
    dropped rather than just that section, since a section this broken is a
    sign the rest of the article's markup is not trustworthy either.

    Args:
        texts: Rendered article markdown.
        max_section_chars: Longest section to tolerate, in characters.

    Returns:
        One keep/drop flag per article, in input order.
    """
    return [
        all(
            len(section) <= max_section_chars
            for section in split_sections(text)
        )
        for text in texts
    ]


@lru_cache(maxsize=1)
def _sentencizer() -> Language:
    """Build a blank Dutch spaCy pipeline with only a rule-based sentencizer.

    Cached per process so a batch with several oversized sections does not
    rebuild the pipeline for each one. No model download is needed: sentence
    boundaries come from punctuation rules alone.
    """
    import spacy

    nlp = spacy.blank("nl")
    nlp.add_pipe("sentencizer")
    return nlp


def truncate_to_sentences(text: str, *, max_words: int) -> str:
    """Cut text to as many complete sentences as fit within max_words.

    Only used on a single section that alone exceeds max_words, since an
    ordinary chunk is built from whole sections and never needs cutting
    mid-section. Everything past the cut is discarded.

    Args:
        text: The oversized section, heading included.
        max_words: The word budget to fit within.

    Returns:
        A prefix of text ending at a sentence boundary. At least one
        sentence is always kept, even one that alone exceeds max_words,
        since a sentence cannot be split any further.

    Examples:
        >>> truncate_to_sentences(
        ...     "Een zin. Nog een zin. En een derde.", max_words=5
        ... )
        'Een zin. Nog een zin.'
    """
    sentence_ends = [sent.end_char for sent in _sentencizer()(text).sents]
    if not sentence_ends:
        return text

    prefix = text[: sentence_ends[0]]
    for end_char in sentence_ends[1:]:
        candidate = text[:end_char]
        if len(candidate.split()) > max_words:
            break
        prefix = candidate
    return prefix


def pack_sections(sections: Sequence[str], *, max_words: int) -> list[str]:
    """Glue consecutive whole sections into chunks that fit within max_words.

    First-fit, in order, no overlap: a chunk keeps absorbing whole sections
    until the next one would push it over max_words, at which point it is
    closed and a new chunk starts with that section. A section that alone
    exceeds max_words can never be a whole-section chunk, so it is cut to
    the last complete sentence that fits with [`truncate_to_sentences`]
    instead, and whatever is left of it is dropped.

    Args:
        sections: The article's sections in order, as returned by
            [`split_sections`].
        max_words: Longest chunk to keep.

    Returns:
        The article's chunks in order, each at most max_words words: either
        the concatenation of one or more whole sections, or one section
        truncated to fit.

    Examples:
        >>> pack_sections(
        ...     ["## A\\n" + "w " * 4, "## B\\n" + "w " * 4], max_words=15
        ... )
        ['## A\\nw w w w ## B\\nw w w w ']
        >>> pack_sections(
        ...     ["## A\\n" + "w " * 4, "## B\\n" + "w " * 4], max_words=6
        ... )
        ['## A\\nw w w w ', '## B\\nw w w w ']
    """
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for section in sections:
        words = len(section.split())
        if words > max_words:
            if current:
                chunks.append("".join(current))
                current, current_words = [], 0
            chunks.append(truncate_to_sentences(section, max_words=max_words))
            continue
        if current and current_words + words > max_words:
            chunks.append("".join(current))
            current, current_words = [], 0
        current.append(section)
        current_words += words

    if current:
        chunks.append("".join(current))

    return chunks


def keep_running_text(
    texts: Sequence[str],
    *,
    min_avg_sentence_words: float,
    max_avg_sentence_words: float,
) -> list[bool]:
    """Drop chunks that read as tables or enumerations rather than prose.

    A markdown table or bullet list rarely carries normal sentence-ending
    punctuation, so the rule-based sentencizer either reads the whole thing
    as one giant "sentence" (average words/sentence far above
    max_avg_sentence_words) or, for short list items that each end in a
    period, as a run of short fragments (average far below
    min_avg_sentence_words). Running prose lands in between.

    Args:
        texts: Chunk texts, already within max_words.
        min_avg_sentence_words: Lowest acceptable average sentence length,
            in words.
        max_avg_sentence_words: Highest acceptable average sentence length,
            in words.

    Returns:
        One keep/drop flag per chunk, in input order.
    """
    keep = []
    for doc in _sentencizer().pipe(texts):
        lengths = [len(sent.text.split()) for sent in doc.sents]
        avg = sum(lengths) / len(lengths) if lengths else 0
        keep.append(min_avg_sentence_words <= avg <= max_avg_sentence_words)
    return keep


def split_articles(
    batch: dict[str, list], *, max_words: int
) -> dict[str, list]:
    """Split each article into chunks that fit max_words.

    An article within max_words passes through untouched, as a single
    chunk. A longer one is split on `## ` sections with [`pack_sections`],
    so what used to be a dropped article becomes several full-context
    samples instead of one truncated one.

    Args:
        batch: One column-major batch, as `Dataset.map` passes it.
        max_words: Longest chunk to keep; forwarded to [`pack_sections`].

    Returns:
        The same columns, expanded to one row per chunk, plus `chunk_index`
        and `num_chunks` for provenance.
    """
    out: dict[str, list] = {key: [] for key in batch}
    out["chunk_index"] = []
    out["num_chunks"] = []

    for i, text in enumerate(batch["text"]):
        if len(text.split()) <= max_words:
            chunks = [text]
        else:
            chunks = pack_sections(split_sections(text), max_words=max_words)
        for chunk_index, chunk in enumerate(chunks):
            for key, values in batch.items():
                out[key].append(chunk if key == "text" else values[i])
            out["chunk_index"].append(chunk_index)
            out["num_chunks"].append(len(chunks))

    return out


def build_seed(
    *,
    num_samples: int,
    seed: int,
    min_words: int,
    max_words: int,
    max_section_chars: int,
    min_avg_sentence_words: float,
    max_avg_sentence_words: float,
    num_proc: int | None,
    output_dir: str,
    hub_id: str | None,
) -> None:
    """Sample finewiki-nl articles and attach a persona, a question type and
    a question/answer length.

    An article longer than max_words is split into chunks along its `## `
    section boundaries with [`split_articles`] rather than dropped, so a
    long article contributes several full-context samples instead of none.
    Two quality gates run alongside that: [`keep_section_sizes`] drops an
    article outright if one of its sections is not just long but absurdly
    so, and [`keep_running_text`] drops a chunk that reads as a table or
    enumeration rather than prose.

    Personas are drawn with replacement, so any `num_samples` works even though
    the persona pool holds 300k rows; the same persona then asks about several
    different articles, which is what the variability is for.

    Args:
        num_samples: Chunks to keep.
        seed: Random seed for the shuffle and every draw (persona, question
            type, question length, answer length).
        min_words: Shortest chunk to keep, exclusive.
        max_words: Longest chunk to keep; articles above this are split into
            several chunks instead of dropped.
        max_section_chars: Longest section to tolerate, in characters,
            before the whole article is dropped as malformed.
        min_avg_sentence_words: Lowest acceptable average sentence length in
            a chunk, in words.
        max_avg_sentence_words: Highest acceptable average sentence length
            in a chunk, in words.
        num_proc: Processes to load and filter with, or ``None`` for one.
        output_dir: Directory to write the seed dataset to.
        hub_id: Hub dataset id to push to, if any.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/finewiki",
        "nl",
        num_proc=num_proc,
        split="train",
    )

    ds = ds.filter(
        keep_articles,
        input_columns=["text"],
        batched=True,
        fn_kwargs={"min_words": min_words},
        num_proc=num_proc,
    )
    print(f"{len(ds):,} articles pass the minimum-length filter")

    ds = ds.filter(
        keep_section_sizes,
        input_columns=["text"],
        batched=True,
        fn_kwargs={"max_section_chars": max_section_chars},
        num_proc=num_proc,
    )
    print(
        f"{len(ds):,} articles pass the section-size sanity check"
        f" (no section over {max_section_chars:,} characters)"
    )

    ds = ds.map(
        split_articles,
        batched=True,
        fn_kwargs={"max_words": max_words},
        num_proc=num_proc,
        remove_columns=ds.column_names,
    )
    ds = ds.filter(
        keep_articles,
        input_columns=["text"],
        batched=True,
        fn_kwargs={"min_words": min_words},
        num_proc=num_proc,
    )
    print(f"{len(ds):,} chunks after splitting oversized articles")
    print("Chunks per article:", Counter(ds["num_chunks"]).most_common())

    ds = ds.filter(
        keep_running_text,
        input_columns=["text"],
        batched=True,
        fn_kwargs={
            "min_avg_sentence_words": min_avg_sentence_words,
            "max_avg_sentence_words": max_avg_sentence_words,
        },
        num_proc=num_proc,
    )
    print(
        f"{len(ds):,} chunks pass the running-text filter"
        f" ({min_avg_sentence_words}-{max_avg_sentence_words}"
        " words/sentence)"
    )

    ds = ds.shuffle(seed=seed).select_columns(
        ["title", "text", "url", "chunk_index", "num_chunks"]
    )

    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
        print(f"Kept {len(ds):,} chunks after sampling")

    personas = load_dataset(
        "nvidia/Nemotron-Personas-Belgium",
        split="nl_BE",
        num_proc=num_proc,
    ).select_columns(["persona"])
    print(f"{len(personas):,} Belgian-Dutch personas to draw from")

    rng = random.Random(seed)
    drawn = personas.select(
        [rng.randrange(len(personas)) for _ in range(len(ds))]
    )["persona"]

    type_keys = rng.choices(list(QUESTION_TYPES), k=len(ds))
    types = [QUESTION_TYPES[key] for key in type_keys]

    length_keys = rng.choices(
        list(QUESTION_LENGTHS), weights=[0.6, 0.4], k=len(ds)
    )
    lengths = [QUESTION_LENGTHS[key] for key in length_keys]

    answer_length_keys = [
        rng.choices(
            list(ANSWER_LENGTHS),
            weights=(0.3, 0.7) if key in DEEP_QUESTION_TYPES else (0.85, 0.15),
        )[0]
        for key in type_keys
    ]
    answer_lengths = [ANSWER_LENGTHS[key] for key in answer_length_keys]

    ds = (
        ds.add_column("persona", drawn)
        .add_column("question_type", types)
        .add_column("question_length", lengths)
        .add_column("answer_length", answer_lengths)
    )
    print("Question types:", Counter(type_keys).most_common())
    print("Question lengths:", Counter(length_keys).most_common())
    print("Answer lengths:", Counter(answer_length_keys).most_common())
    print(ds)

    ds.save_to_disk(output_dir)
    print(f"Wrote {len(ds):,} chunks to {output_dir}")

    if hub_id:
        ds.push_to_hub(hub_id)
        print(f"Pushed to {hub_id}")

    ds.cleanup_cache_files()
    personas.cleanup_cache_files()


def main(args: list[str] | None = None) -> None:
    """Build the persona QA seed from the command line.

    Args:
        args: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sample Dutch Wikipedia articles and pair each with a random"
            " Belgian-Dutch persona, question type and question/answer"
            " length."
        )
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of chunks to keep, or -1 for all"
        " that pass the quality filter.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Shortest chunk to keep (words).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=2560,
        help=(
            "Longest chunk to keep (words), so chunk + persona + prompt +"
            " reasoning + answer fit the max_model_len the configs set. An"
            " article above this is split into several chunks along its"
            " '## ' section headings instead of dropped; a single section"
            " longer than this on its own is sentence-truncated to fit."
        ),
    )
    parser.add_argument(
        "--max-section-chars",
        type=int,
        default=200_000,
        help=(
            "Longest section to tolerate (characters) before the whole"
            " article is dropped as malformed. Set far above any real"
            " section: this is a sanity check against markup/table dumps"
            " masquerading as one section, not a quality control on length."
        ),
    )
    parser.add_argument(
        "--min-avg-sentence-words",
        type=float,
        default=4.0,
        help=(
            "Lowest acceptable average sentence length in a chunk (words)."
            " Below this, a chunk is dropped as likely a bullet list rather"
            " than running text. Requires the 'spacy' extra."
        ),
    )
    parser.add_argument(
        "--max-avg-sentence-words",
        type=float,
        default=60.0,
        help=(
            "Highest acceptable average sentence length in a chunk (words)."
            " Above this, a chunk is dropped as likely a table or other"
            " unpunctuated dump rather than running text. Requires the"
            " 'spacy' extra."
        ),
    )
    parser.add_argument(
        "-j",
        "--num-proc",
        type=int,
        default=None,
        help=(
            "Processes to load and filter with. Default: unset, one process."
            " The output does not depend on this."
        ),
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Local directory to write the seed dataset to.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Optional HF Hub dataset id to push to.",
    )

    build_seed(**vars(parser.parse_args(args)))


if __name__ == "__main__":
    main()
