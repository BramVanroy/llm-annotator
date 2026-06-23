from __future__ import annotations

from datasets import Dataset, load_dataset


def is_stub(wikitexts: list[str], titles: list[str]) -> list[bool]:
    """Return ``True`` when a Wikipedia article is a stub page.

    Args:
        wikitexts: Raw page texts.
        titles: The titles of the pages.

    Returns:
        Whether the pages look like a stub.
    """
    # E.g. overview pages like https://nl.wikipedia.org/wiki/Categorie:Wikipedia:Beginnetje_biologie
    return [
        r"{{beginnetje" in wikitext.lower() or title.startswith("Categorie:")
        for wikitext, title in zip(wikitexts, titles)
    ]


def is_list(titles: list[str]) -> list[bool]:
    """Return ``True`` when a Wikipedia article is a list page.

    Args:
        titles: The titles of the pages.
    Returns:
        Whether the pages look like a list.
    """
    return [title.lower().startswith("lijst van") for title in titles]


def is_short(texts: list[str], max_chars: int = 768) -> list[bool]:
    """Return ``True`` when a Wikipedia article is short.

    Args:
        texts: The texts of the pages.
        max_chars: The maximum number of characters for an article to be considered short.
    Returns:
        Whether the page looks like a short article.
    """
    return [len(text) <= max_chars for text in texts]


def filter_dataset(
    ds: Dataset, num_proc: int | None, hub_id: str | None = None
) -> None:
    num_before = len(ds)
    ds_stubs = ds.filter(
        lambda wikitexts, titles: is_stub(wikitexts, titles),
        input_columns=["wikitext", "title"],
        batched=True,
        num_proc=num_proc,
    )
    ds_stubs.push_to_hub(f"{hub_id}-stubs", private=True)

    print(f"Filtering dataset of {num_before:,} articles...")
    ds = ds.filter(
        lambda wikitexts, titles: [
            not is_text_stub for is_text_stub in is_stub(wikitexts, titles)
        ],
        input_columns=["wikitext", "title"],
        batched=True,
        num_proc=num_proc,
    )
    num_after_stub = len(ds)
    print(
        f"Filtered {num_before - num_after_stub:,} stub pages, {num_after_stub:,} remaining."
    )

    ds = ds.filter(
        lambda titles: [not is_text_list for is_text_list in is_list(titles)],
        input_columns=["title"],
        batched=True,
        num_proc=num_proc,
    )
    num_after_list = len(ds)
    print(
        f"Filtered {num_after_stub - num_after_list:,} list pages, {num_after_list:,} remaining."
    )

    ds = ds.filter(
        lambda texts: [not is_text_short for is_text_short in is_short(texts)],
        input_columns=["text"],
        batched=True,
        num_proc=num_proc,
    )
    num_after_short = len(ds)
    print(
        f"Filtered {num_after_list - num_after_short:,} short articles, {num_after_short:,} remaining."
    )

    if hub_id:
        ds.push_to_hub(hub_id, private=True)
    return ds


def _explode_text_into_sections(examples) -> dict[list]:
    """Split a text into sections based on level-2 markdown headings.

    Args:
        text: The text to split.
    Returns:
        A list of sections.
    """
    all_keys = list(examples.keys())
    data = {k: [] for k in all_keys}
    data["heading"] = []

    for sample_idx, text in enumerate(examples["text"]):
        current_section = []
        heading = ""
        for line in text.splitlines():
            if line.startswith("# "):
                heading = line[2:].strip()
                continue
            if line.startswith("## "):
                if current_section:
                    for k in all_keys:
                        if k != "text":
                            data[k].append(examples[k][sample_idx])
                    data["heading"].append(heading)
                    data["text"].append("\n".join(current_section))
                    current_section = []
            current_section.append(line)

        if current_section:
            for k in all_keys:
                if k != "text":
                    data[k].append(examples[k][sample_idx])
            data["heading"].append(heading)
            data["text"].append("\n".join(current_section))

    return data


def process_section_dataset(ds: Dataset, num_proc: int | None) -> None:
    """
    Process the dataset into sections based on level-2 markdown headings.

    Args:
        ds: The dataset to process.
        num_proc: The number of processes to use for parallel processing.
    Returns:
        The processed dataset with sections.
    """
    # Explode dataset into sections
    ds = ds.map(
        _explode_text_into_sections,
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
    )

    return ds


def main(args: list[str] | None = None) -> None:
    """Preprocess the finewiki source for the Propella example.

    Args:
        args: Optional command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess propella: truncate text to first 50,000 characters."
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--hub-id",
        required=True,
        help="HF Hub dataset ID to push filtered dataset to.",
    )
    parsed_args = parser.parse_args(args)

    ds = load_dataset("HuggingFaceFW/finewiki", "nl", split="train")
    main_ds = filter_dataset(
        ds, num_proc=parsed_args.num_proc, hub_id=f"{parsed_args.hub_id}-flt"
    )

    # Trim to 50,000 characters for propella
    main_ds_trunc = main_ds.map(
        lambda text: {"text_truncated": text[:50_000]},
        input_columns=["text"],
        num_proc=parsed_args.num_proc,
    )
    main_ds_trunc.push_to_hub(f"{parsed_args.hub_id}-trunc", private=True)

    # sections
    section_ds = process_section_dataset(
        main_ds, num_proc=parsed_args.num_proc
    )
    section_ds.push_to_hub(f"{parsed_args.hub_id}-sections", private=True)
    section_ds = filter_dataset(
        section_ds,
        num_proc=parsed_args.num_proc,
        hub_id=f"{parsed_args.hub_id}-sections-flt",
    )

    # Trim to 50,000 characters for propella
    section_ds_trunc = section_ds.map(
        lambda text: {"text_truncated": text[:50_000]},
        input_columns=["text"],
        num_proc=parsed_args.num_proc,
    )
    section_ds_trunc.push_to_hub(
        f"{parsed_args.hub_id}-sections-trunc", private=True
    )


if __name__ == "__main__":
    main()
