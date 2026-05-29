from __future__ import annotations

from typing import Any, cast

from llm_annotator.external.propella import propella


def test_flatten_model_json_schema_inlines_local_defs() -> None:
    # Verifies local $defs are inlined, list recursion is preserved, and external refs survive.
    schema = {
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        },
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "$ref": "#/$defs/Item",
                    "title": "Inline item",
                },
            },
            "external": {
                "$ref": "https://example.com/schema",
                "description": "kept",
            },
            "maybe": {
                "anyOf": [
                    {"$ref": "#/$defs/Item"},
                    {"type": "null"},
                ]
            },
        },
    }

    flattened = propella.flatten_model_json_schema(schema)

    assert "$defs" not in flattened
    assert flattened["properties"]["items"]["items"]["type"] == "object"
    assert flattened["properties"]["items"]["items"]["title"] == "Inline item"
    assert flattened["properties"]["external"]["$ref"] == (
        "https://example.com/schema"
    )
    assert flattened["properties"]["maybe"]["anyOf"][0]["type"] == "object"


def test_get_annotation_response_schema_variants() -> None:
    # Verifies schema generation variants for enum removal, flattening, and string output.
    schema_dict = propella.get_annotation_response_schema(
        use_country_enum=False,
        flatten=False,
        as_string=False,
        compact_whitespace=True,
    )

    schema_dict = cast(dict[str, Any], schema_dict)

    country_prop = schema_dict["properties"]["country_relevance"]
    assert schema_dict["x-guidance"]["whitespace_flexible"] is False
    assert country_prop["type"] == "array"
    assert "Valid values:" in country_prop["description"]

    schema_str = propella.get_annotation_response_schema(
        use_country_enum=True,
        flatten=True,
        as_string=True,
        minify=False,
        compact_whitespace=False,
    )

    assert isinstance(schema_str, str)
    assert "\n" in schema_str
    assert "$defs" not in schema_str


def test_truncate_and_message_helpers() -> None:
    # Verifies document truncation and prompt assembly helpers.
    truncated = propella.truncate_content("abcdef", 3)
    assert truncated == "abc\n<truncated_content>"
    assert propella.truncate_content("abcdef", 0) == "abcdef"

    messages = propella.create_messages("abcdef", max_content_chars=3)
    annotator_messages = propella.create_annotator_messages(
        "abcdef", max_content_chars=3
    )

    assert messages[0]["role"] == "system"
    assert "<truncated_content>" in messages[1]["content"]
    assert annotator_messages[0]["role"] == "system"
    assert "<start_of_document>" in annotator_messages[1]["content"]
    assert propella.annotator_system_prompt
