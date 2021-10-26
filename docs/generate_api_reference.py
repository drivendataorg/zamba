"""Helper script for generating mkdocstrings pages in `docs/docs/api-reference`. Meant
to be run manually once in a while with output committed to repo."""

from pathlib import Path
from textwrap import dedent

from yaml import safe_load


DOCS_DIR = Path(__file__).parent


def generate_file(item: dict):
    key = next(iter(item.keys()))
    value = next(iter(item.values()))
    if isinstance(value, list):
        for sub_item in value:
            generate_file(sub_item)
    elif isinstance(value, str):
        content = dedent(
            f"""\
            # {key}

            ::: {key}
            """
        )
        out_path = DOCS_DIR / "docs" / value
        if not out_path.exists():
            print(f"Creating {out_path} for {key}")
            with out_path.open("w") as fh:
                fh.write(content)
        else:
            print(f"Found existing {out_path}. Skipping.")
    else:
        raise ValueError(f"Something is wrong with this navitem: ({key}, {value})")


def main():
    with (DOCS_DIR / "mkdocs.yml").open("r") as fh:
        mkdocs = safe_load(fh)

    for nav_item in mkdocs["nav"]:
        if "API Reference" in nav_item:
            api_reference = nav_item
            break

    (DOCS_DIR / "docs" / "api-reference").mkdir(parents=True, exist_ok=True)

    for api_reference_item in api_reference["API Reference"]:
        generate_file(api_reference_item)


if __name__ == "__main__":
    main()
