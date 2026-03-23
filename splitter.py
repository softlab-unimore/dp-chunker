import argparse
from splitters.clause_splitter import ClauseSplitter

ALL_SPLIT_TYPES = ClauseSplitter.ALL_SPLIT_TYPES

DEFAULT_SENTENCES = [
    "The English professors and students protested.",
    "Young, beautiful and nice doctors and nurses worked hard.",

    # nominal conj
    "The professors and students protested.",
    "The doctors and nurses worked hard.",
    "The professors and students protested because the exam was unfair.",
    "The teachers and students believe that the system needs reform.",
    "The professors and students who attended protested.",
    "The professors and students who attended said that the lecture was inspiring.",
    "The professors and students protested; the dean announced that the policy would change.",

    # amod coordinati
    "American and British professors attended.",
    "Young and experienced doctors treated the patients.",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clause splitter — ablation mode.", formatter_class=argparse.RawDescriptionHelpFormatter,)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable", nargs="+", metavar="TYPE", choices=sorted(ALL_SPLIT_TYPES),
                       help="Enable only the listed clause types. Choices: {sorted(ALL_SPLIT_TYPES)}")
    group.add_argument("--disable", nargs="+", metavar="TYPE", choices=sorted(ALL_SPLIT_TYPES),
                       help="Enable all clause types except the listed ones.")

    parser.add_argument("--sentence", nargs="+", metavar="SENTENCE", default=None,
                        help="One or more sentences to split (overrides the built-in test set).")
    parser.add_argument("--model", default="en_core_web_lg", help="spaCy model to use (default: en_core_web_lg).")

    return parser


def resolve_enabled_splits(args) -> set:
    """Return the final set of enabled split types from parsed args."""
    if args.enable:
        return set(args.enable)
    if args.disable:
        return ALL_SPLIT_TYPES - set(args.disable)
    return set(ALL_SPLIT_TYPES)  # default: all


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    enabled = resolve_enabled_splits(args)
    sentences = args.sentence if args.sentence else DEFAULT_SENTENCES

    print(f"Enabled split types: {sorted(enabled)}\n")

    splitter = ClauseSplitter(model=args.model, enabled_splits=enabled)

    for s in sentences:
        splits = splitter.split_sentence(s)
        print(f"{s}")
        for split in splits:
            print(f"  - {split}")
        print()


if __name__ == "__main__":
    main()