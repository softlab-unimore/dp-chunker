import argparse
from splitters.clause_splitter import ClauseSplitter
from functools import lru_cache

ALL_SPLIT_TYPES = ClauseSplitter.ALL_SPLIT_TYPES

DEFAULT_SENTENCES = [
    # ── advcl ────────────────────────────────────────────────────────────────
    # simple  --enable advcl
    "She left early because she felt sick.",
    "He called her when she arrived.",
    # medium  --enable advcl ccomp
    "She thinks that he left because he was angry.",
    # complex  --enable advcl relcl
    "The man who left because he was angry never returned.",

    # ── acl ──────────────────────────────────────────────────────────────────
    # simple  --enable acl
    "She had a decision to make.",
    # medium  --enable acl advcl
    "He found a way to win before the time ran out.",
    # complex  --enable acl relcl
    "She had a plan to follow that her mentor recommended.",

    # ── relcl ────────────────────────────────────────────────────────────────
    # simple  --enable relcl
    "The woman who called was his sister.",
    "The man I met was a doctor.",
    # medium  --enable relcl advcl
    "The student who failed because she missed the exam asked for help.",
    # complex  --enable relcl ccomp
    "The book that the author who won the prize wrote inspired millions.",

    # ── ccomp ────────────────────────────────────────────────────────────────
    # simple  --enable ccomp
    "She believes that he is innocent.",
    # medium  --enable ccomp advcl
    "She believes that he is innocent because the evidence is clear.",
    # complex  --enable ccomp
    "She knows that he believes that the earth is flat.",

    # ── conj ─────────────────────────────────────────────────────────────────
    # simple  --enable conj
    "He came home and took a shower.",
    "She cooked dinner and he washed the dishes.",
    # medium  --enable conj ccomp
    "She believes that he is innocent and that the judge agreed.",
    # complex  --enable conj relcl ccomp
    "He came home and said that the woman who called was his sister.",

    # ── parataxis ────────────────────────────────────────────────────────────
    # simple  --enable parataxis
    "I know, I said it before.",
    "She left early: she was tired.",
    # medium  --enable parataxis advcl
    "She left early: she was tired because the meeting had gone badly.",
    # complex  --enable parataxis relcl ccomp
    "The result was clear: the man who led the project had failed.",

    # ── nominal_conj ─────────────────────────────────────────────────────────
    # simple  --enable nominal_conj
    "The scientists and engineers collaborated.",
    "Senior and junior researchers presented.",
    # medium  --enable nominal_conj advcl
    "The managers and employees protested because the policy was unfair.",
    # medium  --enable nominal_conj ccomp
    "The directors and shareholders agreed that the merger was necessary.",
    # complex  --enable nominal_conj relcl
    "The doctors and nurses who treated him worked tirelessly.",
    # complex  --enable nominal_conj relcl ccomp
    "The managers and employees said that the policy was wrong because it hurt everyone.",

    # ── ablation / cross-type ─────────────────────────────────────────────────
    # --disable nominal_conj
    "The professors and students who attended protested.",
    # --disable relcl
    "The scientist who discovered the cure said that he succeeded because he worked hard.",
    # --disable ccomp parataxis
    "I know, she believes that he is innocent because the evidence was clear.",
    # --enable advcl ccomp relcl
    "The scientist who discovered the cure said that he had a chance to publish because the journal accepted his work.",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clause splitter — ablation mode.", formatter_class=argparse.RawDescriptionHelpFormatter,)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--enable", nargs="+", metavar="TYPE", choices=sorted(ALL_SPLIT_TYPES),
                       help="Enable only the listed clause types. Choices: {sorted(ALL_SPLIT_TYPES)}")
    group.add_argument("--disable", nargs="+", metavar="TYPE", choices=sorted(ALL_SPLIT_TYPES),
                       help="Enable all clause types except the listed ones.")

    parser.add_argument("--model", default="en_core_web_lg", help="spaCy model to use (default: en_core_web_lg).")

    return parser


def resolve_enabled_splits(args) -> set:
    """Return the final set of enabled split types from parsed args."""
    if args.enable:
        return set(args.enable)
    if args.disable:
        return ALL_SPLIT_TYPES - set(args.disable)
    return set(ALL_SPLIT_TYPES)  # default: all


def splitter_fn(sentence: str) -> list:
    @lru_cache(max_size=None)
    def get_splitter(model: str, enabled_splits: frozenset):
        return ClauseSplitter(model=model, enabled_splits=enabled_splits)

    parser = build_arg_parser()
    args = parser.parse_args()

    enabled = resolve_enabled_splits(args)

    #print(f"Enabled split types: {sorted(enabled)}\n")

    splitter = get_splitter(args.model, frozenset(enabled))
    return splitter.split_sentence(sentence)


if __name__ == "__main__":
    splitter_fn("sample sentence")