import argparse
from clause_splitters.clause_splitter import ClauseSplitter
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


@lru_cache(maxsize=None)
def get_splitter(model: str, enabled_splits: frozenset):
    return ClauseSplitter(model=model, enabled_splits=enabled_splits)

def splitter_fn(sentences: str | list[str], enabled_splits: list[str], model_name: str) -> list:
    splitter = get_splitter(model_name, frozenset(set(enabled_splits)))
    return splitter.split_sentence(sentences)


if __name__ == "__main__":
    # sentences = [' ! ( The Song Formerly Known As ) " is a song by Australian rock band Regurgitator ']
    # sentences = ['! ( pronounced " blah " ) is the debut studio album by Portuguese singer Cláudia Pascoal .']
    sentences = ['At the ARIA Music Awards of 1999 , the song was nominated for two awards ; ARIA Award for Best Group and ARIA Award for Single of the Year .']
    results = splitter_fn(sentences, ALL_SPLIT_TYPES, "en_core_web_trf")

    for sent in sentences:
        print(f"Original: {sent}")
        print("Splits:")
        for res in results:
            print(res)

