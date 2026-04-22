import argparse
import os

import pandas as pd
import spacy
from tqdm import tqdm

from clause_splitters.clause_splitter import ClauseSplitter
from functools import lru_cache

from complements_splitters.sentence_splitter import split_atomic
from coref import parse_and_resolve_coreferences_with_stanza

ALL_SPLIT_TYPES = ClauseSplitter.ALL_SPLIT_TYPES

DEFAULT_SENTENCES = [
    '! ( pronounced " blah " ) is the debut studio album by Portuguese singer Cláudia Pascoal . the debut studio album '
    'by Portuguese singer Cláudia Pascoal was released in Portugal on 27 March 2020 by Universal Music Portugal . The '
    'album peaked at number six on the Portuguese Albums Chart .'
]


@lru_cache(maxsize=None)
def get_splitter(model: str, enabled_splits: frozenset):
    return ClauseSplitter(model=model, enabled_splits=enabled_splits)

def splitter_fn(sentences: str | list[str], enabled_splits: list[str], model_name: str) -> list:
    splitter = get_splitter(model_name, frozenset(set(enabled_splits)))
    return splitter.split_sentence(sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', type=str, required=True, help='path to input csv')
    parser.add_argument('--output_csv', type=str, required=True, help='path to output csv')
    parser.add_argument("--model", default="en_core_web_lg", help="spaCy model to use (default: en_core_web_lg).")
    parser.add_argument('--no_coreference', action='store_true', help='if set, no coreference resolution will be performed')

    args = vars(parser.parse_args())

    INPUT_CSV = args['input_csv']
    OUTPUT_CSV = args['output_csv']
    MODEL_NAME = args['model']

    sentences = DEFAULT_SENTENCES
    CHUNK_SIZE = 1000

    # output file must be removed if we want a fresh run
    if os.path.exists(OUTPUT_CSV):
        raise ValueError(f"{OUTPUT_CSV} already exists. Cancel it before recomputing")

    nlp = spacy.load("en_core_web_trf")
    first_write = True

    for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE), desc=f"Iterating over chunks of {CHUNK_SIZE} size"):
        rows = []

        for row in tqdm(chunk.itertuples()):
            paragraph = row.contents
            paragraph_id = row.id
            # lines = str(paragraph).splitlines()
            paragraph = "\n".join(paragraph.splitlines()[1:])

            if not args["no_coreference"]:
                # paragraph = parse_and_resolve_coreferences(paragraph, MODEL_NAME)
                paragraph = parse_and_resolve_coreferences_with_stanza(paragraph, lang="en")

            props = split_atomic(paragraph[0], nlp)

            for prop in props:
                print(prop)

            print('here')

            for i, prop in enumerate(props):
                prop_num = str(i)
                if len(prop_num) > 4:
                    raise ValueError("a passage leads to more than 9999 propositions. Reduce passage size")

                prop_num = "0" * (4 - len(prop_num)) + prop_num
                rows.append({
                    "id": paragraph_id + "-" + prop_num,
                    "contents": prop,
                    "metadata": {}
                })

        if rows:
            out_df = pd.DataFrame(rows)

            out_df.to_csv(
                OUTPUT_CSV,
                mode="a",
                header=first_write,
                index=False
            )

            first_write = False


