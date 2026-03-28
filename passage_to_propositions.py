import os
import pandas as pd
import argparse
from tqdm import tqdm
from splitter import splitter_fn
from coref import parse_and_resolve_coreferences

rule_mapping = {
    "all": ["advcl", "acl", "relcl", "conj", "ccomp", "parataxis", "nominal_conj"],
    "coordinates_only": ["conj"], # relcl? nominal_conj?
    "subordinates_only": ["advcl", "acl", "relcl", "ccomp"],
    "parataxis_only": ["parataxis"],
    "adverbs_only": ["advcl"],
    "adnominals_only": ["acl"],
    "adjectives_only": ["relcl"],
    "complements_only": ["ccomp"]
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv', type=str, required=True, help='path to input csv')
    parser.add_argument('--output_csv', type=str, required=True, help='path to output csv')
    parser.add_argument("--model", default="en_core_web_lg", help="spaCy model to use (default: en_core_web_lg).")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--coordinates_only", action="store_true")
    group.add_argument("--subordinates_only", action="store_true")
    group.add_argument("--parataxis_only", action="store_true")
    group.add_argument("--adverbs_only", action="store_true")
    group.add_argument("--adnominals_only", action="store_true")
    group.add_argument("--adjectives_only", action="store_true")
    group.add_argument("--complements_only", action="store_true")
    group.add_argument("--no_coreference", action="store_true")

    args = vars(parser.parse_args())

    INPUT_CSV = args['input_csv']
    OUTPUT_CSV = args['output_csv']
    MODEL_NAME = args['model']
    rules = None
    for k in args:
        if k in ["input_csv", "output_csv"]:
            continue

        if args[k]:
            key = k
            if k == "no_coreference":
                key = "all"

            rules = rule_mapping[key]
            break

    if rules is None:
        raise ValueError("rules cannot be None")

    CHUNK_SIZE = 1000

    # output file must be removed if we want a fresh run
    if os.path.exists(OUTPUT_CSV):
        raise ValueError(f"{OUTPUT_CSV} already exists. Cancel it before recomputing")

    first_write = True

    for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE), desc=f"Iterating over chunks of {CHUNK_SIZE} size"):
        rows = []

        for _, row in chunk.iterrows():
            paragraph = row["contents"]
            paragraph_id = row["id"]
            if not args["no_coreference"]:
                paragraph = parse_and_resolve_coreferences(paragraph, MODEL_NAME)

            props = splitter_fn(paragraph, rules, MODEL_NAME)

            for i, prop in enumerate(props):
                prop_num = str(i)
                if len(prop_num) > 4:
                    raise ValueError("a passage leads to more than 9999 propositions. Reduce passage size")

                prop_num = "0" * (4-len(prop_num)) + prop_num
                rows.append({
                    "id": paragraph_id+"-"+prop_num,
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