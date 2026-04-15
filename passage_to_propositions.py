import os
import pandas as pd
import argparse
from tqdm import tqdm
from splitter import splitter_fn
# from coref import parse_and_resolve_coreferences, parse_and_resolve_coreferences_with_stanza

# rule_mapping = {
#     "all": ["advcl", "acl", "relcl", "conj", "ccomp", "parataxis", "nominal_conj"],
#     "coordinates_only": ["conj"], # relcl? nominal_conj?
#     "subordinates_only": ["advcl", "acl", "relcl", "ccomp"],
#     "parataxis_only": ["parataxis"],
#     "adverbs_only": ["advcl"],
#     "adnominals_only": ["acl"],
#     "adjectives_only": ["relcl"],
#     "complements_only": ["ccomp"]
# }
#

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--input_csv', type=str, required=True, help='path to input csv')
#     parser.add_argument('--output_csv', type=str, required=True, help='path to output csv')
#     parser.add_argument("--model", default="en_core_web_lg", help="spaCy model to use (default: en_core_web_lg).")
#
#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument("--all", action="store_true")
#     group.add_argument("--coordinates_only", action="store_true")
#     group.add_argument("--subordinates_only", action="store_true")
#     group.add_argument("--parataxis_only", action="store_true")
#     group.add_argument("--adverbs_only", action="store_true")
#     group.add_argument("--adnominals_only", action="store_true")
#     group.add_argument("--adjectives_only", action="store_true")
#     group.add_argument("--complements_only", action="store_true")
#     group.add_argument("--no_coreference", action="store_true")
#
#     args = vars(parser.parse_args())
#
#     INPUT_CSV = args['input_csv']
#     OUTPUT_CSV = args['output_csv']
#     MODEL_NAME = args['model']
#     rules = None
#     for k in args:
#         if k in ["input_csv", "output_csv", "model"]:
#             continue
#
#         if args[k]:
#             key = k
#             if k == "no_coreference":
#                 key = "all"
#
#             rules = rule_mapping[key]
#             break
#
#     if rules is None:
#         raise ValueError("rules cannot be None")
#
#     CHUNK_SIZE = 1000
#
#     # output file must be removed if we want a fresh run
#     if os.path.exists(OUTPUT_CSV):
#         raise ValueError(f"{OUTPUT_CSV} already exists. Cancel it before recomputing")
#
#     first_write = True
#
#     for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE), desc=f"Iterating over chunks of {CHUNK_SIZE} size"):
#         rows = []
#
#         for row in tqdm(chunk.itertuples()):
#             paragraph = row.contents
#             paragraph_id = row.id
#             paragraph = "\n".join(paragraph.splitlines()[1:])
#             if not args["no_coreference"]:
#                 # paragraph = parse_and_resolve_coreferences(paragraph, MODEL_NAME)
#                 paragraph = parse_and_resolve_coreferences_with_stanza(paragraph, lang="en")
#
#             props = splitter_fn(paragraph, rules, MODEL_NAME)
#
#             for i, prop in enumerate(props):
#                 prop_num = str(i)
#                 if len(prop_num) > 4:
#                     raise ValueError("a passage leads to more than 9999 propositions. Reduce passage size")
#
#                 prop_num = "0" * (4-len(prop_num)) + prop_num
#                 rows.append({
#                     "id": paragraph_id+"-"+prop_num,
#                     "contents": prop,
#                     "metadata": {}
#                 })
#
#         if rows:
#             out_df = pd.DataFrame(rows)
#
#             out_df.to_csv(
#                 OUTPUT_CSV,
#                 mode="a",
#                 header=first_write,
#                 index=False
#             )
#
#             first_write = False


import os
import math
import pandas as pd
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from splitter import splitter_fn
from coref import parse_and_resolve_coreferences, parse_and_resolve_coreferences_with_stanza

rule_mapping = {
    "all": ["advcl", "acl", "relcl", "conj", "ccomp", "parataxis", "nominal_conj"],
    "coordinates_only": ["conj"],
    "subordinates_only": ["advcl", "acl", "relcl", "ccomp"],
    "parataxis_only": ["parataxis"],
    "adverbs_only": ["advcl"],
    "adnominals_only": ["acl"],
    "adjectives_only": ["relcl"],
    "complements_only": ["ccomp"]
}


def remove_first_line(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[1:]) if len(lines) > 1 else ""


def process_rows(batch, use_coref, rules, model_name, chunk_size=64):
    rows = []

    #for paragraph_id, paragraph in batch:
    for i in range(0, len(batch), chunk_size):
        data = batch[i:i+chunk_size]
        paragraphs = [remove_first_line(paragraph) for _, paragraph in data]
        paragraph_ids = [paragraph_id for paragraph_id, _ in data]

        if use_coref:
            # paragraphs = parse_and_resolve_coreferences(paragraphs, model_name)
            paragraphs = parse_and_resolve_coreferences_with_stanza(paragraphs, "en")

        list_props = splitter_fn(paragraphs, rules, model_name)

        for k, props in enumerate(list_props):
            for j, prop in enumerate(props):
                if j > 9999:
                    raise ValueError("a passage leads to more than 9999 propositions. Reduce passage size")

                rows.append({
                    "id": f"{paragraph_ids[k]}-{j:04d}",
                    "contents": prop,
                    "metadata": {}
                })

    return rows


def chunk_list(lst, n_chunks):
    if not lst:
        return []
    chunk_size = math.ceil(len(lst) / n_chunks)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model", default="en_core_web_trf")
    parser.add_argument("--num_workers", type=int, default=8)

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

    input_csv = args["input_csv"]
    output_csv = args["output_csv"]
    model_name = args["model"]
    num_workers = args["num_workers"]

    rules = None
    for k, v in args.items():
        if k in {"input_csv", "output_csv", "model", "num_workers"}:
            continue
        if v:
            key = "all" if k == "no_coreference" else k
            rules = rule_mapping[key]
            break

    if rules is None:
        raise ValueError("rules cannot be None")

    use_coref = not args["no_coreference"]
    chunk_size = 10000

    if os.path.exists(output_csv):
        raise ValueError(f"{output_csv} already exists. Cancel it before recomputing")

    first_write = True

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk in tqdm(
            pd.read_csv(input_csv, chunksize=chunk_size),
            desc=f"Iterating over chunks of {chunk_size} size"
        ):
            batch = [(row.id, row.contents) for row in chunk.itertuples(index=False)]
            sub_batches = chunk_list(batch, num_workers)

            futures = [
                executor.submit(process_rows, sub_batch, use_coref, rules, model_name)
                for sub_batch in sub_batches
                if sub_batch
            ]

            rows = []
            for future in futures:
                rows.extend(future.result())

            if rows:
                out_df = pd.DataFrame(rows)
                out_df.to_csv(
                    output_csv,
                    mode="a",
                    header=first_write,
                    index=False
                )
                first_write = False
