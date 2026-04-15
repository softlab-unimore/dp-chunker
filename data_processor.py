import pandas as pd

index_root_path = "./indexes/"

index_paths = {
    "gtr-t5-base": {
        "proposition": {
            "index": "gtr-t5-base/gtr-t5-base_proposition_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_proposition_meta.json",
        },
        "sentence": {
            "index": "gtr-t5-base/gtr-t5-base_sentences_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_sentences_meta.json"
        },
        "passage": {
            "index": "gtr-t5-base/gtr-t5-base_passages_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_passages_meta.json",
        },
        "no_coreference": {
            "index": "gtr-t5-base/gtr-t5-base_no_coreference_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_no_coreference_meta.json",
        },
	"all": {
            "index": "gtr-t5-base/gtr-t5-base_all_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_all_meta.json",
        },
	"adjectives_only": {
            "index": "gtr-t5-base/gtr-t5-base_adjectives_only_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_adjectives_only_meta.json",
        },
	"adnominals_only": {
            "index": "gtr-t5-base/gtr-t5-base_adnominals_only_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_adnominals_only_meta.json",
        },
	"adverbs_only": {
            "index": "gtr-t5-base/gtr-t5-base_adverbs_only_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_adverbs_only_meta.json",
        },
	"complements_only": {
            "index": "gtr-t5-base/gtr-t5-base_complements_only_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_complements_only_meta.json",
        },
	"coordinates_only": {
            "index": "gtr-t5-base/gtr-t5-base_coordinates_only_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_coordinates_only_meta.json",
        },
        "parataxis_only": {
            "index": "gtr-t5-base/gtr-t5-base_parataxis_only_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_parataxis_only_meta.json",
        },
        "subordinates_only": {
            "index": "gtr-t5-base/gtr-t5-base_subordinates_only_index.faiss",
            "meta": "gtr-t5-base/gtr-t5-base_subordinates_only_meta.json",
        },
    },
    "contriever": {
        "proposition": {
            "index": "contriever/contriever_proposition_index.faiss",
            "meta": "contriever/contriever_proposition_meta.json",
        },
        "sentence": {
            "index": "contriever/contriever_sentences_index.faiss",
            "meta": "contriever/contriever_sentences_meta.json"
        },
        "passage": {
            "index": "contriever/contriever_passages_index.faiss",
            "meta": "contriever/contriever_passages_meta.json",
        },
	"no_coreference": {
            "index": "contriever/contriever_no_coreference_index.faiss",
            "meta": "contriever/contriever_no_coreference_meta.json",
        },
        "all": {
            "index": "contriever/contriever_all_index.faiss",
            "meta": "contriever/contriever_all_meta.json",
        },
        "adjectives_only": {
            "index": "contriever/contriever_adjectives_only_index.faiss",
            "meta": "contriever/contriever_adjectives_only_meta.json",
        },
	"adnominals_only": {
            "index": "contriever/contriever_adnominals_only_index.faiss",
            "meta": "contriever/contriever_adnominals_only_meta.json",
        },
	"adverbs_only": {
            "index": "contriever/contriever_adverbs_only_index.faiss",
            "meta": "contriever/contriever_adverbs_only_meta.json",
        },
        "complements_only": {
            "index": "contriever/contriever_complements_only_index.faiss",
            "meta": "contriever/contriever_complements_only_meta.json",
        },
	"coordinates_only": {
            "index": "contriever/contriever_coordinates_only_index.faiss",
            "meta": "contriever/contriever_coordinates_only_meta.json",
        },
	"parataxis_only": {
            "index": "contriever/contriever_parataxis_only_index.faiss",
            "meta": "contriever/contriever_parataxis_only_meta.json",
        },
        "subordinates_only": {
            "index": "contriever/contriever_subordinates_only_index.faiss",
            "meta": "contriever/contriever_subordinates_only_meta.json",
        },
    },
    "multilingual-e5-large-instruct": {
        "proposition": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_proposition_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_proposition_meta.json",
        },
        "sentence": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_sentences_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_sentences_meta.json"
        },
        "passage": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_passages_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_passages_meta.json",
        },
	"no_coreference": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_no_coreference_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_no_coreference_meta.json",
        },
        "all": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_all_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_all_meta.json",
	},
        "adjectives_only": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_adjectives_only_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_adjectives_only_meta.json",
	},
        "adnominals_only": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_adnominals_only_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_adnominals_only_meta.json",
        },
        "adverbs_only": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_adverbs_only_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_adverbs_only_meta.json",
        },
        "complements_only": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_complements_only_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_complements_only_meta.json",
        },
	"coordinates_only": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_coordinates_only_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_coordinates_only_meta.json",
        },
	"parataxis_only": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_parataxis_only_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_parataxis_only_meta.json",
        },
	"subordinates_only": {
            "index": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_subordinates_only_index.faiss",
            "meta": "multilingual-e5-large-instruct/multilingual-e5-large-instruct_subordinates_only_meta.json",
        },
    },
    "sup-simcse-bert-base-uncased": {
        "proposition": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_proposition_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_proposition_meta.json",
        },
        "sentence": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_sentences_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_sentences_meta.json"
        },
        "passage": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_passages_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_passages_meta.json",
        },
	"no_coreference": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_no_coreference_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_no_coreference_meta.json",
        },
        "all": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_all_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_all_meta.json",
	},
        "adjectives_only": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_adjectives_only_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_adjectives_only_meta.json",
	},
        "adnominals_only": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_adnominals_only_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_adnominals_only_meta.json",
        },
        "adverbs_only": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_adverbs_only_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_adverbs_only_meta.json",
        },
        "complements_only": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_complements_only_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_complements_only_meta.json",
        },
	"coordinates_only": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_coordinates_only_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_coordinates_only_meta.json",
        },
	"parataxis_only": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_parataxis_only_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_parataxis_only_meta.json",
        },
	"subordinates_only": {
            "index": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_subordinates_only_index.faiss",
            "meta": "sup-simcse-bert-base-uncased/sup-simcse-bert-base-uncased_subordinates_only_meta.json",
        },
    },
}


class DataProcessor:
    def __init__(self):
        self.data_path = None

    def read_data(self):
        if self.data_path is None:
            return None, None

        df = pd.read_csv(self.data_path, sep="\t")
        questions = df.iloc[:,0].tolist()
        results = df.iloc[:,1].tolist()
        results = [eval(res) for res in results]

        return questions, results

class NQProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/nq/test.tsv"

class EQProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/eq/test.tsv"

class SquadProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/squad/test.tsv"

class TriviaQAProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/triviaqa/test.tsv"

class WebQProcessor(DataProcessor):
    def __init__(self):
        self.data_path = "./data/webq/test.tsv"
