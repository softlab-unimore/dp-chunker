import spacy
from functools import lru_cache

from stanza.pipeline.core import DownloadMethod
import stanza

PRONOUNS = {
    "he", "she", "it", "they", "him", "her", "them",
    "his", "hers", "its", "their", "theirs",
    "himself", "herself", "itself", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose"
}


@lru_cache(maxsize=None)
def get_nlp(model_name: str):
    nlp = spacy.load(model_name)
    nlp.add_pipe("coreferee")
    return nlp

@lru_cache(maxsize=None)
def get_stanza(lang: str = "en"):
    nlp = stanza.Pipeline(lang=lang, processors="tokenize,coref", model_dir='./stanza_resources', download_method=DownloadMethod.REUSE_RESOURCES)
    return nlp

def resolve_coreferences(doc):
    resolved_tokens = []

    for token in doc:
        chains = doc._.coref_chains.resolve(token)
        if chains:
            main_token = chains[0]
            modifiers = [child for child in main_token.lefts if child.dep_ in ("compound", "amod", "nummod", "poss", "det")]
            modifiers = sorted(modifiers, key=lambda t: t.i)
            modifiers = [t.text for t in modifiers]
            full_name = " ".join(modifiers + [main_token.text])
            resolved_tokens.append(full_name)
        else:
            resolved_tokens.append(token.text)

    return " ".join(resolved_tokens)

def parse_and_resolve_coreferences(texts: str | list[str], model_name: str) -> list[str]:
    nlp = get_nlp(model_name)

    if not isinstance(texts, list):
        texts = [texts]

    docs = list(nlp.pipe(texts))

    results = []
    for doc in docs:
        results.append(resolve_coreferences(doc))

    return results


def resolve_coreferences_with_stanza(doc: stanza.Document) -> str:
    sentence_offsets = []
    offset = 0
    for sent in doc.sentences:
        sentence_offsets.append(offset)
        offset += len(sent.words)

    all_words = [word for sent in doc.sentences for word in sent.words]

    replacements = {}

    for chain in doc.coref:
        repr_idx = chain.representative_index

        for i, mention in enumerate(chain.mentions):
            if i == repr_idx:
                continue

            sent = doc.sentences[mention.sentence]
            mention_words = sent.words[mention.start_word:mention.end_word]
            mention_text = " ".join(w.text for w in mention_words)

            if mention_text.lower().strip() not in PRONOUNS:
                continue

            sent_offset = sentence_offsets[mention.sentence]
            global_start = sent_offset + mention.start_word
            global_end = sent_offset + mention.end_word

            replacements[global_start] = chain.representative_text
            for j in range(global_start + 1, global_end):
                replacements[j] = None

    result = []
    for i, word in enumerate(all_words):
        if i in replacements:
            if replacements[i] is not None:
                result.append(replacements[i])
        else:
            result.append(word.text)

    return " ".join(result)

def parse_and_resolve_coreferences_with_stanza(texts: str | list[str], lang: str = "en") -> list[str]:
    nlp = get_stanza(lang)

    if not isinstance(texts, list):
        texts = [texts]

    docs = [nlp(text) for text in texts]

    return [resolve_coreferences_with_stanza(doc) for doc in docs]


if __name__ == "__main__":
    data = [
        {
            'content': "! (Cláudia Pascoal album) ! (pronounced \"blah\") is the debut studio album by Portuguese singer Cláudia Pascoal. "
                       "It was released in Portugal on 27 March 2020 by Universal Music Portugal. The album peaked at number six on the Portuguese Albums Chart.",
            'metadata': {
                "title_span": [0, 25],  # "! (Trippie Redd album)"
                "section_span": [25, 25],  # "Background"
                "content_span": [26, 248]  # testo da analizzare
            }
        },
        {
            'content': "! (Trippie Redd album), Background \nIn January 2019, Trippie Redd announced that he had two more projects "
                       "to be released soon in an Instagram live stream, his second studio album, Immortal and Mobile Suit Pussy, "
                       "which was reportedly set to be his fourth commercial mixtape, but it then became scrapped. He explained that "
                       "Immortal would have tracks where deep and romantic concepts are present, while Mobile Suit Pussy would have "
                       "contained tracks that are \"bangers\". Later in March 2019 in another Instagram live stream, Redd stated that his "
                       "second album had \"shifted and changed\" and was no longer titled Immortal. He later revealed that the album would "
                       "be titled !, and inspired by former collaborator XXXTentacion's ? album.",
            'metadata': {
                "title_span": [0, 22],  # "! (The Beatles album)"
                "section_span": [24, 34],  # "Production"
                "content_span": [35, 725]  # testo da analizzare
            }
        }
    ]

    # nlp = spacy.load("en_core_web_lg")
    # nlp.add_pipe("coreferee")
    stanza.download("en", model_dir='./stanza_resources')
    nlp = stanza.Pipeline(lang="en", processors="tokenize,coref", model_dir='./stanza_resources', download_method=DownloadMethod.REUSE_RESOURCES)

    for d in data:
        start, end = d['metadata']['content_span']
        content = d['content'][start:end]

        doc = nlp(content)
        # resolved_text = resolve_coreferences(doc)
        resolved_text = resolve_coreferences_with_stanza(doc)

        # resolved_text = d['content'][:start] + resolved_text
        print("Original:", d['content'])
        print("Resolved:", resolved_text)
