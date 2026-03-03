import spacy


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

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("coreferee")

    for d in data:
        start, end = d['metadata']['content_span']
        content = d['content'][start:end]

        doc = nlp(content)
        resolved_text = resolve_coreferences(doc)

        resolved_text = d['content'][:start] + resolved_text
        print("Original:", d['content'])
        print("Resolved:", resolved_text)