import spacy

if __name__ == "__main__":

    data = [
        {
            'content': "! (Trippie Redd album), Background \nTrippie Redd confirmed in an interview with Zane Lowe of Beats 1 Radio. He also revealed new tour dates.",
            'metadata': {
                "title_span": [0, 22],  # "! (Trippie Redd album)"
                "section_span": [24, 34],  # "Background"
                "content_span": [35, 140]  # testo da analizzare
            }
        },
        {
            'content': "! (The Beatles album), Production \nThe Beatles recorded the album in Abbey Road Studios. They experimented with new sounds and effects.",
            'metadata': {
                "title_span": [0, 22],  # "! (The Beatles album)"
                "section_span": [24, 34],  # "Production"
                "content_span": [35, 135]  # testo da analizzare
            }
        }
    ]

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("coreferee")

    for d in data:
        start, end = d['metadata']['content_span'][0], d['metadata']['content_span'][-1]
        content = d['content'][start:end]
        doc = nlp(content)

        resolved_tokens = []

        for token in doc:
            chains = doc._.coref_chains.resolve(token)

            if chains:
                resolved_tokens.append(chains[0].text)
            else:
                resolved_tokens.append(token.text)

        resolved_text = " ".join(resolved_tokens)

        resolved_text = d['content'][:start] + resolved_text
        print("Original:", d['content'])
        print("Resolved:", resolved_text)