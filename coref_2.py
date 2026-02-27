import spacy
import coreferee

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("coreferee")

text = "My brother, who lives in Paris, is visiting and he will stay for two days."
doc = nlp(text)

resolved_tokens = []

for token in doc:
    # Se il token fa parte di una coreference chain
    chains = doc._.coref_chains.resolve(token)

    if chains:
        # Prendiamo il primo referente trovato
        resolved_tokens.append(chains[0].text)
    else:
        resolved_tokens.append(token.text)

resolved_text = " ".join(resolved_tokens)

print("Original:", text)
print("Resolved:", resolved_text)