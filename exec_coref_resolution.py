from stanza.server import CoreNLPClient

def resolve_coref_text(annotation):
    """
    Sostituisce i pronomi (o altre menzioni nella coref chain)
    con la prima menzione rappresentativa della catena.
    Restituisce il testo coref-resolved.
    """
    # Flatten di tutti i token nel documento
    tokens = [t.word for sent in annotation.sentence for t in sent.token]

    # Mappa globale: token_index -> testo sostitutivo
    replacement_map = {}

    # Itero tutte le catene di coref
    for chain in annotation.corefChain:
        # Prima menzione della catena = rappresentativa
        rep = chain.mention[0]
        rep_text = " ".join(tokens[rep.beginIndex:rep.endIndex])

        # Tutte le altre menzioni vengono sostituite con la rappresentativa
        for mention in chain.mention[1:]:
            for i in range(mention.beginIndex, mention.endIndex):
                replacement_map[i] = rep_text

    # Costruisco il testo sostituendo i token
    resolved_tokens = []
    i = 0
    while i < len(tokens):
        if i in replacement_map:
            resolved_tokens.append(replacement_map[i])
            # Salto tutti i token della menzione
            while i in replacement_map:
                i += 1
        else:
            resolved_tokens.append(tokens[i])
            i += 1

    return " ".join(resolved_tokens)


if __name__ == "__main__":
    # text = "My brother, who lives in Paris, is visiting"
    #text = "My brother is visiting. He will stay for two days."
    text = 'I want to buy some apples. Which ones do you recommend?'

    # Assicurati di aver avviato il server CoreNLP
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
    with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref'],
        memory='4G',
        start_server=False
    ) as client:

        # Annotazione del testo
        annotation = client.annotate(text)

        # Coref resolution
        resolved = resolve_coref_text(annotation)

        print("Original text: ", text)
        print("Coref-resolved:", resolved)