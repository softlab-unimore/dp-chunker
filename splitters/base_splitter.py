class BaseSplitter:

    def __init__(self, nlp):
        self.nlp = nlp

    def has_verb(self, token):
        """Check if the subtree contains a verb (VERB or AUX)."""
        return any(t.pos_ in ["VERB", "AUX"] for t in token.subtree)

    def find_name_modifiers(self, clause_tokens, noun):
        """Find adjectives and determiners that modify the noun and add them to the clause tokens."""
        for child in noun.children:
            if child.dep_ in ["amod", "det"]:
                clause_tokens.append(child)
        return clause_tokens