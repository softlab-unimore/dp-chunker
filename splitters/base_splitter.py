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

    def build_nested_idxs(self, token, deps, include_cc=False):
        """
        Collect indices of all tokens belonging to nested clauses of given dependency types.
        Optionally also collects direct 'cc' children of each nested clause root.
        """
        nested_idxs = set()
        for t in token.subtree:
            if t.dep_ in deps and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)
                if include_cc:
                    for ch in t.children:
                        if ch.dep_ == "cc":
                            nested_idxs.add(ch.i)
        return nested_idxs

    def collect_subtree_idxs(self, token, dep):
        """
        Collect all token indices in subtrees of direct children with a given dependency.
        Used in clause_splitter to exclude ccomp/advcl/relcl from predicates.
        """
        idxs = set()
        for ch in token.children:
            if ch.dep_ == dep:
                idxs.update(t.i for t in ch.subtree)
        return idxs

    def build_clause_text(self, tokens):
        """Sort tokens by position and join their text."""
        return " ".join(t.text for t in sorted(tokens, key=lambda t: t.i))

    def deduplicate_ordered(self, tokens):
        """Remove duplicate tokens preserving order (by first occurrence)."""
        seen = set()
        result = []
        for t in tokens:
            if t.i not in seen:
                result.append(t)
                seen.add(t.i)
        return result