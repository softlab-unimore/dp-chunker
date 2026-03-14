
class BaseSplitter:
    """
    Base class for all clause splitters.

    Subclasses implement a ``split(doc, token, **kwargs)`` method that
    receives a spaCy Doc and a dependency-arc root token, and returns a
    result dict with keys:
        - ``type``        (str)  — dependency label of the extracted clause
        - ``subordinate`` (str)  — reconstructed clause text
        - ``tokens``      (list) — spaCy tokens consumed by this clause
                                   (used by the orchestrator to mark them as
                                   unavailable for further processing)
    """

    def __init__(self, nlp):
        self.nlp = nlp

    def has_verb(self, token) -> bool:
        """Return True if the token's subtree contains at least one VERB or AUX."""
        return any(t.pos_ in {"VERB", "AUX"} for t in token.subtree)

    def is_relative_pronoun(self, token) -> bool:
        """Return True if the token is a relative pronoun (who/that/which/whom)."""
        return token.text.lower() in {"who", "that", "which", "whom"}

    def find_name_modifiers(self, clause_tokens: list, noun) -> list:
        """
        Append direct adjectival and determiner modifiers of *noun* to
        *clause_tokens* in-place and return the list.
        """
        for child in noun.children:
            if child.dep_ in {"amod", "det"}:
                clause_tokens.append(child)
        return clause_tokens

    def collect_noun_np(self, noun) -> list:
        """
        Return the full noun phrase for *noun*: the noun itself plus its
        det, amod, nummod, poss, and compound dependents, sorted by position.
        """
        modifiers = [
            ch for ch in noun.children
            if ch.dep_ in {"det", "amod", "nummod", "poss", "compound"}
        ]
        return sorted([noun] + modifiers, key=lambda t: t.i)

    def build_nested_idxs(self, token, deps: set, include_cc: bool = False) -> set:
        """
        Collect the token indices of all tokens belonging to nested clauses
        whose dependency label is in *deps*.

        Args:
            token:      Root of the subtree to inspect.
            deps:       Set of dependency labels that mark nested clause roots.
            include_cc: If True, also include direct ``cc`` children of each
                        nested clause root.

        Returns:
            A set of integer token indices.
        """
        nested_idxs: set = set()
        for t in token.subtree:
            if t.dep_ in deps and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)
                if include_cc:
                    for ch in t.children:
                        if ch.dep_ == "cc":
                            nested_idxs.add(ch.i)
        return nested_idxs

    def collect_subtree_idxs(self, token, dep: str) -> set:
        """
        Return the union of subtree indices for every direct child of *token*
        whose dependency label equals *dep*.

        Used by the orchestrator to exclude embedded clauses (ccomp, advcl …)
        when reconstructing a predicate.
        """
        idxs: set = set()
        for ch in token.children:
            if ch.dep_ == dep:
                idxs.update(t.i for t in ch.subtree)
        return idxs

    def build_clause_text(self, tokens) -> str:
        """
        Sort *tokens* by their position in the original document and join
        their surface forms with a single space.
        """
        return " ".join(t.text for t in sorted(tokens, key=lambda t: t.i))

    def deduplicate_ordered(self, tokens) -> list:
        """
        Remove duplicate tokens from *tokens* while preserving the order of
        first occurrence (based on token index).
        """
        seen: set = set()
        result = []
        for t in tokens:
            if t.i not in seen:
                result.append(t)
                seen.add(t.i)
        return result