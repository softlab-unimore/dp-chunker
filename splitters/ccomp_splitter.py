from splitters.base_splitter import BaseSplitter


class CcompSplitter(BaseSplitter):
    """
    Extracts clausal complement (ccomp) clauses.

    Nested subordinate clauses, direct ``cc`` children, and ``to + pobj VERB``
    constructions are all excluded from the output so they can be processed
    independently.
    """

    def split(self, doc, token):
        """
        Extract the ccomp clause rooted at *token*.

        Args:
            doc:   spaCy Doc.
            token: The ccomp root token.

        Returns:
            A result dict, or None if the token is an xcomp.
        """
        if token.dep_ == "xcomp":
            return None

        excluded = self._build_excluded_idxs(token)

        clause_tokens = sorted(
            [
                t for t in token.subtree
                if t.dep_ not in {"punct", "mark"} and t.i not in excluded
            ],
            key=lambda t: t.i,
        )

        return {
            "type": "ccomp",
            "subordinate": self.build_clause_text(clause_tokens),
            "tokens": clause_tokens,
        }

    def _build_excluded_idxs(self, token) -> set:
        """
        Build the set of token indices that must be excluded from this clause.

        Excludes:
        - Subtrees of nested clauses (advcl, relcl, acl, conj, ccomp, parataxis).
        - Direct ``cc`` children of the ccomp root.
        - ``to + pobj VERB`` constructions (treated as acl by the orchestrator).
        """
        excluded = self.build_nested_idxs(
            token,
            {"advcl", "relcl", "acl", "conj", "ccomp", "parataxis"},
            include_cc=True,
        )

        for ch in token.children:
            if ch.dep_ == "cc":
                excluded.add(ch.i)

        excluded |= self._collect_to_pobj_verb_idxs(token)
        return excluded

    def _collect_to_pobj_verb_idxs(self, token) -> set:
        """
        Identify and return indices for ``to + pobj VERB`` constructions,
        which are handled as acl clauses by the orchestrator and must not
        be included in the ccomp text.
        """
        idxs: set = set()
        for t in token.subtree:
            if t.dep_ == "pobj" and t.pos_ == "VERB":
                if t.head.dep_ == "prep" and t.head.text.lower() == "to":
                    idxs.add(t.head.i)
                    idxs.update(st.i for st in t.subtree)
            if t.dep_ == "prep" and t.text.lower() == "to":
                for ch in t.children:
                    if ch.dep_ == "pobj" and ch.pos_ == "VERB":
                        idxs.add(t.i)
                        idxs.update(st.i for st in ch.subtree)
        return idxs