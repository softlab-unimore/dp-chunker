from splitters.base_splitter import BaseSplitter


class ParataxisSplitter(BaseSplitter):
    """Extracts paratactic clauses and detects disguised parataxis."""

    PARATAXIS_PUNCT = {":", ";", ","}

    def split(self, doc, token):
        """
        Extract the paratactic clause rooted at *token*.

        Args:
            doc:   spaCy Doc.
            token: The parataxis root token.

        Returns:
            A result dict, or None if the clause contains no verb.
        """
        if not self.has_verb(token):
            return None

        nested_idxs = self.build_nested_idxs(
            token,
            {"advcl", "relcl", "acl", "conj", "ccomp", "parataxis"},
            include_cc=True,
        )

        clause_tokens = sorted(
            [
                t for t in token.subtree
                if t.dep_ not in {"punct", "mark", "cc"} and t.i not in nested_idxs
            ],
            key=lambda t: t.i,
        )

        if not clause_tokens:
            return None

        return {
            "type": "parataxis",
            "subordinate": self.build_clause_text(clause_tokens),
            "tokens": clause_tokens,
        }

    def is_disguised_parataxis(self, doc, token) -> bool:
        """
        Return True if *token* is a ccomp that is actually paratactic —
        i.e. it is separated from its syntactic head by a ``:`` or ``;``.

        The check covers two positions:
        1. Any punctuation token between *token* and its head.
        2. Direct punctuation children of either *token* or its head.
        """
        if token.dep_ != "ccomp":
            return False

        head = token.head
        left, right = (
            (token.i, head.i) if token.i < head.i else (head.i, token.i)
        )

        for t in doc[left:right]:
            if t.dep_ == "punct" and t.text in self.PARATAXIS_PUNCT:
                return True

        for t in list(token.children) + list(head.children):
            if t.dep_ == "punct" and t.text in self.PARATAXIS_PUNCT:
                return True

        return False