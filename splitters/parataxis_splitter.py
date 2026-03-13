from splitters.base_splitter import BaseSplitter


class ParataxisSplitter(BaseSplitter):

    # Punctuation that signals a parataxis boundary
    PARATAXIS_PUNCT = {":", ";", ","}

    def split(self, doc, token):
        if not self.has_verb(token):
            return None

        nested_idxs = self.build_nested_idxs(
            token, {"advcl", "relcl", "acl", "conj", "ccomp", "parataxis"}, include_cc=True
        )

        clause_tokens = sorted(
            [t for t in token.subtree
             if t.dep_ not in {"punct", "mark", "cc"} and t.i not in nested_idxs],
            key=lambda t: t.i
        )

        if not clause_tokens:
            return None

        return {
            "type": "parataxis",
            "subordinate": self.build_clause_text(clause_tokens),
            "tokens": clause_tokens
        }

    def is_disguised_parataxis(self, doc, token):
        """
        Detect ccomp tokens that are actually parataxis:
        the clause is separated from its head by ':' or ';'.
        """
        if token.dep_ != "ccomp":
            return False

        # Check for : or ; between token and its head
        head = token.head
        left, right = (token.i, head.i) if token.i < head.i else (head.i, token.i)
        for t in doc[left:right]:
            if t.dep_ == "punct" and t.text in self.PARATAXIS_PUNCT:
                return True

        # Also check punct children of token or head
        for t in list(token.children) + list(head.children):
            if t.dep_ == "punct" and t.text in self.PARATAXIS_PUNCT:
                return True

        return False