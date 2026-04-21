from clause_splitters.base_splitter import BaseSplitter


class ConjSplitter(BaseSplitter):
    """Extracts coordinated verbal clauses."""

    def split(self, doc, token):
        """
        Extract the verbal clause rooted at *token*.

        Args:
            doc:   spaCy Doc.
            token: The conj root token.

        Returns:
            A result dict, or None for nominal conj (Case C).
        """
        head = token.head

        if token.pos_ in {"NOUN", "ADJ", "PROPN"} and head.pos_ in {"NOUN", "ADJ", "PROPN"}:
            return None

        has_own_subject = any(ch.dep_ in {"nsubj", "nsubjpass"} for ch in token.children)

        if has_own_subject:
            return self._split_with_own_subject(token)
        else:
            return self._split_with_inherited_subject(token)

    def _split_with_own_subject(self, token):
        """Case B: the conj verb has its own explicit subject."""
        nested_idxs = self.build_nested_idxs(
            token, {"ccomp", "relcl", "acl", "advcl", "parataxis"}, include_cc=True
        )
        clause_tokens = sorted(
            [
                t for t in token.subtree
                if t.dep_ not in {"punct", "cc"} and t.i not in nested_idxs
            ],
            key=lambda t: t.i,
        )
        return {
            "type": "conj",
            "subordinate": self.build_clause_text(clause_tokens),
            "tokens": clause_tokens,
        }

    def _split_with_inherited_subject(self, token):
        """Case A: the conj verb inherits its subject from the chain root."""
        chain_root = token
        while chain_root.dep_ == "conj":
            chain_root = chain_root.head

        inherited_subj = [
            t for t in chain_root.children if t.dep_ in {"nsubj", "nsubjpass"}
        ]

        nested_idxs = self.build_nested_idxs(
            token, {"conj", "ccomp", "relcl", "acl", "advcl"}, include_cc=True
        )
        clause_tokens = sorted(
            {t for t in token.subtree if t.dep_ not in {"punct", "cc"} and t.i not in nested_idxs},
            key=lambda t: t.i,
        )
        clause_text = self.build_clause_text(inherited_subj + clause_tokens)

        return {
            "type": "conj",
            "subordinate": clause_text.strip(),
            "tokens": clause_tokens,
        }