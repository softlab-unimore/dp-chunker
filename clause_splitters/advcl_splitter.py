from clause_splitters.base_splitter import BaseSplitter


class AdvclSplitter(BaseSplitter):
    """
    Extracts adverbial subordinate clauses.

    Subordinating advmod tokens (when, while, before …) are stripped from
    the clause text but are still marked as consumed so they cannot leak
    into other clauses.
    """

    SUBORDINATING_ADVMOD = {"when", "where", "while", "whenever", "wherever", "before", "after", "once"}

    def split(self, doc, token):
        """
        Extract the adverbial clause rooted at *token*.

        Args:
            doc:   spaCy Doc.
            token: The advcl root token.

        Returns:
            A result dict, or None if the clause contains no verb.
        """
        if not self.has_verb(token):
            return None

        nested_idxs = self.build_nested_idxs(token, {"relcl", "acl", "advcl"})

        clause_tokens = [
            t for t in token.subtree
            if t.dep_ not in {"mark", "punct"}
            and not (t.dep_ == "advmod" and t.text.lower() in self.SUBORDINATING_ADVMOD)
            and t.i not in nested_idxs
        ]

        clause_tokens = self._maybe_inject_implicit_subject(token, clause_tokens)
        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)

        subordinating_tokens = [
            t for t in token.subtree
            if t.dep_ == "advmod" and t.text.lower() in self.SUBORDINATING_ADVMOD
        ]
        used_tokens = sorted(set(clause_tokens + subordinating_tokens), key=lambda t: t.i)

        return {
            "type": "advcl",
            "subordinate": self.build_clause_text(clause_tokens),
            "tokens": used_tokens,
        }

    def _maybe_inject_implicit_subject(self, token, clause_tokens: list) -> list:
        """
        When the advcl has no explicit subject, attempt to recover one from
        the ``pobj`` of a preceding prepositional sibling.

        Example: "After the storm, *the city flooded* because the drains were blocked."
        Here "the city" is not syntactically inside the advcl, but can be
        inferred from context.
        """
        has_subj = any(t.dep_ in {"nsubj", "nsubjpass"} for t in clause_tokens)
        if has_subj:
            return clause_tokens

        for sib in reversed(list(token.head.children)):
            if sib.dep_ == "prep" and sib.i < token.i:
                for ch in sib.children:
                    if ch.dep_ == "pobj":
                        implicit_subj = [ch] + [
                            t for t in ch.children if t.dep_ in {"det", "amod"}
                        ]
                        return sorted(implicit_subj, key=lambda t: t.i) + clause_tokens
                break

        return clause_tokens