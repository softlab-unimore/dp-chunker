from clause_splitters.base_splitter import BaseSplitter


class AclSplitter(BaseSplitter):
    """
    Extracts adjectival (infinitival) clauses.

    The noun that the acl modifies is included in the output text so that
    the extracted clause is self-contained.  The *noun* argument lets the
    orchestrator override which noun to attach (used for ``pobj VERB``
    redirects).
    """

    def split(self, doc, token, noun=None):
        """
        Extract the adjectival clause rooted at *token*.

        Args:
            doc:   spaCy Doc.
            token: The acl root token.
            noun:  Override the head noun.  Defaults to ``token.head``.

        Returns:
            A result dict.
        """
        if noun is None:
            noun = token.head

        clause_tokens = self._collect_clause_tokens(token, noun)
        acl_clause = self.build_clause_text(clause_tokens)
        consumed = [t for t in clause_tokens if t.i > noun.i]

        return {
            "type": "acl",
            "subordinate": acl_clause.strip(),
            "tokens": consumed,
        }

    def _collect_clause_tokens(self, token, noun) -> list:
        """Build the sorted, deduplicated token list for the acl clause."""
        nested_idxs = self.build_nested_idxs(
            token, {"advcl", "relcl", "acl", "conj", "ccomp", "parataxis"}
        )

        tokens = [noun]
        self.find_name_modifiers(tokens, noun)

        for t in token.subtree:
            if t.dep_ != "punct" and t.i not in nested_idxs:
                tokens.append(t)

        return sorted(set(tokens), key=lambda t: t.i)