from splitters.base_splitter import BaseSplitter


class AclSplitter(BaseSplitter):

    def split(self, doc, token, noun=None):
        if noun is None:
            noun = token.head

        clause_tokens = [noun]
        self.find_name_modifiers(clause_tokens, noun)

        nested_idxs = self.build_nested_idxs(token, {"advcl", "relcl", "acl", "conj", "ccomp", "parataxis"})

        for t in token.subtree:
            if t.dep_ == "punct" or t.i in nested_idxs:
                continue
            clause_tokens.append(t)

        clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
        acl_clause = self.build_clause_text(clause_tokens)
        clause_tokens = [t for t in clause_tokens if t.i > noun.i]

        return {"type": "acl", "subordinate": acl_clause.strip(), "tokens": clause_tokens}