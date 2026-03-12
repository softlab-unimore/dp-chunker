from splitters.base_splitter import BaseSplitter


class AdvclSplitter(BaseSplitter):

    def split(self, doc, token):
        if not self.has_verb(token):
            return None

        nested_idxs = self.build_nested_idxs(token, {"relcl", "acl", "advcl"})

        clause_tokens = [
            t for t in token.subtree
            if not (t.dep_ in ["mark", "punct"] and t.head.i == token.i)
            and t.i not in nested_idxs
        ]
        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)

        return {"type": "advcl", "subordinate": self.build_clause_text(clause_tokens), "tokens": clause_tokens}