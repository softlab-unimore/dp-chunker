from splitters.base_splitter import BaseSplitter


class AdvclSplitter(BaseSplitter):

    def split(self, doc, token):
        if not self.has_verb(token):
            return None

        nested_idxs = set()
        for t in token.subtree:
            if t.dep_ in {"relcl", "acl", "advcl"} and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)

        clause_tokens = [
            t for t in token.subtree
            if not (t.dep_ in ["mark", "punct"] and t.head.i == token.i)
            and t.i not in nested_idxs
        ]
        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)
        adv_clause = " ".join(t.text for t in clause_tokens)

        return {"type": "advcl", "subordinate": adv_clause.strip(), "tokens": clause_tokens}