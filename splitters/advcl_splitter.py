from splitters.base_splitter import BaseSplitter


class AdvclSplitter(BaseSplitter):

    def split(self, doc, token):
        if not self.has_verb(token):
            return None

        clause_tokens = list(token.subtree)
        clause_tokens = [t for t in clause_tokens if t.dep_ not in ["advmod", "mark", "punct"]]
        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)
        adv_clause = " ".join(t.text for t in clause_tokens)

        return {"type": "advcl", "subordinate": adv_clause.strip(), "tokens": clause_tokens}