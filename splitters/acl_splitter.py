from splitters.base_splitter import BaseSplitter


class AclSplitter(BaseSplitter):

    def split(self, doc, token):
        noun = token.head
        clause_tokens = [noun]
        clause_tokens = self.find_name_modifiers(clause_tokens, noun)

        nested_idxs = set()
        for t in token.subtree:
            if t.dep_ in {"advcl", "relcl"} and t.i != token.i:
                nested_idxs.update(st.i for st in t.subtree)

        for t in token.subtree:
            if t.dep_ == "punct":
                continue
            if t.i in nested_idxs:
                continue
            clause_tokens.append(t)

        clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
        acl_clause = " ".join(t.text for t in clause_tokens)
        clause_tokens = [t for t in clause_tokens if t.i > noun.i]

        return {"type": "acl", "subordinate": acl_clause.strip(), "tokens": clause_tokens}