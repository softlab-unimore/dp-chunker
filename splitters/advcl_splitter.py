from splitters.base_splitter import BaseSplitter


class AdvclSplitter(BaseSplitter):

    SUBORDINATING_ADVMOD = {"when", "where", "while", "whenever", "wherever", "before", "after", "once"}

    def split(self, doc, token):
        if not self.has_verb(token):
            return None

        nested_idxs = self.build_nested_idxs(token, {"relcl", "acl", "advcl"})

        clause_tokens = [
            t for t in token.subtree
            if t.dep_ not in {"mark", "punct"}
               and not (t.dep_ == "advmod" and t.text.lower() in self.SUBORDINATING_ADVMOD)
               and t.i not in nested_idxs
        ]

        # Se non c'è soggetto esplicito, cerca il pobj del prep fratello precedente
        has_subj = any(t.dep_ in {"nsubj", "nsubjpass"} for t in clause_tokens)
        if not has_subj:
            siblings = list(token.head.children)
            for sib in reversed(siblings):
                if sib.dep_ == "prep" and sib.i < token.i:
                    for ch in sib.children:
                        if ch.dep_ == "pobj":
                            # Aggiungi pobj e i suoi modificatori come soggetto implicito
                            implicit_subj = [ch] + [
                                t for t in ch.children
                                if t.dep_ in {"det", "amod"}
                            ]
                            clause_tokens = sorted(implicit_subj, key=lambda t: t.i) + clause_tokens
                    break

        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)
        return {"type": "advcl", "subordinate": self.build_clause_text(clause_tokens), "tokens": clause_tokens}