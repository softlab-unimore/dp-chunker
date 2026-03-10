import spacy

class ClauseSplitter:

    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)
        self.splitters = {
            "advcl": self.split_advcl,
            "acl": self.split_acl,
        }

    def has_verb(self, token):
        """Check if the subtree contains a verb (VERB or AUX)."""
        return any(t.pos_ in ["VERB", "AUX"] for t in token.subtree)

    def find_name_modifiers(self, clause_tokens, noun):
        """Find adjectives and determiners that modify the noun and add them to the clause tokens."""
        for child in noun.children:
            if child.dep_ in ["amod", "det"]:
                clause_tokens.append(child)
        return clause_tokens

    def split_advcl(self, doc, token):

        if not self.has_verb(token):
            return None

        clause_tokens = list(token.subtree)
        clause_tokens = [t for t in clause_tokens if t.dep_ not in ["advmod", "mark", "punct"]]
        clause_tokens = sorted(clause_tokens, key=lambda t: t.i)
        adv_clause = " ".join(t.text for t in clause_tokens)

        return {"type": "advcl", "subordinate": adv_clause.strip(), "tokens": clause_tokens}

    def split_acl(self, doc, token):
        noun = token.head
        clause_tokens = [noun]
        clause_tokens = self.find_name_modifiers(clause_tokens, noun)

        for t in token.subtree:
            if t.dep_ != "punct":
                clause_tokens.append(t)

        clause_tokens = sorted(set(clause_tokens), key=lambda t: t.i)
        acl_clause = " ".join(t.text for t in clause_tokens)
        clause_tokens = [t for t in clause_tokens if t.i > noun.i]

        return {"type": "acl", "subordinate": acl_clause.strip(), "tokens": clause_tokens}

    # -----------------------------
    # MAIN DISPATCH METHOD
    # -----------------------------
    def split_sentence(self, sentence):
        doc = self.nlp(sentence)
        splits = []
        used_tokens = set()

        for token in doc:
            if token.dep_ in self.splitters and token.i not in used_tokens:
                split_result = self.splitters[token.dep_](doc, token)
                if split_result:
                    splits.append({
                        "type": split_result["type"],
                        "subordinate": split_result["subordinate"]
                    })
                    used_tokens.update(t.i for t in split_result["tokens"])

        main_tokens = [t for t in doc if t.i not in used_tokens and t.dep_ not in ["punct", "mark"]]
        if main_tokens:
            main_text = " ".join(t.text for t in sorted(main_tokens, key=lambda t: t.i))
            splits.insert(0, {"type": "main", "subordinate": main_text.strip()})

        return [s["subordinate"] for s in splits]


if __name__=='__main__':

    sentences = [

    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print("\nSentence:", s)
        for split in splits:
            print(" -", split)