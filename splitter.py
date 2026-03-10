import spacy

class ClauseSplitter:

    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)
        self.splitters = {
            "advcl": self.split_advcl,
            "acl": self.split_acl,
            "relcl": self.split_relcl,
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

    def split_relcl(self, doc, token):
        noun = token.head

        noun_mod_deps = {"det", "amod", "nummod", "poss", "compound"}
        noun_np = [noun]
        for ch in noun.children:
            if ch.dep_ in noun_mod_deps:
                noun_np.append(ch)
        noun_np = sorted(noun_np, key=lambda t: t.i)

        rel_pron = None
        for ch in noun.children:
            if ch.text.lower() in {"who", "that", "which", "whom"}:
                rel_pron = ch
                break

        subtree_tokens = [t for t in token.subtree if t.dep_ != "punct"]

        rel_subj = [t for t in subtree_tokens if t.dep_ in {"nsubj", "nsubjpass"}]
        rel_dobj = [t for t in subtree_tokens if t.dep_ in {"dobj", "obj", "pobj"}]

        other_tokens = [t for t in subtree_tokens if t not in rel_subj + rel_dobj and t != token]

        clause_tokens = []

        if rel_subj:
            clause_tokens.extend(sorted(rel_subj, key=lambda t: t.i))
            clause_tokens.append(token)  # verbo
            if other_tokens:
                clause_tokens.extend(sorted(other_tokens, key=lambda t: t.i))

            if not rel_dobj:
                clause_tokens.extend(noun_np)
            else:
                clause_tokens.extend(sorted(rel_dobj, key=lambda t: t.i))
                for dobj in rel_dobj:
                    clause_tokens = self.find_name_modifiers(clause_tokens, dobj)
        else:
            clause_tokens.extend(noun_np)
            clause_tokens.append(token)
            if other_tokens:
                clause_tokens.extend(sorted(other_tokens, key=lambda t: t.i))
            if rel_dobj:
                clause_tokens.extend(sorted(rel_dobj, key=lambda t: t.i))
                for dobj in rel_dobj:
                    clause_tokens = self.find_name_modifiers(clause_tokens, dobj)

        seen = set()
        clause_ordered = []
        for t in clause_tokens:
            if t.i not in seen:
                clause_ordered.append(t)
                seen.add(t.i)

        rel_clause = " ".join(t.text for t in clause_ordered)

        used_tokens = []
        noun_np_idxs = {t.i for t in noun_np}
        for t in subtree_tokens:
            if t.i not in noun_np_idxs:
                used_tokens.append(t)
        if rel_pron is not None and rel_pron.i not in {t.i for t in used_tokens}:
            used_tokens.append(rel_pron)

        used_tokens = sorted([t for t in used_tokens if t.i > noun.i], key=lambda t: t.i)

        return {"type": "relcl", "subordinate": rel_clause.strip(), "tokens": used_tokens}

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

        #relcl con oggetto implicito
        # "I saw the man you love.",
        # "She met the author everyone admires.",
        # "We bought the house they built.",
        # "He found the book she recommended.",

        #relcl con pronome relativo soggetto
        "The woman who looked happy danced.",
        "The boy who won the race celebrated.",
        "The scientist who discovered the cure received an award.",
        "The student who solved the problem smiled.",

        # relcl con pronome relativo oggetto
        "The book that John wrote won a prize.",
        "The movie that we watched was amazing.",
        "The car that she bought is very fast.",
        "The song that they played became famous.",

        # relcl con which
        "The house which Jack built collapsed.",
        "The computer which she repaired works perfectly.",
        "The painting which the museum bought is valuable.",

        # relcl senza pronome relativo (zero relative)
        "The man I met yesterday was friendly.",
        "The car we rented broke down.",
        "The movie we saw last night was boring.",

    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print("\nSentence:", s)
        for split in splits:
            print(" -", split)