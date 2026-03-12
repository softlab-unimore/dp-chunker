from splitters.base_splitter import BaseSplitter


class RelclSplitter(BaseSplitter):

    def split(self, doc, token):
        noun = token.head

        noun_mod_deps = {"det", "amod", "nummod", "poss", "compound"}
        noun_np = [noun]
        for ch in noun.children:
            if ch.dep_ in noun_mod_deps:
                noun_np.append(ch)
        noun_np = sorted(noun_np, key=lambda t: t.i)

        subtree_tokens = [t for t in token.subtree if t.dep_ != "punct"]

        rel_pron = None
        for ch in token.children:
            if ch.dep_ in {"nsubj", "nsubjpass"} and ch.text.lower() in {"who", "that", "which", "whom"}:
                rel_pron = ch
                break
        if rel_pron is None:
            for ch in noun.children:
                if ch.text.lower() in {"who", "that", "which", "whom"}:
                    rel_pron = ch
                    break

        rel_subj_heads = [
            t for t in subtree_tokens
            if t.dep_ in {"nsubj", "nsubjpass"} and t.text.lower() not in {"who", "that", "which", "whom"}
        ]
        rel_subj_with_mods = []
        for subj in rel_subj_heads:
            for ch in subj.children:
                if ch.dep_ in {"amod", "det"}:
                    rel_subj_with_mods.append(ch)
            rel_subj_with_mods.append(subj)
        rel_subj = sorted(rel_subj_with_mods, key=lambda t: t.i)

        rel_dobj = [t for t in subtree_tokens if t.dep_ in {"dobj", "obj"}]

        has_rel_pron = rel_pron is not None or any(
            t.text.lower() in {"that", "which", "whom", "who"}
            for t in subtree_tokens
        )

        excluded = set()
        excluded.update(t.i for t in rel_subj)
        excluded.update(t.i for t in rel_dobj)
        excluded.add(token.i)
        if rel_pron:
            excluded.add(rel_pron.i)

        other_tokens = [t for t in subtree_tokens if t.i not in excluded]

        # -------------------------------------------------------
        # CASE 1: relative pronoun is SUBJECT (who/which subject)
        # -------------------------------------------------------
        if rel_pron is not None and not rel_subj:
            clause_tokens = noun_np + [token] + other_tokens + rel_dobj
            for dobj in rel_dobj:
                self.find_name_modifiers(clause_tokens, dobj)

            seen = set()
            clause_ordered = []
            np_idxs = {t.i for t in noun_np}
            for t in sorted(clause_tokens, key=lambda t: (0 if t.i in np_idxs else 1, t.i)):
                if t.i not in seen:
                    clause_ordered.append(t)
                    seen.add(t.i)

            rel_clause = " ".join(t.text for t in clause_ordered)
            used_tokens = sorted([t for t in subtree_tokens if t.i > noun.i], key=lambda t: t.i)
            return {"type": "relcl", "subordinate": rel_clause.strip(), "tokens": used_tokens}

        # -------------------------------------------------------
        # CASE 2: explicit subject + explicit relative pronoun
        # -------------------------------------------------------
        elif rel_subj and has_rel_pron:
            rel_pron_as_obj = [
                t for t in rel_dobj
                if t.text.lower() in {"that", "which", "whom", "who"} or t.pos_ == "VERB"
            ]
            true_dobj = [t for t in rel_dobj if t not in rel_pron_as_obj]

            main_verb_idxs = set()
            for v in rel_pron_as_obj:
                if v.pos_ == "VERB":
                    main_verb_idxs.update(t.i for t in v.subtree)

            excluded.update(t.i for t in rel_pron_as_obj)
            excluded.update(main_verb_idxs)

            other_tokens = [t for t in subtree_tokens if t.i not in excluded]
            clause_tokens = rel_subj + [token] + other_tokens

            if not true_dobj and (rel_pron_as_obj or rel_pron is not None):
                clause_tokens += noun_np
            else:
                clause_tokens += true_dobj
                for dobj in true_dobj:
                    self.find_name_modifiers(clause_tokens, dobj)

            seen = set()
            clause_ordered = []
            for t in clause_tokens:
                if t.i not in seen:
                    clause_ordered.append(t)
                    seen.add(t.i)

            rel_clause = " ".join(t.text for t in clause_ordered)
            used_tokens = sorted(
                [t for t in subtree_tokens if t.i > noun.i and t.i not in main_verb_idxs],
                key=lambda t: t.i
            )
            return {"type": "relcl", "subordinate": rel_clause.strip(), "tokens": used_tokens}

        # -------------------------------------------------------
        # CASE 3: zero relative (no relative pronoun)
        # -------------------------------------------------------
        else:
            zero_subj = [t for t in subtree_tokens if t.dep_ in {"nsubj", "nsubjpass"}]

            if not zero_subj:
                clause_tokens = noun_np + other_tokens + [token]
            else:
                clause_tokens = zero_subj + [token] + other_tokens + noun_np
                for subj in zero_subj:
                    self.find_name_modifiers(clause_tokens, subj)

            seen = set()
            clause_ordered = []
            for t in clause_tokens:
                if t.i not in seen:
                    clause_ordered.append(t)
                    seen.add(t.i)

            rel_clause = " ".join(t.text for t in clause_ordered)
            used_tokens = sorted([t for t in subtree_tokens if t.i > noun.i], key=lambda t: t.i)
            return {"type": "relcl", "subordinate": rel_clause.strip(), "tokens": used_tokens}