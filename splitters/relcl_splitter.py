from splitters.base_splitter import BaseSplitter


class RelclSplitter(BaseSplitter):

    def split(self, doc, token):
        noun = token.head

        noun_np = [noun] + [
            ch for ch in noun.children
            if ch.dep_ in {"det", "amod", "nummod", "poss", "compound"}
        ]
        noun_np = sorted(noun_np, key=lambda t: t.i)

        nested_idxs = self.build_nested_idxs(token, {"advcl", "relcl", "acl", "ccomp"})

        subtree_tokens = [
            t for t in token.subtree
            if t.dep_ != "punct" and t.i not in nested_idxs
        ]

        # Find relative pronoun
        rel_pron = next(
            (ch for ch in token.children
             if ch.dep_ in {"nsubj", "nsubjpass"}
             and ch.text.lower() in {"who", "that", "which", "whom"}),
            None
        )
        if rel_pron is None:
            rel_pron = next(
                (ch for ch in noun.children
                 if ch.text.lower() in {"who", "that", "which", "whom"}),
                None
            )

        # Find explicit subject (non-pronoun)
        rel_subj_heads = [
            t for t in subtree_tokens
            if t.dep_ in {"nsubj", "nsubjpass"}
            and t.text.lower() not in {"who", "that", "which", "whom"}
        ]
        rel_subj = sorted(
            [ch for subj in rel_subj_heads for ch in subj.children if ch.dep_ in {"amod", "det"}]
            + rel_subj_heads,
            key=lambda t: t.i
        )

        rel_dobj = [t for t in subtree_tokens if t.dep_ in {"dobj", "obj"}]

        has_rel_pron = rel_pron is not None or any(
            t.text.lower() in {"that", "which", "whom", "who"} for t in subtree_tokens
        )

        excluded = (
            {t.i for t in rel_subj}
            | {t.i for t in rel_dobj}
            | {token.i}
            | ({rel_pron.i} if rel_pron else set())
        )

        def other(extra_excluded=None):
            ex = excluded | (extra_excluded or set())
            return [
                t for t in subtree_tokens
                if t.i not in ex
                and t.i not in nested_idxs
                and t.dep_ != "mark"
            ]

        def make_result(clause_tokens, used_tokens, preserve_order=False):
            ordered = self.deduplicate_ordered(clause_tokens)
            text = " ".join(t.text for t in ordered) if preserve_order else self.build_clause_text(ordered)
            return {"type": "relcl", "subordinate": text.strip(), "tokens": used_tokens}

        # -------------------------------------------------------
        # CASE 1: relative pronoun is SUBJECT (who/which subject)
        # -------------------------------------------------------
        if rel_pron is not None and not rel_subj:
            # Include npadvmod tokens that spaCy attaches to the main verb
            # but logically belong to the relcl (positioned between relcl verb and main verb)
            main_verb = noun.head if noun.head.pos_ in {"VERB", "AUX"} else None
            npadvmod_tokens = []
            if main_verb is not None:
                npadvmod_tokens = [
                    t for t in main_verb.children
                    if t.dep_ == "npadvmod"
                    and t.i > token.i
                    and t.i < main_verb.i
                ]
                # Include their subtrees (e.g. "every" det of "day")
                npadvmod_full = []
                for t in npadvmod_tokens:
                    npadvmod_full.extend(t.subtree)
                npadvmod_tokens = npadvmod_full

            np_idxs = {t.i for t in noun_np}
            clause_tokens = sorted(
                noun_np + [token] + other() + rel_dobj + npadvmod_tokens,
                key=lambda t: (0 if t.i in np_idxs else 1, t.i)
            )
            for dobj in rel_dobj:
                self.find_name_modifiers(clause_tokens, dobj)
            used = sorted(
                [t for t in subtree_tokens if t.i > noun.i] + npadvmod_tokens,
                key=lambda t: t.i
            )
            return make_result(clause_tokens, used)

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

            extra_excl = {t.i for t in rel_pron_as_obj} | main_verb_idxs
            clause_tokens = rel_subj + [token] + other(extra_excl)

            if not true_dobj and (rel_pron_as_obj or rel_pron is not None):
                clause_tokens += noun_np
            else:
                clause_tokens += true_dobj
                for dobj in true_dobj:
                    self.find_name_modifiers(clause_tokens, dobj)

            used = sorted(
                [t for t in subtree_tokens if t.i > noun.i and t.i not in main_verb_idxs],
                key=lambda t: t.i
            )
            return make_result(clause_tokens, used, preserve_order=True)

        # -------------------------------------------------------
        # CASE 3: zero relative (no relative pronoun)
        # -------------------------------------------------------
        else:
            zero_subj = [t for t in subtree_tokens if t.dep_ in {"nsubj", "nsubjpass"}]

            if not zero_subj:
                clause_tokens = noun_np + other() + [token]
            else:
                clause_tokens = zero_subj + [token] + other() + noun_np
                for subj in zero_subj:
                    self.find_name_modifiers(clause_tokens, subj)

            used = sorted([t for t in subtree_tokens if t.i > noun.i], key=lambda t: t.i)
            return make_result(clause_tokens, used, preserve_order=True)