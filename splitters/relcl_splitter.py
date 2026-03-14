"""
relcl_splitter.py
-----------------
Handles relative clause modifiers (dep=relcl).

Three structural cases are distinguished based on how the relative
pronoun and the clause subject are expressed:

- **Case 1** – relative pronoun as subject (``who``/``which`` subject):
  *"The woman* ***who called*** *was his sister."*
- **Case 2** – explicit subject + explicit relative pronoun (object gap):
  *"The book* ***that John wrote*** *became famous."*
- **Case 3** – zero relative (no overt relative pronoun):
  *"The man* ***I met*** *was a doctor."*
"""

from splitters.base_splitter import BaseSplitter


class RelclSplitter(BaseSplitter):
    """Extracts relative clause modifiers."""

    def split(self, doc, token):
        """
        Extract the relative clause rooted at *token*.

        Args:
            doc:   spaCy Doc.
            token: The relcl root token.

        Returns:
            A result dict.
        """
        noun = token.head
        noun_np = self.collect_noun_np(noun)

        nested_idxs = self.build_nested_idxs(token, {"advcl", "relcl", "acl", "ccomp"})

        subtree_tokens = [
            t for t in token.subtree
            if t.dep_ != "punct" and t.i not in nested_idxs
        ]

        rel_pron = self._find_relative_pronoun(token, noun)
        rel_subj = self._find_explicit_subject(subtree_tokens)
        rel_dobj = [t for t in subtree_tokens if t.dep_ in {"dobj", "obj"}]
        has_rel_pron = rel_pron is not None or any(
            self.is_relative_pronoun(t) for t in subtree_tokens
        )

        if rel_pron is not None and not rel_subj:
            return self._case1_pron_as_subject(
                token, noun, noun_np, subtree_tokens, nested_idxs,
                rel_pron, rel_dobj,
            )
        elif rel_subj and has_rel_pron:
            return self._case2_explicit_subject(
                token, noun, noun_np, subtree_tokens, nested_idxs,
                rel_pron, rel_subj, rel_dobj, has_rel_pron,
            )
        else:
            return self._case3_zero_relative(
                token, noun, noun_np, subtree_tokens, rel_subj,
            )

    def _case1_pron_as_subject(
        self, token, noun, noun_np, subtree_tokens, nested_idxs,
        rel_pron, rel_dobj,
    ):
        npadvmod_tokens = self._collect_npadvmod(token, noun)

        np_idxs = {t.i for t in noun_np}
        other = self._other_tokens(
            subtree_tokens, nested_idxs,
            excluded={t.i for t in rel_dobj} | {token.i} | {rel_pron.i},
        )

        clause_tokens = sorted(
            noun_np + [token] + other + rel_dobj + npadvmod_tokens,
            key=lambda t: (0 if t.i in np_idxs else 1, t.i),
        )
        for dobj in rel_dobj:
            self.find_name_modifiers(clause_tokens, dobj)

        used = sorted(
            [t for t in subtree_tokens if t.i > noun.i] + npadvmod_tokens,
            key=lambda t: t.i,
        )
        return self._make_result(clause_tokens, used)

    def _case2_explicit_subject(
        self, token, noun, noun_np, subtree_tokens, nested_idxs,
        rel_pron, rel_subj, rel_dobj, has_rel_pron,
    ):
        """
        Case 2 – the relative clause has an explicit subject and a relative
        pronoun filling the object gap.

        Reconstructed pattern:
            <explicit subject> <relcl verb> [other] <noun NP>
        """
        rel_pron_as_obj = [
            t for t in rel_dobj
            if self.is_relative_pronoun(t) or t.pos_ == "VERB"
        ]
        true_dobj = [t for t in rel_dobj if t not in rel_pron_as_obj]

        main_verb_idxs: set = set()
        for v in rel_pron_as_obj:
            if v.pos_ == "VERB":
                main_verb_idxs.update(t.i for t in v.subtree)

        excluded = (
            {t.i for t in rel_subj}
            | {t.i for t in rel_dobj}
            | {token.i}
            | ({rel_pron.i} if rel_pron else set())
            | {t.i for t in rel_pron_as_obj}
            | main_verb_idxs
        )
        other = self._other_tokens(subtree_tokens, nested_idxs, excluded=excluded)
        clause_tokens = rel_subj + [token] + other

        if not true_dobj and (rel_pron_as_obj or rel_pron is not None):
            clause_tokens += noun_np
        else:
            clause_tokens += true_dobj
            for dobj in true_dobj:
                self.find_name_modifiers(clause_tokens, dobj)

        used = sorted(
            [t for t in subtree_tokens if t.i > noun.i and t.i not in main_verb_idxs],
            key=lambda t: t.i,
        )
        return self._make_result(clause_tokens, used, preserve_order=True)

    def _case3_zero_relative(self, token, noun, noun_np, subtree_tokens, rel_subj):
        """
        Case 3 – zero relative clause (no overt relative pronoun).

        Reconstructed pattern (no explicit subject):
            <noun NP> [other] <relcl verb>
        Reconstructed pattern (with zero subject):
            <zero subject> <relcl verb> [other] <noun NP>
        """
        zero_subj = [t for t in subtree_tokens if t.dep_ in {"nsubj", "nsubjpass"}]
        other = self._other_tokens(
            subtree_tokens, set(),
            excluded={token.i} | {t.i for t in zero_subj},
        )

        if not zero_subj:
            clause_tokens = noun_np + other + [token]
        else:
            clause_tokens = zero_subj + [token] + other + noun_np
            for subj in zero_subj:
                self.find_name_modifiers(clause_tokens, subj)

        used = sorted([t for t in subtree_tokens if t.i > noun.i], key=lambda t: t.i)
        return self._make_result(clause_tokens, used, preserve_order=True)

    def _find_relative_pronoun(self, token, noun):
        """
        Return the relative pronoun token for this relcl, or None.

        First searches among the children of *token* (nsubj/nsubjpass
        that are relative pronouns), then falls back to children of *noun*.
        """
        rel_pron = next(
            (
                ch for ch in token.children
                if ch.dep_ in {"nsubj", "nsubjpass"} and self.is_relative_pronoun(ch)
            ),
            None,
        )
        if rel_pron is None:
            rel_pron = next(
                (ch for ch in noun.children if self.is_relative_pronoun(ch)),
                None,
            )
        return rel_pron

    def _find_explicit_subject(self, subtree_tokens) -> list:
        """
        Return the explicit (non-pronoun) subject tokens of the relative clause,
        including their det/amod modifiers, sorted by position.
        """
        subj_heads = [
            t for t in subtree_tokens
            if t.dep_ in {"nsubj", "nsubjpass"} and not self.is_relative_pronoun(t)
        ]
        modifiers = [
            ch for h in subj_heads for ch in h.children if ch.dep_ in {"amod", "det"}
        ]
        return sorted(modifiers + subj_heads, key=lambda t: t.i)

    def _collect_npadvmod(self, token, noun) -> list:
        """
        Collect ``npadvmod`` tokens that spaCy attaches to the main verb but
        logically belong to the relative clause (e.g. "every day" in
        "She is a person who works every day").
        """
        main_verb = noun.head if noun.head.pos_ in {"VERB", "AUX"} else None
        if main_verb is None:
            return []

        npadvmod_tokens = [
            t for t in main_verb.children
            if t.dep_ == "npadvmod" and token.i < t.i < main_verb.i
        ]
        result = []
        for t in npadvmod_tokens:
            result.extend(t.subtree)
        return result

    def _other_tokens(self, subtree_tokens, nested_idxs, excluded: set) -> list:
        """
        Return all subtree tokens that are not in *excluded*, not in
        *nested_idxs*, and not markers.
        """
        return [
            t for t in subtree_tokens
            if t.i not in excluded
            and t.i not in nested_idxs
            and t.dep_ != "mark"
        ]

    def _make_result(self, clause_tokens, used_tokens, preserve_order: bool = False) -> dict:
        """Build the standard result dict for a relative clause."""
        ordered = self.deduplicate_ordered(clause_tokens)
        text = (
            " ".join(t.text for t in ordered)
            if preserve_order
            else self.build_clause_text(ordered)
        )
        return {
            "type": "relcl",
            "subordinate": text.strip(),
            "tokens": used_tokens,
        }