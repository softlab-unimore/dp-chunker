import spacy

from splitters.base_splitter import BaseSplitter
from splitters.advcl_splitter import AdvclSplitter
from splitters.acl_splitter import AclSplitter
from splitters.relcl_splitter import RelclSplitter
from splitters.conj_splitter import ConjSplitter
from splitters.ccomp_splitter import CcompSplitter
from splitters.parataxis_splitter import ParataxisSplitter


class ClauseSplitter(BaseSplitter):
    """
    Sentence-level clause splitter.

    Usage::

        splitter = ClauseSplitter()
        clauses = splitter.split_sentence("She left because she was tired.")
        # ["She left", "she was tired"]
    """

    # All clause types that can be individually enabled/disabled.
    ALL_SPLIT_TYPES = {"advcl", "acl", "relcl", "conj", "ccomp", "parataxis", "nominal_conj"}

    def __init__(self, model: str = "en_core_web_lg", enabled_splits: set | None = None):
        """
        Args:
            model:          spaCy model name.
            enabled_splits: Set of clause types to activate.  Accepted values:
                            ``advcl``, ``acl``, ``relcl``, ``conj``, ``ccomp``,
                            ``parataxis``, ``nominal_conj``.
                            Pass ``None`` (default) to enable all types.
        """
        self.nlp = spacy.load(model)
        self.advcl_splitter     = AdvclSplitter(self.nlp)
        self.acl_splitter       = AclSplitter(self.nlp)
        self.relcl_splitter     = RelclSplitter(self.nlp)
        self.conj_splitter      = ConjSplitter(self.nlp)
        self.ccomp_splitter     = CcompSplitter(self.nlp)
        self.parataxis_splitter = ParataxisSplitter(self.nlp)

        self.enabled_splits: set = (
            self.ALL_SPLIT_TYPES if enabled_splits is None else set(enabled_splits)
        )

        all_splitters = {
            "advcl":     lambda doc, token: self.advcl_splitter.split(doc, token),
            "acl":       lambda doc, token: self.acl_splitter.split(doc, token),
            "relcl":     lambda doc, token: self.relcl_splitter.split(doc, token),
            "conj":      lambda doc, token: self.conj_splitter.split(doc, token),
            "ccomp":     lambda doc, token: self.ccomp_splitter.split(doc, token),
            "parataxis": lambda doc, token: self.parataxis_splitter.split(doc, token),
            "pobj":      self._split_pobj_verb,
        }
        # Keep only the dep-label splitters that are enabled.
        # "pobj" is a virtual label that proxies "acl", so it follows "acl".
        self.splitters = {
            dep: fn for dep, fn in all_splitters.items()
            if dep == "pobj" and "acl" in self.enabled_splits
            or dep != "pobj" and dep in self.enabled_splits
        }

    def _split_pobj_verb(self, doc, token):
        """
        Handle ``to + pobj VERB`` constructions as acl clauses.

        When a VERB token appears as ``pobj`` of a ``to`` preposition, it is
        treated as an infinitival acl modifying the preceding object noun.
        """
        if not (
            token.pos_ == "VERB"
            and token.head.dep_ == "prep"
            and token.head.text.lower() == "to"
        ):
            return None

        noun = next(
            (
                ch for ch in token.head.head.children
                if ch.dep_ in {"dobj", "obj"} and ch.i < token.head.i
            ),
            token.head.head,
        )
        return self.acl_splitter.split(doc, token, noun=noun)

    def split_sentence(self, sentence: str) -> list[str]:
        """
        Split *sentence* into its constituent clauses.

        Args:
            sentence: A single English sentence.

        Returns:
            A list of clause strings.  The main clause, when present, is
            always the first element.
        """
        doc = self.nlp(sentence)
        splits: list[dict] = []
        used_tokens: set[int] = set()
        root = next(t for t in doc if t.dep_ == "ROOT")

        nominal_groups = (
            self.expand_nominal_conj(doc)
            if "nominal_conj" in self.enabled_splits
            else []
        )

        self._process_nominal_groups(doc, root, nominal_groups, splits, used_tokens)
        self._process_subordinates(doc, splits, used_tokens)
        self._process_main_clause(doc, root, nominal_groups, splits, used_tokens)

        return [s["subordinate"] for s in splits]


    def expand_nominal_conj(self, doc) -> list[dict]:
        """
        Detect and return all nominal coordination groups in *doc*.

        Three kinds of groups are returned:

        - **Noun conj groups**: a head noun coordinated with one or more
          nouns via ``conj`` arcs.  The group carries the shared predicate,
          any shared modifiers, and any relative clauses attached to the head.
        Returns:
            A list of group dicts (see ``_make_noun_conj_group``).
        """
        results = []
        visited: set[int] = set()

        for token in doc:
            if token.i in visited:
                continue

            if not (token.dep_ not in {"conj", "punct", "cc"} and token.pos_ in {"NOUN", "PROPN"}):
                continue

            has_noun_conj = any(
                ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"}
                for ch in token.children
            )

            if has_noun_conj:
                group = self._make_noun_conj_group(token, visited)
                results.append(group)

        return results

    def _make_noun_conj_group(self, token, visited: set) -> dict:
        """Build a nominal conj group for a head noun with coordinated nouns."""
        nouns = [token] + self._collect_conj_nouns(token)
        for n in nouns:
            visited.add(n.i)

        shared_mods = self._collect_shared_mods(token)
        floating = self._collect_floating_adjs(token, {m.i for m in shared_mods})
        shared_mods = sorted(shared_mods + floating, key=lambda t: t.i)

        relcl_list = [ch for ch in token.children if ch.dep_ == "relcl"]

        return {
            "nouns": nouns,
            "mods": shared_mods,
            "relcl": relcl_list,
            "head_token": token,
        }

    def _collect_floating_adjs(self, token, exclude_idxs: set) -> list:
        """
        Return ADJ tokens that spaCy attached to the predicate head
        (as nsubj/amod) instead of to *token* as amod, but that positionally
        precede *token* and logically modify it.
        """
        pred_head = token.head if token.head.pos_ in {"VERB", "AUX"} else None
        if pred_head is None:
            return []

        result = []
        for ch in pred_head.children:
            if (
                ch.dep_ in {"nsubj", "nsubjpass", "amod"}
                and ch.pos_ == "ADJ"
                and ch.i < token.i
                and ch.i not in exclude_idxs
            ):
                result.append(ch)
                # Also grab conj siblings of this floating adj
                result.extend(self._collect_conj_adjs(ch))
        return sorted(result, key=lambda t: t.i)

    def _collect_conj_nouns(self, token) -> list:
        """Recursively collect NOUN/PROPN tokens chained via conj arcs."""
        result = []
        for ch in token.children:
            if ch.dep_ == "conj" and ch.pos_ in {"NOUN", "PROPN"}:
                result.append(ch)
                result.extend(self._collect_conj_nouns(ch))
        return result

    def _collect_conj_adjs(self, token) -> list:
        """Recursively collect ADJ tokens chained via conj arcs."""
        result = []
        for ch in token.children:
            if ch.dep_ == "conj" and ch.pos_ == "ADJ":
                result.append(ch)
                result.extend(self._collect_conj_adjs(ch))
        return result

    def _collect_shared_mods(self, token) -> list:
        """Return amod/det modifiers of *token*, including their conj siblings."""
        shared = []
        for ch in token.children:
            if ch.dep_ in {"amod", "det"} and ch.pos_ in {"ADJ", "DET"}:
                shared.append(ch)
                shared += [c for c in ch.children if c.dep_ == "conj"]
        return shared

    def _process_nominal_groups(self, doc, root, nominal_groups, splits, used_tokens):
        """Iterate over nominal groups and dispatch the appropriate handler."""
        for group in nominal_groups:
            self._handle_noun_conj_group(doc, root, group, splits, used_tokens)

    def _handle_noun_conj_group(self, doc, root, group, splits, used_tokens):
        """
        Process a noun-coordination group.

        For each noun in the group, generates one clause replicating the
        shared predicate.  Shared relative clauses are propagated to every
        noun.  Embedded subordinates (ccomp, advcl, parataxis) are
        dispatched individually.
        """
        nouns     = group["nouns"]
        mods      = group["mods"]
        relcl_list = group["relcl"]

        self._mark_noun_group_tokens(nouns, mods, relcl_list, used_tokens)

        nominal_idxs = self._build_nominal_idxs(nouns, mods)
        head_noun    = group["head_token"]
        pred_root    = head_noun.head if head_noun.head.pos_ in {"VERB", "AUX"} else root
        predicate_tokens = self._build_predicate_tokens(pred_root, nominal_idxs, relcl_list)
        used_tokens.update(t.i for t in predicate_tokens)

        adj_mods    = [m for m in mods if m.pos_ == "ADJ"]
        adj_mod_idxs = {m.i for m in adj_mods}
        head_det    = [ch for ch in head_noun.children if ch.dep_ == "det"]

        for noun in nouns:
            self._emit_noun_clause(
                doc, noun, nouns, predicate_tokens, relcl_list,
                adj_mods, adj_mod_idxs, head_det, splits,
            )

        self._dispatch_pred_root_subordinates(doc, pred_root, splits, used_tokens)
        self._maybe_emit_root_clause(doc, root, pred_root, splits, used_tokens)

    def _mark_noun_group_tokens(self, nouns, mods, relcl_list, used_tokens):
        """Mark all tokens belonging to a noun-conj group as consumed."""
        for noun in nouns:
            used_tokens.add(noun.i)
            for ch in noun.children:
                if ch.dep_ in {"compound", "cc"}:
                    used_tokens.add(ch.i)
        for mod in mods:
            used_tokens.add(mod.i)
        for relcl in relcl_list:
            used_tokens.update(t.i for t in relcl.subtree)

    def _build_nominal_idxs(self, nouns, mods) -> set:
        """
        Return the set of token indices that belong to the nominal group
        (nouns + their cc/compound children + shared modifiers).
        """
        idxs: set = set()
        for noun in nouns:
            idxs.add(noun.i)
            idxs.update(ch.i for ch in noun.children if ch.dep_ in {"cc", "compound"})
        for mod in mods:
            idxs.add(mod.i)
            idxs.update(ch.i for ch in mod.children if ch.dep_ == "cc")
        return idxs

    def _build_predicate_tokens(self, pred_root, nominal_idxs, relcl_list) -> list:
        """
        Build the shared predicate token list by starting from *pred_root*'s
        subtree and excluding nouns, relcl subtrees, and embedded clauses.
        """
        excluded = (
            nominal_idxs
            | {r.i for relcl in relcl_list for r in relcl.subtree}
            | self.collect_subtree_idxs(pred_root, "ccomp")
            | self.collect_subtree_idxs(pred_root, "advcl")
            | self.collect_subtree_idxs(pred_root, "parataxis")
            | self.collect_subtree_idxs(pred_root, "relcl")
        )
        return sorted(
            [t for t in pred_root.subtree if t.i not in excluded and t.dep_ != "punct"],
            key=lambda t: t.i,
        )

    def _emit_noun_clause(
        self, doc, noun, nouns, predicate_tokens, relcl_list,
        adj_mods, adj_mod_idxs, head_det, splits,
    ):
        """
        Emit one main-predicate clause for *noun* and one relcl clause per
        relative clause in *relcl_list*.
        """
        noun_mods  = self._get_noun_mods(noun, adj_mod_idxs, head_det)
        noun_tokens = sorted(noun_mods + [noun], key=lambda t: t.i)
        filtered_pred = self._filter_pred_for_noun(noun, nouns, predicate_tokens, relcl_list, doc)

        all_tokens = sorted(
            {t.i: t for t in filtered_pred + noun_tokens + adj_mods}.values(),
            key=lambda t: t.i,
        )
        splits.append({"type": "nominal_conj", "subordinate": self.build_clause_text(all_tokens)})

        for relcl in relcl_list:
            self._emit_relcl_for_noun(noun, relcl, adj_mods, adj_mod_idxs, head_det, splits)

    def _emit_relcl_for_noun(self, noun, relcl, adj_mods, adj_mod_idxs, head_det, splits):
        """Emit a propagated relative clause for *noun*."""
        result = self.relcl_splitter.split(None, relcl)
        if not result:
            return

        noun_mods = self._get_noun_mods(noun, adj_mod_idxs, head_det)
        noun_text = (
            " ".join(t.text for t in noun_mods) + " " + noun.text
            if noun_mods else noun.text
        )

        rel_pron_idxs = {
            ch.i for ch in relcl.children if self.is_relative_pronoun(ch)
        }
        head_np_idxs = {relcl.head.i} | {
            ch.i for ch in relcl.head.children
            if ch.dep_ in {"det", "amod", "nummod", "poss", "compound"}
        }
        relcl_body = " ".join(
            t.text for t in sorted(result["tokens"], key=lambda t: t.i)
            if t.i not in head_np_idxs
            and t.i not in rel_pron_idxs
            and t.dep_ != "punct"
        )

        adj_prefix = " ".join(m.text for m in adj_mods) + " " if adj_mods else ""
        splits.append({
            "type": "relcl_propagated",
            "subordinate": f"{adj_prefix}{noun_text} {relcl_body}",
        })

    def _get_noun_mods(self, noun, adj_mod_idxs: set, head_det: list) -> list:
        """
        Return the positionally-sorted modifier tokens for *noun* (det,
        amod, compound), inheriting the head noun's determiner when the
        noun has none of its own.
        """
        mods = sorted(
            [
                ch for ch in noun.children
                if ch.dep_ in {"det", "amod", "compound"} and ch.i not in adj_mod_idxs
            ],
            key=lambda t: t.i,
        )
        if not any(m.dep_ == "det" for m in mods):
            mods = sorted(head_det + mods, key=lambda t: t.i)
        return mods

    def _filter_pred_for_noun(self, noun, nouns, predicate_tokens, relcl_list, doc) -> list:
        """
        Return the predicate token list filtered for a specific *noun*:
        tokens that positionally belong to other nouns (or are cc/relcl
        tokens) are removed.
        """
        noun_positions = sorted(n.i for n in nouns)
        pos_idx  = noun_positions.index(noun.i)
        pos_start = noun.i
        pos_end   = (
            noun_positions[pos_idx + 1] if pos_idx + 1 < len(noun_positions) else len(doc)
        )

        other_direct_mods: set = set()
        for other in nouns:
            if other.i != noun.i:
                other_direct_mods.add(other.i)
                for ch in other.children:
                    if ch.dep_ in {"compound", "det", "amod"}:
                        other_direct_mods.add(ch.i)

        relcl_idxs: set = set()
        for relcl in relcl_list:
            relcl_idxs.update(t.i for t in relcl.subtree)

        out_range: set = set()
        in_range: set  = set()
        for other in nouns:
            if other.i != noun.i:
                for t in other.subtree:
                    if t.i in other_direct_mods or t.i in relcl_idxs or t.dep_ == "cc":
                        out_range.add(t.i)
                    elif pos_start <= t.i < pos_end:
                        in_range.add(t.i)
                    else:
                        out_range.add(t.i)

        return [
            t for t in predicate_tokens
            if t.i not in out_range and t.i not in in_range
        ]

    def _dispatch_pred_root_subordinates(self, doc, pred_root, splits, used_tokens):
        """Dispatch ccomp, advcl and parataxis children of *pred_root*."""
        for dep in ("ccomp", "advcl", "parataxis"):
            for ch in pred_root.children:
                if ch.dep_ == dep and ch.i not in used_tokens:
                    self.dispatch(doc, ch, splits, used_tokens, recurse=(dep != "advcl"))

    def _maybe_emit_root_clause(self, doc, root, pred_root, splits, used_tokens):
        """
        When *pred_root* is not the sentence ROOT, emit a separate clause
        for the ROOT itself (stripping ccomp and advcl subtrees).
        """
        if pred_root == root or root.i in used_tokens:
            return

        excluded = (
            self.collect_subtree_idxs(root, "ccomp")
            | self.collect_subtree_idxs(root, "advcl")
        )
        root_tokens = sorted(
            [
                t for t in root.subtree
                if t.i not in used_tokens
                and t.i not in excluded
                and t.dep_ not in {"punct", "mark", "cc"}
            ],
            key=lambda t: t.i,
        )
        if root_tokens:
            splits.append({"type": "main", "subordinate": self.build_clause_text(root_tokens)})
            used_tokens.update(t.i for t in root_tokens)

        self._dispatch_pred_root_subordinates(doc, root, splits, used_tokens)

    def _process_subordinates(self, doc, splits, used_tokens):
        """Dispatch all remaining tokens with a registered dependency label."""
        for token in doc:
            if token.dep_ in self.splitters and token.i not in used_tokens:
                self.dispatch(doc, token, splits, used_tokens)

    def _process_main_clause(self, doc, root, nominal_groups, splits, used_tokens):
        """
        Reconstruct the main clause from unconsumed tokens and insert it at
        the front of *splits*.

        Skipped when nominal groups were found (they produce their own
        main-like clauses during Pass 1).

        Noun heads of top-level relative clauses are reintroduced even if
        they were consumed during relcl processing, because they are
        structurally part of the main clause.
        """
        if nominal_groups:
            return

        relcl_heads = self._collect_top_level_relcl_heads(doc, root)

        main_tokens = [
            t for t in doc
            if (t.i not in used_tokens or t.i in relcl_heads)
            and t.dep_ not in {"punct", "mark", "cc"}
        ]
        if main_tokens:
            splits.insert(0, {
                "type": "main",
                "subordinate": self.build_clause_text(main_tokens),
            })

    def _collect_top_level_relcl_heads(self, doc, root) -> set:
        """
        Return the indices of noun heads (and their direct modifiers) for
        relative clauses that are directly attached to the sentence root.

        These nouns are part of the main clause even though they were
        consumed during relative clause processing.
        """
        relcl_heads: set = set()
        for token in doc:
            if token.dep_ == "relcl":
                head = token.head
                if head.head == root or head == root:
                    relcl_heads.add(head.i)
                    for ch in head.children:
                        if ch.dep_ in {"det", "amod", "compound"}:
                            relcl_heads.add(ch.i)
        return relcl_heads

    def resolve_dep(self, token, doc=None) -> str:
        """
        Normalise a raw spaCy dependency label before dispatch.

        Returns the canonical label, ``"SKIP"`` to suppress processing, or
        the original label unchanged.
        """
        dep = token.dep_

        if dep == "relcl" and any(
            ch.dep_ == "aux" and ch.text.lower() == "to" for ch in token.children
        ):
            return "acl"

        if dep == "acl" and token.head.dep_ == "dobj" and self._is_inside_relcl(token.head):
            return "SKIP"

        if dep == "conj":
            chain_root = token
            while chain_root.dep_ == "conj":
                chain_root = chain_root.head
            if chain_root.dep_ in {"ccomp", "xcomp"}:
                return "ccomp"

        if dep == "ccomp" and doc is not None:
            if self.parataxis_splitter.is_disguised_parataxis(doc, token):
                return "parataxis"

        return dep

    def _is_inside_relcl(self, token) -> bool:
        """Return True if *token* is nested inside a relcl subtree."""
        t = token.head
        while t != t.head:
            if t.dep_ == "relcl":
                return True
            t = t.head
        return False

    def dispatch(self, doc, token, splits, used_tokens, recurse: bool = True):
        """
        Resolve the dependency label of *token*, call the appropriate
        splitter, append the result to *splits*, and mark consumed tokens.

        Args:
            doc:         spaCy Doc.
            token:       Token to dispatch.
            splits:      Accumulator list of result dicts.
            used_tokens: Set of already-consumed token indices (mutated).
            recurse:     If True, also process nested subordinates inside
                         the extracted clause.
        """
        actual_dep = self.resolve_dep(token, doc)
        if actual_dep not in self.splitters:
            return

        result = self.splitters[actual_dep](doc, token)
        if not result:
            return

        splits.append({"type": result["type"], "subordinate": result["subordinate"]})
        used_tokens.update(t.i for t in result["tokens"])

        if recurse and actual_dep in {"ccomp", "relcl", "acl", "conj", "parataxis"}:
            self.process_nested(doc, token, splits, used_tokens)

        if token.dep_ == "pobj" and token.pos_ == "VERB":
            if token.head.dep_ == "prep" and token.head.text.lower() == "to":
                used_tokens.add(token.head.i)

    def process_nested(self, doc, token, splits, used_tokens):
        """
        Recursively dispatch subordinate clauses found inside *token*'s
        subtree that have not yet been consumed.
        """
        for t in token.subtree:
            if t.i == token.i or t.i in used_tokens:
                continue
            if self.resolve_dep(t, doc) in self.splitters:
                self.dispatch(doc, t, splits, used_tokens, recurse=True)