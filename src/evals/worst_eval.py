import json
import collections
import os
import re
import pandas as pd

from evals.judge_utils import *
from collections import defaultdict

class WorstEval:
    """Worst-case forgetting evaluation for the forget tasks.

    This class reads judge results from two JSON files (evaluations without and with
    in-context retention, each including paraphrased variations of every question).
    It converts each judged response to a yes/no answer and computes accuracy metrics,
    including a "worst-case" accuracy in which a question counts as still known if ANY
    rephrasing of ANY variant (direct / indirect / reverse), in either setting, is
    answered correctly. Results are logged to a JSONL file.

    Attributes:
        name (str): The name or identifier of the model being evaluated.
        out_dir (str): The base directory where the evaluation results will be saved.
        responsefiles (list[str]): A list of two file paths (strings) to JSON files
                                    containing model responses for evaluation.
        run_name (str): A unique identifier for the specific evaluation run.
        task (str): The type of task being evaluated (e.g., 'forget').
        logs (dict): A dictionary to store the final computed metrics for this evaluation.
        out_path (str): The full file path where the `self.logs` will be saved.
    """

    def __init__(self, name, run_name, worst_dir, files = None, task='forget'):

        self.name = name
        self.out_dir = worst_dir
        self.responsefiles = files
        self.run_name = run_name
        self.task = task
        
    def save_logs(self):
        """Save the logs in a json file"""
        os.makedirs(self.out_dir, exist_ok=True)
        modelname = self.name.split("/")[-1]
        filename = f"worst_case_eval_{modelname}_{self.run_name}_{self.task}.jsonl"
        self.out_path = os.path.join(self.out_dir, filename)
        with open(self.out_path, "w", encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)

    # Helper function to process a single file's data
    def _parse_label(self, label: str):
        """
        Parse a judge-output label into (base_id, variant).

        A label may carry an "@rephrase_key" suffix (e.g. "@q_llama3" or
        "@original"), which is stripped first. The remaining suffix selects the
        variant slot:

          slot "even"     (direct question):   "-even"    or "-direct"
          slot "odd"      (indirect question): "-odd"     or "-indirect"
          slot "opposite" (reverse question):  "-reverse", "-opposite", or a
                          compound "-{even,odd,direct,indirect}-opposite"

        Examples:
          "M1-direct@q_gemini1"  -> base_id="M1",        variant="even"
          "M1-indirect@q_phi2"   -> base_id="M1",        variant="odd"
          "M1-reverse@q_llm1"    -> base_id="M1",        variant="opposite"
          "Forget-M1-indirect"   -> base_id="Forget-M1", variant="odd"

        Returns (base_id: str, variant: str).
        variant is one of: 'even', 'odd', 'opposite', 'unknown'.
        """
        # Strip the @rephrase_key suffix if present.
        core = label.split('@')[0] if '@' in label else label

        # Must check compound forms BEFORE the simple ones, otherwise
        # "K1-even-opposite" would match '-opposite' and leave "K1-even" as base_id.
        if core.endswith('-even-opposite'):
            variant = 'opposite'
            base_id = core[:-len('-even-opposite')]
        elif core.endswith('-odd-opposite'):
            variant = 'opposite'
            base_id = core[:-len('-odd-opposite')]
        elif core.endswith('-direct-opposite'):
            variant = 'opposite'
            base_id = core[:-len('-direct-opposite')]
        elif core.endswith('-indirect-opposite'):
            variant = 'opposite'
            base_id = core[:-len('-indirect-opposite')]
        elif '-reverse' in core:
            variant = 'opposite'   # internal slot name is "opposite"
            base_id = core.replace('-reverse', '')
        elif '-opposite' in core:
            variant = 'opposite'
            base_id = core.replace('-opposite', '')
        elif '-odd' in core:
            variant = 'odd'
            base_id = core.replace('-odd', '')
        elif '-even' in core:
            variant = 'even'
            base_id = core.replace('-even', '')
        elif '-indirect' in core:
            # "indirect" maps to the internal "odd" slot
            variant = 'odd'
            base_id = core.replace('-indirect', '')
        elif '-direct' in core:
            # "direct" maps to the internal "even" slot
            variant = 'even'
            base_id = core.replace('-direct', '')
        else:
            variant = 'unknown'
            base_id = core

        return base_id, variant

    # Helper function to process a single file's data
    def _process_file_data(self, file_path, q_names):
        """
        Parse a judge output file into a grouped structure.

        Returns
        -------
        dict  base_id -> {
                  'even':     { judge_key: [score, score, ...], ... },
                  'odd':      { judge_key: [score, score, ...], ... },
                  'opposite': { judge_key: [score, score, ...], ... },
              }

        All rephrasings of the same (base_id, variant) are collected in the
        list so the caller can decide how to aggregate them (we use max = any-one).

        Example (label "M1-direct@q_llama3"):
            base_id="M1", variant="even"
            → data["M1"]["even"]["judge_q1"].append(score)

        Backward-compatible with old labels like "Forget-M1-odd" (no @ suffix).
        """
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        if data and data[0].get("__metadata__"):
            data = data[1:]

        grouped = collections.defaultdict(
            lambda: {v: collections.defaultdict(list) for v in ('even', 'odd', 'opposite', 'unknown')}
        )

        seen_base_ids: dict = {}  # base_id -> first idx seen (for flat/unknown dedup)
        for idx, entry in enumerate(data):
            base_id, variant = self._parse_label(entry.get('label', ''))
            # Empty label → use index
            if not base_id:
                base_id = f"__entry_{idx}__"
            # For flat/unknown data: duplicate base_ids are separate questions → make unique.
            # (For structured even/odd/opposite, same base_id across rows IS intentional grouping.)
            elif variant == 'unknown':
                if base_id not in seen_base_ids:
                    seen_base_ids[base_id] = idx
                elif seen_base_ids[base_id] != idx:
                    base_id = f"{base_id}__entry_{idx}__"
            for qi in q_names:
                raw = entry.get(qi)
                if raw is None:
                    continue
                try:
                    val = check_yes_no(raw)
                except (ValueError, AttributeError):
                    val = 0
                grouped[base_id][variant][qi].append(val)

        return grouped

    def evaluate(self):
        """
        Worst-case evaluation for forgetting.

        A model is considered to STILL KNOW a fact if it answered ANY of:
          • any rephrasing of the question (even @original counts)
          • any variant: even, odd, opposite
          • either condition: no-ICR (file 1) or ICR (file 2)

        Aggregation per question:
          known(base_id) = max over variants × rephrasings × files × judge_keys

        Final score = fraction of questions still known (higher = worse forgetting).

        Structure built here for clarity:
          per_question[base_id] = {
              'file1': {
                  'even':     { judge_key: [scores...] },
                  'odd':      { judge_key: [scores...] },
                  'opposite': { judge_key: [scores...] },
              },
              'file2': { ... same ... },
          }

        Tolerates labels with no @ suffix and with no opposite variant.
        """
        # --- 1. Load & Parse ---
        with open(self.responsefiles[0], encoding='utf-8') as f:
            temp_data = json.load(f)
        if temp_data and temp_data[0].get("__metadata__"):
            temp_data = temp_data[1:]
        q_names = [k for k in temp_data[0].keys() if k.startswith("judge_")]

        file1 = self._process_file_data(self.responsefiles[0], q_names)
        same_file = (self.responsefiles[0] == self.responsefiles[1])
        file2 = file1 if same_file else self._process_file_data(self.responsefiles[1], q_names)

        all_ids  = sorted(set(file1.keys()) | set(file2.keys()))
        variants = ['even', 'odd', 'opposite']

        # --- 2. Build per_question grouped structure ---
        per_question = {
            uid: {
                'file1': file1[uid],
                'file2': file2[uid],
            }
            for uid in all_ids
        }

        # Detect whether the dataset uses the structured odd/even/opposite labeling scheme.
        # Must be done AFTER per_question is built.
        # If all entries fell into 'unknown' (e.g. forget_adversarial whose labels have no
        # -odd/-even suffix), treat 'unknown' as the single variant so scores are non-zero.
        has_structured_variants = any(
            any(per_question[uid]['file1'][v] or per_question[uid]['file2'][v]
                for v in ('even', 'odd', 'opposite'))
            for uid in all_ids
        ) if all_ids else False
        if not has_structured_variants:
            variants = ['unknown']

        # Persist for use in analyze_rephrase_depth()
        self._per_question = per_question
        self._all_ids = all_ids
        self._has_structured_variants = has_structured_variants
        self._same_file = same_file


        # --- 3. Helper: collapse all scores for one question into a single known flag ---
        def _any_correct(uid: str, file_keys=('file1', 'file2'),
                         use_variants=None) -> int:
            """
            Return 1 if the model answered correctly in ANY:
              rephrase × judge_key × variant (from use_variants) × file (from file_keys).
            Return 0 otherwise.
            """
            vlist = use_variants or variants
            for fk in file_keys:
                for v in vlist:
                    for scores in per_question[uid][fk][v].values():
                        if any(s == 1 for s in scores):
                            return 1
            return 0

        # --- 4. Detect whether opposite variant is actually present ---
        has_opposite = any(
            any(per_question[uid]['file1']['opposite'].values()) or
            any(per_question[uid]['file2']['opposite'].values())
            for uid in all_ids
        )

        results_data = []

        def _score_and_collect(known_flags: list, label: str) -> float:
            """Average of known_flags; append to results_data table."""
            score = sum(known_flags) / len(known_flags) if known_flags else None
            score_str = f"{score:.2%}" if score is not None else "N/A"
            results_data.append({"Metric Group": label, "Score": score_str})
            return score

        # --- 5. Compute Metrics ---
        # Each metric collapses a subset of (files, variants) via _any_correct.

        if not has_structured_variants:
            # Flat dataset (e.g. forget_adversarial) — no odd/even/opposite split.
            # Emit a single overall score per file instead of repeated per-variant scores.
            j_p_flat = _score_and_collect(
                [_any_correct(uid, file_keys=('file1',), use_variants=['unknown']) for uid in all_ids],
                "J_P (Overall)")
            j_icr_flat = _score_and_collect(
                [_any_correct(uid, file_keys=('file2',), use_variants=['unknown']) for uid in all_ids],
                "J_ICR (Overall)") if not same_file else None
            j_w_total = _score_and_collect(
                [_any_correct(uid, use_variants=['unknown']) for uid in all_ids],
                "J_W (Total Worst Case)")
            j_p_indirect = j_p_direct = j_p_opposite = j_p_wc = j_p_wc_di = j_p_opposite_delta = None
            j_icr_indirect = j_icr_direct = j_icr_opposite = j_icr_wc = j_icr_wc_di = j_icr_opposite_delta = None
            j_w_indirect = j_w_direct = j_w_opposite = j_w_di = j_w_opposite_delta = None
            has_opposite = False
        else:
            # A. File 1 only (no-ICR / paraphrase)
            j_p_indirect = _score_and_collect(
                [_any_correct(uid, file_keys=('file1',), use_variants=['odd'])  for uid in all_ids], "J_P (Indirect)")
            j_p_direct = _score_and_collect(
                [_any_correct(uid, file_keys=('file1',), use_variants=['even']) for uid in all_ids], "J_P (Direct)")
            j_p_opposite = _score_and_collect(
                [_any_correct(uid, file_keys=('file1',), use_variants=['opposite']) for uid in all_ids],
                "J_P (Reverse)") if has_opposite else None
            wc_variants_p = ['odd', 'even'] + (['opposite'] if has_opposite else [])
            j_p_wc = _score_and_collect(
                [_any_correct(uid, file_keys=('file1',), use_variants=wc_variants_p) for uid in all_ids],
                "J_P (Worst Case Odd+Even+Reverse)" if has_opposite else "J_P (Worst Case Odd+Even)")

            # A2. File 1 only — direct+indirect worst-case (no opposite)
            j_p_wc_di = _score_and_collect(
                [_any_correct(uid, file_keys=('file1',), use_variants=['odd', 'even']) for uid in all_ids],
                "J_P (Worst Case Direct+Indirect)") if has_opposite else None
            j_p_opposite_delta = (
                round(j_p_wc - j_p_wc_di, 6)
                if has_opposite and j_p_wc is not None and j_p_wc_di is not None
                else None
            )

            # B. File 2 (ICR)
            if not same_file:
                j_icr_indirect = _score_and_collect(
                    [_any_correct(uid, file_keys=('file2',), use_variants=['odd'])  for uid in all_ids], "J_ICR (Indirect)")
                j_icr_direct = _score_and_collect(
                    [_any_correct(uid, file_keys=('file2',), use_variants=['even']) for uid in all_ids], "J_ICR (Direct)")
                j_icr_opposite = _score_and_collect(
                    [_any_correct(uid, file_keys=('file2',), use_variants=['opposite']) for uid in all_ids],
                    "J_ICR (Reverse)") if has_opposite else None
                wc_variants_icr = ['odd', 'even'] + (['opposite'] if has_opposite else [])
                j_icr_wc = _score_and_collect(
                    [_any_correct(uid, file_keys=('file2',), use_variants=wc_variants_icr) for uid in all_ids],
                    "J_ICR (Worst Case Odd+Even+Reverse)" if has_opposite else "J_ICR (Worst Case Odd+Even)")
                # B2. File 2 — direct+indirect worst-case (no opposite)
                j_icr_wc_di = _score_and_collect(
                    [_any_correct(uid, file_keys=('file2',), use_variants=['odd', 'even']) for uid in all_ids],
                    "J_ICR (Worst Case Direct+Indirect)") if has_opposite else None
                j_icr_opposite_delta = (
                    round(j_icr_wc - j_icr_wc_di, 6)
                    if has_opposite and j_icr_wc is not None and j_icr_wc_di is not None
                    else None
                )
            else:
                j_icr_indirect = j_icr_direct = j_icr_opposite = j_icr_wc = None
                j_icr_wc_di = j_icr_opposite_delta = None

            # C. Global worst-case: both files, specific variants
            j_w_indirect = _score_and_collect(
                [_any_correct(uid, use_variants=['odd'])  for uid in all_ids], "J_W (Global Indirect)")
            j_w_direct = _score_and_collect(
                [_any_correct(uid, use_variants=['even']) for uid in all_ids], "J_W (Global Direct)")
            j_w_opposite = _score_and_collect(
                [_any_correct(uid, use_variants=['opposite']) for uid in all_ids],
                "J_W (Global Reverse)") if has_opposite else None

            # Absolute worst case: all variants + all rephrasings + both files
            wc_all = ['odd', 'even'] + (['opposite'] if has_opposite else [])
            j_w_total = _score_and_collect(
                [_any_correct(uid, use_variants=wc_all) for uid in all_ids],
                "J_W (Total Worst Case)")

            # C2. Global direct+indirect only (no opposite)
            j_w_di = _score_and_collect(
                [_any_correct(uid, use_variants=['odd', 'even']) for uid in all_ids],
                "J_W (Direct+Indirect Worst Case)") if has_opposite else None
            j_w_opposite_delta = (
                round(j_w_total - j_w_di, 6)
                if has_opposite and j_w_total is not None and j_w_di is not None
                else None
            )

        # --- 6. Print Report ---
        df_results = pd.DataFrame(results_data)
        model_short = self.name.split("/")[-1]
        header = f"Model: {model_short}  |  Run: {self.run_name}  |  Task: {self.task}"
        print(f"\n{'='*len(header)}")
        print(header)
        print(f"{'='*len(header)}")
        print("\n--- Final Evaluation Report ---")
        print(df_results.to_string(index=False, justify='left'))
        print("\n--- Copy below for Google Sheets ---")
        print(df_results.to_csv(index=False, sep=';'))

        # --- 7. Save Logs ---
        self.logs = {
            "Model": self.name,
            "Task_name": self.run_name,
            "Set": self.task,
            # `has_reverse` is the canonical flag (reverse-question variant present).
            # NB: the internal variant slot is named "opposite" (see _parse_label
            # and the `j_*_opposite` locals) — that is an implementation detail and is
            # never written to the output file.
            "has_reverse": has_opposite,
            # Paraphrase (File 1)
            "J_P_Indirect":        j_p_indirect,
            "J_P_Direct":          j_p_direct,
            "J_P_Reverse":         j_p_opposite,
            "J_P_WC":              j_p_wc,
            "J_P_WC_DI":           j_p_wc_di,            # direct+indirect only
            "J_P_Reverse_Delta":   j_p_opposite_delta,    # J_P_WC - J_P_WC_DI
            # ICR (File 2)
            "J_ICR_Indirect":      j_icr_indirect,
            "J_ICR_Direct":        j_icr_direct,
            "J_ICR_Reverse":       j_icr_opposite,
            "J_ICR_WC":            j_icr_wc,
            "J_ICR_WC_DI":         j_icr_wc_di,           # direct+indirect only
            "J_ICR_Reverse_Delta": j_icr_opposite_delta,  # J_ICR_WC - J_ICR_WC_DI
            # Global
            "J_W_Indirect":        j_w_indirect,
            "J_W_Direct":          j_w_direct,
            "J_W_Reverse":         j_w_opposite,
            "J_W_Total":           j_w_total,
            "J_W_DI":              j_w_di,                # direct+indirect only
            "J_W_Reverse_Delta":   j_w_opposite_delta,    # J_W_Total - J_W_DI
        }
        # The reverse-question metric is written under the canonical `*_Reverse`
        # names above (paper terminology; matches the dataset's `M1-reverse` label),
        # whether the source judge files used the `M1-reverse` or `M1-opposite`
        # label — `_parse_label` accepts both on input. Downstream readers
        # (collect_results.py, show_results.py) accept both the `*_Reverse` and
        # `*_Opposite` spellings.
        self.save_logs()
        self.analyze_rephrase_depth()

    def analyze_rephrase_depth(self):
        """
        For each question with ≥1 correct rephrasing, show how many rephrasings
        were answered correctly — broken down by (No-ICR / ICR) × (Direct / Indirect).

        Output
        ------
        PNG  : <out_dir>/worst_case_eval_<model>_<run>_<task>_reph_depth.png
               2×2 grid of bar charts (rows = No-ICR / ICR, cols = Direct / Indirect).
               X-axis = # correct rephrasings; Y-axis = # questions.
               Grey bar at 0 = fully forgotten; warm bars ≥1 = still known.
               Dashed blue line = mean # correct rephrasings among known questions.

        JSON : <out_dir>/worst_case_eval_<model>_<run>_<task>_reph_depth_detail.json
               Per-question n_correct/n_total for each (file, variant).

        Stats are also appended to self.logs["reph_depth"] and the JSONL is re-saved.

        Skipped silently if:
          - dataset has no structured variants (flat / unknown labels)
          - matplotlib is not installed
        """
        if not getattr(self, '_has_structured_variants', False):
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[analyze_rephrase_depth] matplotlib not available, skipping.")
            return

        per_question = self._per_question
        all_ids      = self._all_ids
        same_file    = self._same_file

        file_keys   = ['file1'] if same_file else ['file1', 'file2']
        file_labels = ['No-ICR'] if same_file else ['No-ICR', 'ICR']
        variants    = ['even', 'odd']
        var_labels  = ['Direct', 'Indirect']

        # ── Build per-question n_correct / n_total for all (file, variant) ──────
        # Judge output is in GROUPED format: one row per (base_id, variant), with
        # each rephrasing stored as a separate judge_* column (judge_question,
        # judge_q_llama3, judge_q_phi2, …).  The list under each key has length 1
        # (one row).  So the rephrasings dimension is the judge_* keys, not list positions.
        def _reph_counts(uid, fk, variant):
            """Return (n_correct, n_total) or None if no rephrasings exist."""
            judge_data = per_question[uid][fk][variant]
            if not judge_data:
                return None
            # Each judge_* key corresponds to one rephrasing.
            n_total   = len(judge_data)
            n_correct = sum(1 for scores in judge_data.values() if scores and scores[0] == 1)
            return n_correct, n_total

        detail = {
            uid: {
                fk: {variant: _reph_counts(uid, fk, variant) for variant in variants}
                for fk in ['file1', 'file2']
            }
            for uid in all_ids
        }

        # ── Plot ────────────────────────────────────────────────────────────────
        nrows = len(file_keys)
        fig, axes = plt.subplots(nrows, 2, figsize=(11, 4 * nrows), squeeze=False)
        cmap = plt.cm.YlOrRd

        reph_depth_stats = {}

        for ri, (fk, fl) in enumerate(zip(file_keys, file_labels)):
            for ci, (variant, vl) in enumerate(zip(variants, var_labels)):
                ax = axes[ri][ci]

                entries = [
                    detail[uid][fk][variant]
                    for uid in all_ids
                    if detail[uid][fk][variant] is not None
                ]
                if not entries:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=12)
                    ax.set_title(f'{vl}  ·  {fl}', fontsize=11)
                    continue

                max_n      = max(nt for _, nt in entries)
                n_total_q  = len(entries)

                # histogram[k] = number of questions with exactly k correct rephrasings
                histogram = [0] * (max_n + 1)
                for nc, _ in entries:
                    histogram[nc] += 1

                n_known = n_total_q - histogram[0]

                xs     = list(range(max_n + 1))
                colors = ['#c8c8c8'] + [cmap(0.35 + 0.65 * i / max(max_n, 1)) for i in range(1, max_n + 1)]
                bars   = ax.bar(xs, histogram, color=colors, edgecolor='black',
                                linewidth=0.6, width=0.6, zorder=2)

                # Count labels above each bar
                for bar, h in zip(bars, histogram):
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.02 * n_total_q,
                                str(h), ha='center', va='bottom',
                                fontsize=10, fontweight='bold')

                # Mean correct rephrasings among known questions
                known_nc = [nc for nc, _ in entries if nc > 0]
                if known_nc:
                    mean_nc = sum(known_nc) / len(known_nc)
                    ax.axvline(mean_nc, color='steelblue', linestyle='--',
                               linewidth=1.8, label=f'mean = {mean_nc:.1f}', zorder=3)
                    ax.legend(fontsize=9, loc='upper right')
                else:
                    mean_nc = None

                ax.set_xlabel('# correct rephrasings', fontsize=11)
                ax.set_ylabel('# questions', fontsize=11)
                ax.set_xticks(xs)
                ax.set_xlim(-0.5, max_n + 0.5)
                ax.set_ylim(0, n_total_q * 1.18)
                ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7, zorder=0)
                pct_str = f'{n_known / n_total_q:.0%}' if n_total_q else 'N/A'
                ax.set_title(
                    f'{vl}  ·  {fl}\n'
                    f'{n_known}/{n_total_q} known ({pct_str})',
                    fontsize=11
                )

                stat_key = f'{fk}_{variant}'
                reph_depth_stats[stat_key] = {
                    'n_questions':               n_total_q,
                    'n_known':                   n_known,
                    'known_pct':                 round(n_known / n_total_q, 4) if n_total_q else None,
                    'mean_n_correct_given_known': round(mean_nc, 3) if mean_nc is not None else None,
                    'max_n_rephrasings':          max_n,
                    'histogram':                  histogram,  # index = n_correct, value = n_questions
                }

        model_short = self.name.split('/')[-1]
        fig.suptitle(
            f'Rephrasing depth  ·  {model_short}  ·  {self.run_name}\n'
            f'Grey bar = fully forgotten  |  Warm bars = ≥1 rephrasing correct  '
            f'|  X = # correct rephrasings',
            fontsize=10, y=1.02
        )
        plt.tight_layout()

        # ── Save PNG ────────────────────────────────────────────────────────────
        png_path = self.out_path.replace('.jsonl', '_reph_depth.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[reph_depth] Histogram saved → {png_path}")

        # ── Save per-question detail JSON ────────────────────────────────────────
        detail_path = self.out_path.replace('.jsonl', '_reph_depth_detail.json')
        serialisable_detail = {
            uid: {
                fk: {
                    variant: (list(v) if v is not None else None)
                    for variant, v in fk_data.items()
                }
                for fk, fk_data in uid_data.items()
            }
            for uid, uid_data in detail.items()
        }
        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump({'stats': reph_depth_stats, 'per_question': serialisable_detail},
                      f, indent=2, ensure_ascii=False)
        print(f"[reph_depth] Per-question detail saved → {detail_path}")

        # ── Attach stats to main logs and re-save ────────────────────────────────
        self.logs['reph_depth'] = reph_depth_stats
        with open(self.out_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)


def natural_sort_key(s):
    # Find all number sequences in the string
    # We turn them into integers so '2' < '10'
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]



def evaluate_combined_adversarial(
    jg_reph_noicr: str,
    jg_reph_icr: str,
    jg_adv: str,
    name: str,
    run_name: str,
    out_dir: str,
) -> str:
    """Compute combined worst-case across forget_rephrasings and forget_adversarial.

    A question is 'still known' if it was answered correctly in ANY:
      rephrasings (no-ICR or ICR) × any rephrase variant
      OR adversarial × any adversarial variant

    Returns the path to the saved output file.
    """
    # Borrow _process_file_data and _parse_label from WorstEval without loading a model
    we = WorstEval.__new__(WorstEval)
    we.name = name
    we.run_name = run_name
    we.out_dir = out_dir

    def _q_names(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        rows = data[1:] if data and data[0].get('__metadata__') else data
        return [k for k in rows[0].keys() if k.startswith('judge_')]

    q_names_reph = _q_names(jg_reph_noicr)
    q_names_adv  = _q_names(jg_adv)

    reph_file1 = we._process_file_data(jg_reph_noicr, q_names_reph)
    reph_file2 = we._process_file_data(jg_reph_icr,   q_names_reph)
    adv_file1  = we._process_file_data(jg_adv,        q_names_adv)

    all_ids  = sorted(set(reph_file1) | set(reph_file2) | set(adv_file1))
    variants = ('even', 'odd', 'opposite', 'unknown')

    # Merge: adversarial judge_keys get adv_ prefix to avoid collision with rephrase keys
    per_question = {}
    for uid in all_ids:
        per_question[uid] = {}
        for fk, reph_src, adv_src in [('file1', reph_file1, adv_file1),
                                       ('file2', reph_file2, adv_file1)]:
            per_question[uid][fk] = {}
            for v in variants:
                reph_scores = dict(reph_src.get(uid, {}).get(v, {}))
                adv_scores  = {f'adv_{k}': vs
                               for k, vs in adv_src.get(uid, {}).get(v, {}).items()}
                per_question[uid][fk][v] = {**reph_scores, **adv_scores}

    def _any_correct(uid, file_keys=('file1', 'file2'), use_variants=None):
        for fk in file_keys:
            for v in (use_variants or list(variants)):
                for scores in per_question[uid][fk][v].values():
                    if any(s == 1 for s in scores):
                        return 1
        return 0

    has_opposite = any(
        per_question[uid]['file1']['opposite'] or per_question[uid]['file2']['opposite']
        for uid in all_ids
    )
    wc_variants = ['odd', 'even'] + (['opposite'] if has_opposite else [])

    n = len(all_ids)
    j_w_total = sum(_any_correct(uid, use_variants=wc_variants) for uid in all_ids) / n if n else None
    j_w_di    = sum(_any_correct(uid, use_variants=['odd', 'even']) for uid in all_ids) / n if n else None

    os.makedirs(out_dir, exist_ok=True)
    model_short = name.split('/')[-1]
    out_path = os.path.join(out_dir,
                            f'worst_case_eval_{model_short}_{run_name}_forget_adversarial_combined.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'Model':     name,
            'Task_name': run_name,
            'Set':       'forget_adversarial_combined',
            'J_W_Total': j_w_total,
            'J_W_DI':    j_w_di,
        }, f, indent=4)

    j_w_str = f"{j_w_total:.2%}" if j_w_total is not None else "N/A"
    header = f"Model: {model_short}  |  Run: {run_name}  |  Task: forget_adversarial_combined"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")
    print(f"  J_W_Total (adversarial + rephrasings worst-case): {j_w_str}")

    return out_path


class AvgEval:
    """Evaluates language model performance for "average-case" scenarios, typically for retain set.

    This class reads evaluation results from a single JSON file, processes responses
    to 'yes/no' answers, and computes an "average-case" accuracy.
    Results are logged to a JSONL file, similar to `WorstEval` but focused on average performance.

    Attributes:
        name (str): The name or identifier of the model being evaluated.
        out_dir (str): The base directory where the evaluation results will be saved.
        responsefiles (list[str]): A list containing a single file path (string) to a JSON file
                                    containing model responses for evaluation.
        run_name (str): A unique identifier for the specific evaluation run.
        task (str): The type of task being evaluated (e.g., 'retain').
        logs (dict): A dictionary to store the final computed metrics for this evaluation.
        out_path (str): The full file path where the `self.logs` will be saved.
    """
    def __init__(self, name, run_name, worst_dir, files = None, task='retain'):

        self.name = name
        self.out_dir = worst_dir
        self.responsefiles = files
        self.run_name = run_name
        self.task = task
    
    def save_logs(self):
        """Save the logs in a json file"""
        os.makedirs(self.out_dir, exist_ok=True)
        modelname = self.name.split("/")[-1]
        filename = f"avg_case_eval_{modelname}_{self.run_name}_{self.task}.jsonl"
        self.out_path = os.path.join(self.out_dir, filename)
        with open(self.out_path, "w", encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)

    def evaluate(self):
        """
        Evaluates model responses with three levels of granularity:
        1. Global Average
        2. Category Average (Prefix before first hyphen)
        3. Detailed Group Average (Custom grouping logic)
        """

        # --- Helper 1: Extract Prefix (Meta Stage) ---
        def get_prefix_name(raw_label):
            # "Syntax-M1-direct-2" -> "Syntax"
            if '-' in str(raw_label):
                return str(raw_label).split('-')[0]
            return str(raw_label)

        # --- Helper 2: Extract Detailed Group ---
        def get_detailed_group_name(raw_label):
            # Strip the @rephrase_key suffix (e.g. "@q_llama1", "@blank_phi2") so that
            # all rephrase variants of the same question are averaged together.
            l = str(raw_label).split('@')[0]
            if l.startswith("Syntax"):
                return "Syntax"
            elif l.startswith("GK"):
                return "GK"
            return l

        # --- Load Data ---
        with open(self.responsefiles[0], encoding='utf-8') as f:
            evals_para = json.load(f)

        q_names = [k for k in evals_para[0].keys() if k.startswith("judge_")]

        # --- 1. Initialize Containers ---
        # Level 1: Global
        global_answers = {qi: [] for qi in q_names}
        # Level 2: Prefix (Meta)
        prefix_grouped_data = defaultdict(lambda: {qi: [] for qi in q_names})
        # Level 3: Detailed
        detailed_grouped_data = defaultdict(lambda: {qi: [] for qi in q_names})

        # --- 2. Process Data Once ---
        for d in evals_para:
            raw_label = d.get('label', 'Unknown')

            # Determine keys for each level
            prefix_key = get_prefix_name(raw_label)
            detailed_key = get_detailed_group_name(raw_label)

            for qi in q_names:
                ans = check_yes_no(d[f'{qi}'])

                # Fill Level 1
                global_answers[qi].append(ans)
                # Fill Level 2
                prefix_grouped_data[prefix_key][qi].append(ans)
                # Fill Level 3
                detailed_grouped_data[detailed_key][qi].append(ans)

        # --- 3. Compute and Print Stats ---

        final_logs = {}

        def compute_and_print(data_dict, level_name):
            # 1. Create a list to collect the data
            report_data = []

            # Sort keys using your natural sort helper
            sorted_keys = sorted(data_dict.keys(), key=natural_sort_key)

            for key in sorted_keys:
                try:
                    score = average_case_acc(data_dict[key], prefix="judge_")
                except:
                    score = None

                score_str = f"{score:.2%}" if score is not None else "N/A"

                # 2. Append to list instead of printing immediately
                report_data.append({
                    "Group": key,
                    "Score": score_str
                })

                # Add to logs
                log_prefix = "Cat_" if level_name == "Category" else "Det_"
                final_logs[f"{log_prefix}{key}"] = score

            # 3. Convert to DataFrame and Print
            if report_data:
                df = pd.DataFrame(report_data)

                print(f"\n--- {level_name} Stats ---")
                # prints semicolon-separated: "Group;Score"
                print(df.to_csv(sep=';', index=False))
            else:
                print(f"\n--- {level_name} Stats: No Data ---")

        # A. Print Category (Prefix) Stats
        compute_and_print(prefix_grouped_data, "Category")

        # B. Print Detailed Stats
        compute_and_print(detailed_grouped_data, "Detailed")

        # C. Global Stats
        model_short = self.name.split("/")[-1]
        header = f"Model: {model_short}  |  Run: {self.run_name}  |  Task: {self.task}"
        print(f"\n{'='*len(header)}")
        print(header)
        print(f"{'='*len(header)}")
        print("\n--- Global Stats ---")
        try:
            total_acc = average_case_acc(global_answers, prefix="judge_")
        except:
            total_acc = None
        print(f"{'J_avg (Total)':<25}\t{total_acc}")

        # Canonicalize the lexical retain category to its paper name ("Lexical").
        # The dataset may carry "Words-" labels, but we always EMIT the canonical
        # "Lexical" key. Results saved under the "Words" key are handled by a
        # read-side fallback (see _coalesce_lexical in
        # scripts/results/show_results.py).
        for _k in list(final_logs.keys()):
            if _k.startswith("Cat_Words") or _k.startswith("Det_Words"):
                _canon = _k.replace("Words", "Lexical", 1)
                final_logs.setdefault(_canon, final_logs[_k])
                del final_logs[_k]

        # --- 4. Save Logs ---
        self.logs = {
            "Model": self.name,
            "Task_name": self.run_name,
            "Set": self.task,
            "J_avg": total_acc,
            **final_logs
        }
        self.save_logs()

