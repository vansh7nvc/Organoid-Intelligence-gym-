# What changed in this revision

## Applied (text and figures only)
- Abstract, contributions, Discussion, Conclusion rewritten to claims the data support; DSI, near-criticality, "indispensable", plasticity-as-tuning claims removed.
- C7 merged into C3 (identical rows/bars): 14 conditions, 140 runs; condition list, Table III, Appendix B, Figs. 3B/5B/7B updated.
- Baselines relabeled "step 0.25"; Stage-4/OOD baseline comparison stated as confounded; Fig. 8B (curriculum) behaviour described.
- OOD evaluation text rewritten (no "100 episodes", no "drop significantly", no topology interpretation).
- Phase-space text: grid means, GAR-only effect (47.1 vs 57.4 Hz), 12.0 Hz asymptotic ceiling vs 5 s transient.
- C6b/C6c, plasticity attribution, C3 explanation, utility metric, avalanche/branching text rewritten with caveats.
- Fano factor corrected to the table value (121.49, 6.4x); episode-count claim removed; sensitivity appendix rewritten to match Fig. 13.
- Trujillo sentence limited to what the paper reports; "75% gray-matter energy" claim softened; Beggs citation moved to avalanche analysis.
- Figures: units set to a.u., SDM/mitochondrial/wet-lab/21D/Action 0..7 labels fixed, DSI reference lines relabeled, Fig. 6 regenerated from the published cell values.

## Not applied (needs your data/experiments or author input)
- Rerun B1/B2/B4 with step <= goal radius (0.05); report all four curriculum stages; OOD with true n and CIs.
- A GAR-off + metabolism-off training condition; long-horizon (30-60 s) rate runs; GAR trigger counts in the sweep.
- Return decomposition, displacement/collision logs, ||dw|| per episode, reward-normalization test.
- Confirm Eq. (branching) matches the estimator in your code; add goodness-of-fit for alpha or drop it.
- Re-verify Table I cells; fill author block, funding, availability URL/DOI, license.
- Figures not editable here: Fig. 8A legend still says "Fast Tag / Slow Protein Tag"; Fig. 1 arrows unchanged.

## Assumptions to confirm
- C3 had GAR enabled (so it equals the planned GAR-only C7).
- The 2-20 Hz band is a modeling assumption (Trujillo numbers could not be verified).
- Phase-space cell values were transcribed from the PDF text layer.
