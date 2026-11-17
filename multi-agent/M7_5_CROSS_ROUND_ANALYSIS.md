# M7.5 Cross-Round Analysis

## Evidence chain

- M7.2: 50-case scientific judgment benchmark; 14-case pilot, INCONCLUSIVE.
- M7.3: 4-arm memory benchmark; 40 decisions, INCONCLUSIVE due metric construct defect.
- M7.4: 2-round Base/Candidate RSI pilot; no development candidate met MME, no adoption.
- M7.5: A/B/C/D integrated pilot; 7 cases per group, INCONCLUSIVE due insufficient power.

## Integrated pilot results

| Group | n | Evidence accuracy | Novelty | Falsifiability | Tokens |
|---|---:|---:|---:|---:|---:|
| A_BASE | 7 | 0.571 | 0.857 | 0.857 | 2205 |
| B_MEMORY | 7 | 0.571 | 0.857 | 1.000 | 3171 |
| C_RSI | 7 | 0.714 | 1.000 | 0.857 | 2382 |
| D_FULL | 7 | 0.714 | 1.000 | 1.000 | 3349 |

## Interpretation

- B_MEMORY did not improve the primary outcome, but falsifiability accuracy increased by 0.143 with 966 extra tokens.
- C_RSI improved the primary outcome by 0.143 with 177 extra tokens.
- D_FULL matched C_RSI on the primary outcome and reached the best falsifiability accuracy, but used 1164 extra tokens versus A_BASE.
- There is no evidence of synergy: adding memory to RSI did not improve the primary outcome beyond RSI alone.
- n=7 per group is far below the pre-registered minimum families. These are pilot signals, not validated capability gains.

## Conclusion

The integrated mechanism ran, but integrated capability is not validated. No finding is confirmed and no candidate was adopted.
