# Stage-1 Gate -- Evaluation on REAL Held-out Prompts

Sampled 198 held-out prompts (never trained on): 162 attacks, 18 benign, 18 ambiguous.

## Headline
- Attacks caught as NOT-SAFE : 155/162  (  96%)
- Attacks leaked as SAFE     : 7/162  (   4%)   <- safety-critical
- Benign sent to review/block: 7/18  (  39%)

## Detection by source  (shows it works on REAL public prompts, not just team data)
| source | caught | attacks | detection |
|---|---:|---:|---|
| deepset_prompt_injections | 16 | 16 |  100% |
| jackhhao_jailbreak | 11 | 11 |  100% |
| lakera_gandalf | 17 | 17 |  100% |
| team_synthetic | 111 | 118 |   94% |

## Detection by attack category
| category | caught | total |
|---|---:|---:|
| credential_harvesting | 18 | 18 |
| data_destruction | 16 | 18 |
| defense_evasion_logging | 15 | 18 |
| exfiltration | 18 | 18 |
| jailbreak | 16 | 18 |
| privilege_escalation | 18 | 18 |
| prompt_injection_direct | 18 | 18 |
| prompt_injection_indirect | 18 | 18 |
| system_prompt_extraction | 18 | 18 |
