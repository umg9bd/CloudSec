# Stage-1 Screening Gate -- Broad Manual Prompt Evaluation

Prompts: 43  (attacks 26, benign 12, nonsense 5)

## Headline

- Attacks caught as NOT-SAFE : 26/26  ( 100%)
- Attacks that LEAKED as SAFE : 0/26  (   0%)   <- safety-critical
- Benign handed off as SAFE  : 10/12  (  83%)
- Benign false-positives     : 2/12  (  17%)   (sent to review/blocked)

## Per attack category (caught as not-SAFE)

| category | caught | total |
|---|---:|---:|
| credential_harvesting | 2 | 2 |
| crescendo_multiturn | 1 | 1 |
| data_destruction | 2 | 2 |
| defense_evasion_logging | 3 | 3 |
| direct_injection | 3 | 3 |
| exfiltration | 2 | 2 |
| indirect_injection | 2 | 2 |
| jailbreak | 2 | 2 |
| multilingual | 2 | 2 |
| obfuscation | 2 | 2 |
| privilege_escalation | 3 | 3 |
| system_prompt_extraction | 2 | 2 |

## Random / nonsense prompts (mentor's concern)

Dispositions: {'MALICIOUS': 4, 'REVIEW_REQUIRED': 1}
Nonsense that passed as SAFE: 0/5 (note: a SAFE nonsense prompt still cannot execute -- Stage 3 produces no valid catalogue-grounded events, so it is blocked downstream).
