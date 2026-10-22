# Ablation A-G (real-time pipeline)

Dev sessions: 546. Selection on dev only; test replayed once per variant.

Test: 821 sessions (107 attack).

| Exp | P | R | F1 [95% CI] | AUC | AP | dF1 vs A [95% CI], p |
|---|---|---|---|---|---|---|
| A | 0.856 | 0.888 | 0.872 [0.821, 0.916] | 0.991 | 0.920 | reference |
| B | 0.818 | 0.925 | 0.868 [0.819, 0.912] | 0.987 | 0.911 | -0.003 [-0.031, +0.024], p=0.803 |
| C | 0.814 | 0.897 | 0.853 [0.800, 0.900] | 0.978 | 0.903 | -0.018 [-0.069, +0.031], p=0.470 |
| D-avg | 0.841 | 0.888 | 0.864 [0.812, 0.910] | 0.991 | 0.924 | -0.008 [-0.040, +0.023], p=0.652 |
| D-or | 0.782 | 0.972 | 0.867 [0.817, 0.910] | nan | nan | -0.005 [-0.045, +0.035], p=0.811 |
| E-control | 0.842 | 0.897 | 0.869 [0.818, 0.914] | 0.989 | 0.931 | -0.003 [-0.032, +0.026], p=0.893 |
| E-privilege | 0.790 | 0.916 | 0.848 [0.795, 0.895] | 0.986 | 0.904 | -0.023 [-0.054, +0.006], p=0.121 |
| E-frequency | 0.844 | 0.860 | 0.852 [0.798, 0.899] | 0.983 | 0.907 | -0.020 [-0.051, +0.008], p=0.185 |
| E-action_identity | 0.795 | 0.869 | 0.830 [0.773, 0.881] | 0.987 | 0.907 | -0.041 [-0.079, -0.006], p=0.026 |
| E-node_structure | 0.826 | 0.888 | 0.856 [0.804, 0.902] | 0.989 | 0.921 | -0.016 [-0.041, +0.008], p=0.261 |
| E-rel_assumes | 0.856 | 0.888 | 0.872 [0.821, 0.916] | 0.972 | 0.915 | +0.000 [-0.031, +0.030], p=1.000 |
| F-control | 0.838 | 0.822 | 0.830 [0.772, 0.881] | 0.975 | 0.865 | -0.041 [-0.090, +0.005], p=0.092 |
| F-timing | 0.717 | 0.925 | 0.808 [0.751, 0.860] | 0.975 | 0.841 | -0.063 [-0.105, -0.025], p=0.000 |
| F-risk_priors | 0.762 | 0.897 | 0.824 [0.768, 0.875] | 0.977 | 0.866 | -0.048 [-0.083, -0.016], p=0.003 |
| F-causal_lineage | 0.780 | 0.897 | 0.835 [0.780, 0.884] | 0.960 | 0.883 | -0.037 [-0.066, -0.012], p=0.002 |
| F-permissions | 0.808 | 0.907 | 0.855 [0.802, 0.901] | 0.976 | 0.875 | -0.017 [-0.044, +0.009], p=0.207 |
| E-control@wA | 0.848 | 0.888 | 0.868 [0.817, 0.912] | 0.988 | 0.925 | -0.004 [-0.031, +0.021], p=0.882 |
| E-privilege@wA | 0.782 | 0.907 | 0.840 [0.785, 0.888] | 0.987 | 0.911 | -0.032 [-0.062, -0.004], p=0.027 |
| E-frequency@wA | 0.847 | 0.879 | 0.862 [0.810, 0.908] | 0.988 | 0.921 | -0.009 [-0.036, +0.016], p=0.519 |
| E-action_identity@wA | 0.795 | 0.869 | 0.830 [0.773, 0.881] | 0.987 | 0.907 | -0.041 [-0.079, -0.006], p=0.026 |
| E-node_structure@wA | 0.800 | 0.897 | 0.846 [0.792, 0.893] | 0.990 | 0.929 | -0.026 [-0.055, +0.002], p=0.073 |
| E-rel_assumes@wA | 0.856 | 0.888 | 0.872 [0.821, 0.916] | 0.972 | 0.915 | +0.000 [-0.031, +0.030], p=1.000 |
| F-control@wA | 0.759 | 0.944 | 0.842 [0.787, 0.889] | 0.981 | 0.895 | -0.030 [-0.070, +0.008], p=0.124 |
| F-timing@wA | 0.735 | 0.907 | 0.812 [0.754, 0.863] | 0.975 | 0.837 | -0.060 [-0.097, -0.027], p=0.000 |
| F-risk_priors@wA | 0.754 | 0.916 | 0.827 [0.771, 0.876] | 0.978 | 0.873 | -0.045 [-0.081, -0.012], p=0.008 |
| F-causal_lineage@wA | 0.780 | 0.897 | 0.835 [0.780, 0.884] | 0.960 | 0.883 | -0.037 [-0.066, -0.012], p=0.002 |
| F-permissions@wA | 0.742 | 0.916 | 0.820 [0.763, 0.870] | 0.974 | 0.878 | -0.051 [-0.088, -0.017], p=0.003 |
| G-Random Forest | 0.856 | 0.888 | 0.872 [0.822, 0.916] | nan | nan | +0.000 [-0.036, +0.036], p=1.000 |
| G-XGBoost | 0.869 | 0.869 | 0.869 [0.817, 0.914] | nan | nan | -0.002 [-0.048, +0.042], p=0.922 |

E/F groups vs their retrained control (same recipe and weight rule, nothing removed):

- E-privilege vs E-control: -0.020 [-0.046, +0.005], p=0.113
- E-frequency vs E-control: -0.017 [-0.051, +0.015], p=0.308
- E-action_identity vs E-control: -0.038 [-0.073, -0.006], p=0.019
- E-node_structure vs E-control: -0.013 [-0.043, +0.015], p=0.404
- E-rel_assumes vs E-control: +0.003 [-0.026, +0.031], p=0.884
- F-timing vs F-control: -0.022 [-0.075, +0.031], p=0.409
- F-risk_priors vs F-control: -0.006 [-0.057, +0.044], p=0.809
- F-causal_lineage vs F-control: +0.005 [-0.048, +0.055], p=0.861
- F-permissions vs F-control: +0.024 [-0.021, +0.071], p=0.306
- E-privilege@wA vs E-control@wA: -0.028 [-0.056, -0.001], p=0.046
- E-frequency@wA vs E-control@wA: -0.005 [-0.030, +0.019], p=0.705
- E-action_identity@wA vs E-control@wA: -0.037 [-0.073, -0.004], p=0.031
- E-node_structure@wA vs E-control@wA: -0.022 [-0.050, +0.005], p=0.117
- E-rel_assumes@wA vs E-control@wA: +0.004 [-0.024, +0.032], p=0.818
- F-timing@wA vs F-control@wA: -0.030 [-0.061, -0.002], p=0.040
- F-risk_priors@wA vs F-control@wA: -0.015 [-0.036, +0.003], p=0.122
- F-causal_lineage@wA vs F-control@wA: -0.007 [-0.038, +0.023], p=0.671
- F-permissions@wA vs F-control@wA: -0.022 [-0.045, -0.002], p=0.037

## Dev selection

- A: variant=live, kind=ens, w=0.8000, thr=0.1849, dev_f1=0.9220, dev_ap=0.9104
- B: variant=live, kind=hgt, thr=0.0760, dev_f1=0.9178, dev_ap=0.8918
- C: variant=live, kind=lstm, thr=0.1629, dev_f1=0.8533, dev_ap=0.9196
- D-avg: variant=live, kind=ens, w=0.5000, thr=0.1783, dev_f1=0.8951, dev_ap=0.9149
- D-or: variant=live, kind=or
- E-control: variant=hgt_none, kind=ens, w=0.7000, thr=0.1364, dev_f1=0.9091, dev_ap=0.9078
- E-privilege: variant=hgt_privilege, kind=ens, w=0.9000, thr=0.0729, dev_f1=0.9054, dev_ap=0.8873
- E-frequency: variant=hgt_frequency, kind=ens, w=1.0000, thr=0.1939, dev_f1=0.9296, dev_ap=0.8864
- E-action_identity: variant=hgt_action_identity, kind=ens, w=0.8000, thr=0.0831, dev_f1=0.8750, dev_ap=0.8782
- E-node_structure: variant=hgt_node_structure, kind=ens, w=0.9000, thr=0.0844, dev_f1=0.9091, dev_ap=0.8820
- E-rel_assumes: variant=hgt_rel_assumes, kind=ens, w=0.8000, thr=0.2133, dev_f1=0.9041, dev_ap=0.8935
- F-control: variant=lstm_control, kind=ens, w=0.0000, thr=0.6771, dev_f1=0.8794, dev_ap=0.8962
- F-timing: variant=lstm_timing, kind=ens, w=0.9000, thr=0.1173, dev_f1=0.8466, dev_ap=0.7782
- F-risk_priors: variant=lstm_risk_priors, kind=ens, w=0.5000, thr=0.3969, dev_f1=0.8553, dev_ap=0.8274
- F-causal_lineage: variant=lstm_causal_lineage, kind=ens, w=0.8000, thr=0.2067, dev_f1=0.8933, dev_ap=0.8503
- F-permissions: variant=lstm_permissions, kind=ens, w=0.5000, thr=0.4139, dev_f1=0.8800, dev_ap=0.8441
- E-control@wA: variant=hgt_none, kind=ens, w=0.8000, thr=0.1381, dev_f1=0.9091, dev_ap=0.9000
- E-privilege@wA: variant=hgt_privilege, kind=ens, w=0.8000, thr=0.0834, dev_f1=0.9041, dev_ap=0.8957
- E-frequency@wA: variant=hgt_frequency, kind=ens, w=0.8000, thr=0.1661, dev_f1=0.9231, dev_ap=0.8998
- E-action_identity@wA: variant=hgt_action_identity, kind=ens, w=0.8000, thr=0.0831, dev_f1=0.8750, dev_ap=0.8782
- E-node_structure@wA: variant=hgt_node_structure, kind=ens, w=0.8000, thr=0.0838, dev_f1=0.9028, dev_ap=0.8902
- E-rel_assumes@wA: variant=hgt_rel_assumes, kind=ens, w=0.8000, thr=0.2133, dev_f1=0.9041, dev_ap=0.8935
- F-control@wA: variant=lstm_control, kind=ens, w=0.8000, thr=0.1750, dev_f1=0.8466, dev_ap=0.8541
- F-timing@wA: variant=lstm_timing, kind=ens, w=0.8000, thr=0.2098, dev_f1=0.8428, dev_ap=0.7781
- F-risk_priors@wA: variant=lstm_risk_priors, kind=ens, w=0.8000, thr=0.1938, dev_f1=0.8447, dev_ap=0.8438
- F-causal_lineage@wA: variant=lstm_causal_lineage, kind=ens, w=0.8000, thr=0.2067, dev_f1=0.8933, dev_ap=0.8503
- F-permissions@wA: variant=lstm_permissions, kind=ens, w=0.8000, thr=0.2103, dev_f1=0.8447, dev_ap=0.8359
