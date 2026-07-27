# Unified slip dataset (schema GCD)

Built from `slip_nn` + `slip_nn_policy` + `slip_nn_policy2` + `slip_nn_policy2_heavy` with a **shared label schema**.

- Features: 26-D × 40 steps
- Labels: `y_event`, `y_grip`/`y_policy`/`y_grip_p2`, `y_wr/y_wp/y_wy`
- Missing wrist teachers → 0; missing slip → 0
- Split: stratified by `source|case`

Summary: `{'train': 14939, 'val': 1426, 'test': 1426, 'total': 17791, 'sources': {'slip_nn': 8298, 'slip_nn_policy': 5993, 'slip_nn_policy2': 2000, 'slip_nn_policy2_heavy': 1500}, 'wrist_teacher_frac': 0.1967286765575409, 'slip_positive_frac': 0.3156652239896577, 'y_grip_train_mean': 0.09578323364257812, 'y_grip_train_max': 0.25, 'manifest': '/workspace/data/slip_nn_unified/manifest.json'}`

Retrain all heads on this data before comparing.
