# Unified slip dataset (schema GCD + quality filter)

Built from `slip_nn` + `slip_nn_policy` + `slip_nn_policy2` + `slip_nn_policy2_heavy` with a **shared label schema**.

- Features: 26-D × 40 steps
- Labels: `y_event`, `y_grip`/`y_policy`/`y_grip_p2`, `y_wr/y_wp/y_wy`
- Missing wrist teachers → 0; missing slip → 0
- Quality: prefer stronger teacher per base_case; cap per source|case; boost hard cases
- Split: stratified by `source|base_case`

Summary: `{'train': 7229, 'val': 901, 'test': 902, 'total': 9032, 'sources': {'slip_nn': 2305, 'slip_nn_policy': 3227, 'slip_nn_policy2': 2000, 'slip_nn_policy2_heavy': 1500}, 'wrist_teacher_frac': 0.38751107454299927, 'slip_positive_frac': 0.237488928255093, 'hard_case_counts': {'friction_div2': 1500, 'friction_s040': 461, 'friction_s045': 500}, 'quality': {'raw_n': 17791, 'dedupe': {'dropped_weaker_teachers': 8759, 'kept': 9032}, 'cap': {'capped': True, 'max_per_source_case': 500, 'trimmed': 1000}, 'boost': {'boosted': True, 'cases': {'friction_div2': {'n': 500, 'mult': 3, 'added': 1000}}, 'added_total': 1000}}, 'y_grip_train_mean': 0.10090314596891403, 'y_grip_train_max': 0.25, 'manifest': '/workspace/data/slip_nn_unified/manifest.json'}`

Retrain NN-2 / grip-only / grip+wrist on this data before comparing.
