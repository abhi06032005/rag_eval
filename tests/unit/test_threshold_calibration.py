from ragbench.experiments.grid import calibrate_dev_max_f1, calibrate_dev_q05_gold


def test_calibrate_dev_q05_gold():
    scores = [0.95, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
    q05 = calibrate_dev_q05_gold(scores)
    assert 0.50 <= q05 <= 0.60


def test_calibrate_dev_max_f1():
    # hits: (score, is_relevant)
    hits = [
        (0.95, True),
        (0.90, True),
        (0.85, True),
        (0.70, False),
        (0.60, False),
        (0.50, False),
    ]
    best_thresh = calibrate_dev_max_f1(hits)
    assert best_thresh >= 0.85  # Cutoff separating true from false positives
