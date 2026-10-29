from daso.evaluation import select_metric


def test_paper_metric_keys() -> None:
    assert select_metric("arc_easy", {"results": {"arc_easy": {"acc,none": 0.7}}}) == 0.7
    assert select_metric("arc_challenge", {"acc_norm": 0.5}) == 0.5

