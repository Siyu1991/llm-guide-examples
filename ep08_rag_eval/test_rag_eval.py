from rag_eval import run, retrieval_metrics
def test_metrics():
    m = retrieval_metrics(["a", "b", "c"], ["b"], 3)
    assert m["hit"] == 1 and m["mrr"] == 0.5 and abs(m["ndcg"] - 0.6309) < 1e-3
def test_run_and_attribution():
    agg, rows = run(3, "bm25")
    assert len(rows) == 24 and 0 <= agg["correct"] <= 1
    assert agg["fail_retrieval"] + agg["fail_generation"] == sum(1 for r in rows if not r["correct"])
if __name__ == "__main__":
    test_metrics(); test_run_and_attribution(); print("ep08 tests passed")
