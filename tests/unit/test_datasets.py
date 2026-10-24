import numpy as np
import pandas as pd

from harm_detection.datasets import civil_comments as cc
from harm_detection.datasets.leakage import (
    check_leakage,
    content_hash,
    dedupe_exact,
    file_sha256,
    find_near_duplicates,
)


def test_content_hash_normalizes_case_and_whitespace():
    assert content_hash("Hello   World") == content_hash(" hello world ")
    assert content_hash("hello world") != content_hash("hello world!")


def test_dedupe_exact_keeps_first():
    r = dedupe_exact(["a b", "A  B", "c"])
    assert r.kept == [0, 2] and r.duplicates == [(1, 0)]


def test_near_duplicates_detected():
    pairs = find_near_duplicates(
        ["I know where you live Sam... I will break your legs!"],
        ["I know where you live, Sam. I will break your legs."],
    )
    assert len(pairs) == 1 and pairs[0][2] == 1.0


def test_short_texts_not_near_matched():
    assert find_near_duplicates(["ok sure"], ["ok sure"]) == []


def test_leakage_report():
    dev = [
        ("d1", "s1", "the quick brown fox jumps"),
        ("d2", "s2", "completely different text here ok"),
    ]
    hold = [
        ("h1", "s1", "unrelated words entirely fine now"),
        ("h2", "s9", "The quick  brown fox jumps"),
        ("h3", "s8", "completely different text here ok!"),
    ]
    rep = check_leakage(dev, hold)
    assert rep.shared_source_ids == ["s1"]
    assert rep.exact_overlaps == [("h2", "d1")]
    assert [x[:2] for x in rep.near_overlaps] == [("h3", "d2")]
    assert not rep.clean


def test_file_hash(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("abc")
    assert file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def synth_split(n, seed, split, threat_rate=0.05, insult_rate=0.1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "text": [
                f"{split} comment number {i} with words {rng.integers(1e9)}" for i in range(n)
            ],
            **{f: rng.random(n) * 0.2 for f in cc.ANNOTATION_FIELDS},
        }
    )
    df.loc[rng.random(n) < threat_rate, "threat"] = 0.8
    df.loc[rng.random(n) < insult_rate, "insult"] = 0.7
    df.loc[3, "text"] = df.loc[2, "text"]  # duplicate
    df.loc[4, "text"] = "   "  # empty
    df["row_index"] = np.arange(n)
    df["source_split"] = split
    return df


def test_sampling_reproducible_and_leak_free():
    plan = cc.SamplingPlan(
        seed=7, threat_enriched=10, insult_enriched=10, borderline=5, representative=30, holdout=40
    )
    dev = synth_split(500, 1, "validation")
    test = synth_split(500, 2, "test")
    # Inject a holdout record that duplicates a development text.
    test.loc[10, "text"] = dev.loc[20, "text"]
    a = cc.prepare(dev, test, plan, revision="r", label_mapping_version="m")
    b = cc.prepare(dev, test, plan, revision="r", label_mapping_version="m")
    assert [r["case_id"] for r in a.development] == [r["case_id"] for r in b.development]
    assert [r["case_id"] for r in a.holdout] == [r["case_id"] for r in b.holdout]
    counts = a.manifest["sampling"]["development_strata_counts"]
    assert counts["threat_enriched"]["selected"] == 10
    assert len(a.representative) == 30 and len(a.holdout) == 40
    dev_hashes = {content_hash(r["text"]) for r in a.development}
    assert not any(content_hash(r["text"]) in dev_hashes for r in a.holdout)
    assert a.manifest["duplicate_removal"]["final_leakage_check"]["clean"]
    assert a.manifest["eligibility"]["development_source"]["exact_duplicates"] >= 1
    assert a.manifest["eligibility"]["development_source"]["empty_text"] == 1
    # Original annotations preserved verbatim.
    rec = a.development[0]
    row = dev[dev.row_index == int(rec["source_record_id"].split(":")[1])].iloc[0]
    assert rec["reference_labels"]["threat"] == float(row["threat"])


def test_stratum_fallback_disclosed():
    plan = cc.SamplingPlan(
        seed=1, threat_enriched=1000, insult_enriched=5, borderline=5, representative=10, holdout=10
    )
    dev = synth_split(200, 3, "validation")
    test = synth_split(200, 4, "test")
    out = cc.prepare(dev, test, plan, revision="r", label_mapping_version="m")
    c = out.manifest["sampling"]["development_strata_counts"]["threat_enriched"]
    assert c["selected"] < c["requested"] and c["selected"] == c["eligible_pool"]
