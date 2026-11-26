
import pytest

from src.data.pair_models import PairRecord, PairSplit, PairType
from src.data.pair_candidate_partitioner import PairCandidatePartitioner
from src.data.pair_label_quota_planner import PairLabelQuotaPlanner
from src.data.pair_selection_models import PairSelectionConfig, PairSelectionStatus
from src.data.pair_final_selector import PairFinalSelector
from src.data.pair_selection_result_assembler import PairSelectionResultAssembler


def make_pair(pair_id, pair_type, split, problem_a="problem_1", problem_b=None):
    positive = pair_type == PairType.POSITIVE

    if problem_b is None:
        problem_b = problem_a if positive else "problem_2"

    return PairRecord.create_canonical(
        pair_id=pair_id,
        sample_a_id=f"{pair_id}_a",
        sample_b_id=f"{pair_id}_b",
        label=1 if positive else 0,
        pair_type=pair_type,
        split=split,
        source_a_id=f"{pair_id}_src_a",
        source_b_id=f"{pair_id}_src_a" if positive else f"{pair_id}_src_b",
        problem_a_id=problem_a,
        problem_b_id=problem_b,
        generation_method="test",
        created_seed=42,
    )


def make_candidates():
    return (
        make_pair("tr_p1", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p2", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_n1", PairType.NEGATIVE, PairSplit.TRAIN),
        make_pair("tr_n2", PairType.NEGATIVE, PairSplit.TRAIN),
        make_pair("va_p1", PairType.POSITIVE, PairSplit.VALIDATION),
        make_pair("va_n1", PairType.NEGATIVE, PairSplit.VALIDATION),
        make_pair("te_p1", PairType.POSITIVE, PairSplit.TEST),
        make_pair("te_n1", PairType.NEGATIVE, PairSplit.TEST),
    )


def run_selection_pipeline(candidates, seed=None, config=None):
    if config is None:
        config = PairSelectionConfig()

    effective_seed = config.seed if seed is None else seed

    partition = PairCandidatePartitioner().partition(candidates)

    quotas = PairLabelQuotaPlanner().plan(
        partition,
        config,
    )

    selector = PairFinalSelector()
    selected = []

    split_groups = (
        partition.train_pairs,
        partition.validation_pairs,
        partition.test_pairs,
    )

    for split_pairs in split_groups:
        if not split_pairs:
            continue

        split_name = split_pairs[0].split.value

        quota = next(
            quota
            for quota in quotas
            if quota.split == split_name
        )

        selected.extend(
            selector.select(
                split_pairs,
                quota,
                seed=effective_seed,
                prefer_endpoint_diversity=config.prefer_endpoint_diversity,
                prefer_problem_balance=config.prefer_problem_balance,
            )
        )

    result, report = PairSelectionResultAssembler().assemble(
        partition,
        quotas,
        tuple(selected),
        selector_version=config.selector_version,
        seed=effective_seed,
    )

    return partition, quotas, result, report


def test_pipeline_runs_from_candidates_to_assembled_result():
    candidates = make_candidates()

    partition, quotas, result, report = run_selection_pipeline(
        candidates
    )

    assert partition.total_count == len(candidates)
    assert len(quotas) == 3

    assert result.total_count == 8
    assert result.positive_count == 4
    assert result.negative_count == 4

    assert report.status == PairSelectionStatus.PASS


def test_pipeline_preserves_original_candidate_objects():
    candidates = make_candidates()

    _, _, result, _ = run_selection_pipeline(candidates)

    candidate_ids = {id(pair) for pair in candidates}

    assert all(
        id(pair) in candidate_ids
        for pair in result.selected_pairs
    )


def test_pipeline_has_no_duplicate_selected_pairs():
    candidates = make_candidates()

    _, _, result, _ = run_selection_pipeline(candidates)

    pair_ids = [
        pair.pair_id
        for pair in result.selected_pairs
    ]

    canonical_keys = [
        pair.canonical_key
        for pair in result.selected_pairs
    ]

    assert len(pair_ids) == len(set(pair_ids))
    assert len(canonical_keys) == len(set(canonical_keys))


def test_pipeline_preserves_split_assignments():
    candidates = make_candidates()

    _, _, result, _ = run_selection_pipeline(candidates)

    selected_by_split = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for pair in result.selected_pairs:
        selected_by_split[pair.split.value].append(pair)

    assert len(selected_by_split["train"]) == 4
    assert len(selected_by_split["validation"]) == 2
    assert len(selected_by_split["test"]) == 2

    assert all(
        pair.split == PairSplit.TRAIN
        for pair in selected_by_split["train"]
    )

    assert all(
        pair.split == PairSplit.VALIDATION
        for pair in selected_by_split["validation"]
    )

    assert all(
        pair.split == PairSplit.TEST
        for pair in selected_by_split["test"]
    )


def test_pipeline_is_reproducible_with_same_seed():
    candidates = make_candidates()

    _, _, result_a, report_a = run_selection_pipeline(
        candidates,
        seed=42,
    )

    _, _, result_b, report_b = run_selection_pipeline(
        candidates,
        seed=42,
    )

    selected_ids_a = tuple(
        pair.pair_id
        for pair in result_a.selected_pairs
    )

    selected_ids_b = tuple(
        pair.pair_id
        for pair in result_b.selected_pairs
    )

    assert selected_ids_a == selected_ids_b
    assert report_a.to_dict() == report_b.to_dict()


def test_pipeline_report_matches_selected_result():
    candidates = make_candidates()

    _, _, result, report = run_selection_pipeline(candidates)

    stats = {
        stat.split: stat
        for stat in report.split_stats
    }

    for split_name in ("train", "validation", "test"):
        split_pairs = [
            pair
            for pair in result.selected_pairs
            if pair.split.value == split_name
        ]

        positive_count = sum(
            pair.pair_type == PairType.POSITIVE
            for pair in split_pairs
        )

        negative_count = sum(
            pair.pair_type == PairType.NEGATIVE
            for pair in split_pairs
        )

        assert stats[split_name].selected_positive_count == positive_count
        assert stats[split_name].selected_negative_count == negative_count
        assert stats[split_name].selected_count == len(split_pairs)

def test_pipeline_handles_imbalanced_candidates_without_oversampling():
    candidates = (
        make_pair("tr_p1", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p2", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p3", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p4", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_n1", PairType.NEGATIVE, PairSplit.TRAIN),
        make_pair("tr_n2", PairType.NEGATIVE, PairSplit.TRAIN),
        make_pair("va_p1", PairType.POSITIVE, PairSplit.VALIDATION),
        make_pair("va_n1", PairType.NEGATIVE, PairSplit.VALIDATION),
        make_pair("te_p1", PairType.POSITIVE, PairSplit.TEST),
        make_pair("te_n1", PairType.NEGATIVE, PairSplit.TEST),
    )

    partition, quotas, result, report = run_selection_pipeline(
        candidates,
        seed=42,
    )

    train_stats = next(
        stat
        for stat in report.split_stats
        if stat.split == "train"
    )

    assert train_stats.candidate_positive_count == 4
    assert train_stats.candidate_negative_count == 2

    assert train_stats.target_positive_count == 2
    assert train_stats.target_negative_count == 2

    assert train_stats.selected_positive_count == 2
    assert train_stats.selected_negative_count == 2

    assert result.total_count == 8
    assert result.positive_count == 4
    assert result.negative_count == 4

def test_pipeline_handles_split_with_only_one_label():
    candidates = (
        make_pair("tr_p1", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p2", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p3", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("va_p1", PairType.POSITIVE, PairSplit.VALIDATION),
        make_pair("va_n1", PairType.NEGATIVE, PairSplit.VALIDATION),
        make_pair("te_p1", PairType.POSITIVE, PairSplit.TEST),
        make_pair("te_n1", PairType.NEGATIVE, PairSplit.TEST),
    )

    _, _, result, report = run_selection_pipeline(
        candidates,
        seed=42,
    )

    train_stats = next(
        stat
        for stat in report.split_stats
        if stat.split == "train"
    )

    assert train_stats.candidate_positive_count == 3
    assert train_stats.candidate_negative_count == 0

    assert train_stats.target_positive_count == 0
    assert train_stats.target_negative_count == 0

    assert train_stats.selected_positive_count == 0
    assert train_stats.selected_negative_count == 0

    assert result.total_count == 4
    assert result.positive_count == 2
    assert result.negative_count == 2

def test_pipeline_handles_empty_split():
    candidates = (
        make_pair("tr_p1", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_n1", PairType.NEGATIVE, PairSplit.TRAIN),
        make_pair("te_p1", PairType.POSITIVE, PairSplit.TEST),
        make_pair("te_n1", PairType.NEGATIVE, PairSplit.TEST),
    )

    partition, quotas, result, report = run_selection_pipeline(
        candidates,
        seed=42,
    )

    validation_stats = next(
        stat
        for stat in report.split_stats
        if stat.split == "validation"
    )

    assert partition.validation_pairs == ()

    assert validation_stats.candidate_positive_count == 0
    assert validation_stats.candidate_negative_count == 0

    assert validation_stats.target_positive_count == 0
    assert validation_stats.target_negative_count == 0

    assert validation_stats.selected_positive_count == 0
    assert validation_stats.selected_negative_count == 0

    assert result.total_count == 4
    assert result.positive_count == 2
    assert result.negative_count == 2

    assert {stat.split for stat in report.split_stats} == {
        "train",
        "validation",
        "test",
    }

def test_pipeline_disables_label_balancing_when_configured():
    candidates = (
        make_pair("tr_p1", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p2", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p3", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_p4", PairType.POSITIVE, PairSplit.TRAIN),
        make_pair("tr_n1", PairType.NEGATIVE, PairSplit.TRAIN),
        make_pair("tr_n2", PairType.NEGATIVE, PairSplit.TRAIN),
        make_pair("va_p1", PairType.POSITIVE, PairSplit.VALIDATION),
        make_pair("va_n1", PairType.NEGATIVE, PairSplit.VALIDATION),
        make_pair("te_p1", PairType.POSITIVE, PairSplit.TEST),
        make_pair("te_n1", PairType.NEGATIVE, PairSplit.TEST),
    )

    config = PairSelectionConfig(balance_labels=False)

    _, _, result, report = run_selection_pipeline(
        candidates,
        seed=42,
        config=config,
    )

    train_stats = next(
        stat
        for stat in report.split_stats
        if stat.split == "train"
    )

    assert train_stats.candidate_positive_count == 4
    assert train_stats.candidate_negative_count == 2

    assert train_stats.target_positive_count == 4
    assert train_stats.target_negative_count == 2

    assert train_stats.selected_positive_count == 4
    assert train_stats.selected_negative_count == 2

    assert result.total_count == 10
    assert result.positive_count == 6
    assert result.negative_count == 4

    assert result.selection_status == PairSelectionStatus.PASS

def test_pipeline_uses_config_seed_when_seed_is_not_explicit():
    config = PairSelectionConfig(seed=123)

    _, _, _, report = run_selection_pipeline(
        make_candidates(),
        config=config,
    )

    assert report.seed == 123


def test_pipeline_uses_config_selector_version():
    config = PairSelectionConfig(
        selector_version="audit-test-2.0.0"
    )

    _, _, _, report = run_selection_pipeline(
        make_candidates(),
        config=config,
    )

    assert report.selector_version == "audit-test-2.0.0"

def test_pipeline_uses_config_seed_when_seed_is_not_explicit():
    config = PairSelectionConfig(seed=123)

    _, _, _, report = run_selection_pipeline(
        make_candidates(),
        config=config,
    )

    assert report.seed == 123


def test_pipeline_uses_config_selector_version():
    config = PairSelectionConfig(
        selector_version="audit-test-2.0.0"
    )

    _, _, _, report = run_selection_pipeline(
        make_candidates(),
        config=config,
    )

    assert report.selector_version == "audit-test-2.0.0"

def test_pipeline_uses_config_selector_version_short_version():
    config = PairSelectionConfig(
        selector_version="audit-test-2.0"
    )

    _, _, _, report = run_selection_pipeline(
        make_candidates(),
        config=config,
    )

    assert report.selector_version == "audit-test-2.0"
