"""相對排序與可替換的絕對尺寸對照表（介面一致，節點不用改）。"""
import pytest

from vision_pkg.vision_core.size_ranker import (AbsoluteTableRanker, RelativeRanker, make_ranker)


def test_relative_rank_orders_small_to_large():
    r = RelativeRanker(1.5).rank([40, 20, 32, 28, 48, 24, 36, 44])
    assert r.ranks == [6, 1, 4, 3, 8, 2, 5, 7]
    assert not r.low_confidence and r.min_gap == 4


def test_relative_rank_flags_low_confidence_when_gap_below_margin():
    r = RelativeRanker(3.0).rank([20, 24, 25, 32])
    assert r.low_confidence and r.min_gap == 1
    assert r.ranks == [1, 2, 3, 4]


def test_relative_rank_gap_equal_to_margin_is_ok():
    assert not RelativeRanker(3.0).rank([10, 13, 20]).low_confidence


def test_check_order_ascending_ok():
    c = RelativeRanker(1.0).check_order([30, 32, None, 36, None, 40, 42, 44])
    assert c.order_ok and c.wrong_slots == []


def test_check_order_reports_swapped_slots():
    c = RelativeRanker(1.0).check_order([30, 32, 40, 36, 38, 34, 42, 44])
    assert not c.order_ok and c.wrong_slots == [3, 6]


def test_check_order_pairs_with_empty_slots_use_slot_numbers():
    c = RelativeRanker(1.0).check_order([None, 44, None, 30, None, None, None, 46])
    assert c.wrong_slots == [2] or c.wrong_slots == [4]  # 兩者擇一與 LIS 一致
    assert not c.order_ok


def test_check_order_trivial_cases():
    r = RelativeRanker(1.0)
    assert r.check_order([None] * 8).order_ok
    assert r.check_order([None, 30, None, None, None, None, None, None]).order_ok


def test_check_order_equal_sizes_count_as_wrong():
    c = RelativeRanker(1.0).check_order([30, 30])
    assert not c.order_ok


def test_check_order_ambiguity_sets_low_confidence():
    c = RelativeRanker(3.0).check_order([30, 31, 40])
    assert c.order_ok and c.low_confidence


def test_absolute_table_ranker_classifies_by_nominal_size():
    table = [20, 24, 28, 32, 36, 40, 44, 48]
    r = AbsoluteTableRanker(table, 1.5)
    res = r.rank([47.2, 20.5, 31.0, 36.4])
    assert res.ranks == [8, 1, 4, 5]
    assert not res.low_confidence
    assert r.rank([20, 20.4]).low_confidence  # 兩個物件歸到同一尺寸類別


def test_both_rankers_share_one_interface_via_factory():
    rel = make_ranker('relative', 1.5)
    absr = make_ranker('absolute_table', 1.5, table=[20, 30, 40], table_tolerance=2.0)
    for ranker in (rel, absr):
        res = ranker.rank([40.0, 20.0, 30.0])
        assert res.ranks == [3, 1, 2]
        assert ranker.check_order([20.0, 30.0, 40.0]).order_ok
        assert ranker.check_order([30.0, 20.0, 40.0]).wrong_slots in ([1], [2])


def test_factory_rejects_unknown_type():
    with pytest.raises(ValueError):
        make_ranker('magic', 1.0)
