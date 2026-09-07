import pytest

from ragcore.fusion import rank_positions, reciprocal_rank_fusion


class TestRRF:
    def test_agreement_beats_a_single_strong_vote(self):
        """The defining property of RRF, and the reason it is used here.

        Document 5 is ranked first by one retriever and nowhere by the other.
        Document 2 is ranked second and second. Two second places should
        outrank one first place — that is what makes fusion a genuine
        combination rather than a tie-break on the stronger retriever.
        """
        dense = [5, 2, 9]
        sparse = [7, 2, 4]
        fused = reciprocal_rank_fusion([dense, sparse], k=60)
        assert fused[0][0] == 2

    def test_unanimous_first_place_wins(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [1, 3, 2]], k=60)
        assert fused[0][0] == 1

    def test_all_documents_are_kept(self):
        fused = reciprocal_rank_fusion([[1, 2], [3, 4]], k=60)
        assert {doc for doc, _ in fused} == {1, 2, 3, 4}

    def test_scores_are_descending(self):
        fused = reciprocal_rank_fusion([[1, 2, 3, 4], [4, 3, 2, 1]], k=60)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_single_list_preserves_order(self):
        fused = reciprocal_rank_fusion([[9, 8, 7]], k=60)
        assert [doc for doc, _ in fused] == [9, 8, 7]

    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_is_deterministic_under_ties(self):
        # Two documents with identical rank profiles must always come back in
        # the same order, or the UI's citation numbering flickers between
        # otherwise identical requests.
        a = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60)
        b = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60)
        assert a == b

    def test_two_seconds_always_beat_one_first(self):
        """A closed-form property of RRF worth pinning down.

        One first place scores 1/(k+1); two second places score 2/(k+2).
        The latter is larger for every k > 0, with no dependence on the
        corpus. So under RRF, agreement at rank 2 *always* beats a lone
        rank-1 hit — the choice of k cannot change that.
        """
        for k in (1, 10, 60, 200):
            fused = reciprocal_rank_fusion([[5, 2], [7, 2]], k=k)
            assert fused[0][0] == 2, f"failed at k={k}"

    def test_k_sets_how_deep_agreement_must_be_to_win(self):
        """What k actually controls.

        Document 5 is rank 1 in one list only; document 2 is rank 5 in both.
        Agreement wins only when k is large enough to flatten the difference
        between rank 1 and rank 5 — solving 2/(k+5) > 1/(k+1) gives k > 3.
        The default k=60 sits far on the agreement-favouring side, which is
        the intended behaviour for hybrid retrieval.
        """
        dense = [5, 10, 11, 12, 2]
        sparse = [7, 13, 14, 15, 2]
        assert reciprocal_rank_fusion([dense, sparse], k=60)[0][0] == 2
        assert reciprocal_rank_fusion([dense, sparse], k=1)[0][0] == 5

    def test_weights_shift_the_outcome(self):
        lists = [[1, 2], [2, 1]]
        heavy_first = reciprocal_rank_fusion(lists, k=60, weights=[5.0, 1.0])
        assert heavy_first[0][0] == 1
        heavy_second = reciprocal_rank_fusion(lists, k=60, weights=[1.0, 5.0])
        assert heavy_second[0][0] == 2

    def test_mismatched_weights_raise(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([[1], [2]], weights=[1.0])


class TestRankPositions:
    def test_is_one_based(self):
        assert rank_positions([7, 8, 9]) == {7: 1, 8: 2, 9: 3}

    def test_empty(self):
        assert rank_positions([]) == {}
