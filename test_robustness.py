"""Robustness battery for the ProteinMPNN fork's utility surface.

Targets pure parsing/scoring functions (no model weights needed):
parse_fasta / parse_PDB_biounits file edge cases, loss masking
semantics, and gather helpers.
"""
import os
import tempfile

import numpy as np
import pytest
import torch

import protein_mpnn_utils as U


# ---------- fixtures ------------------------------------------------------

PDB_LINES = [
    "ATOM      1  N   ALA A   1      11.104  13.207   9.480  1.00 20.00           N  ",
    "ATOM      2  CA  ALA A   1      12.560  13.300   9.480  1.00 20.00           C  ",
    "ATOM      3  C   ALA A   1      13.000  14.500   9.000  1.00 20.00           C  ",
    "ATOM      4  O   ALA A   1      12.400  15.300   8.400  1.00 20.00           O  ",
    "ATOM      5  N   GLY A   2      14.200  14.800   9.200  1.00 20.00           N  ",
    "ATOM      6  CA  GLY A   2      14.800  16.000   8.800  1.00 20.00           C  ",
    "ATOM      7  C   GLY A   2      16.200  16.000   9.000  1.00 20.00           C  ",
    "ATOM      8  O   GLY A   2      16.800  15.000   8.600  1.00 20.00           O  ",
    "TER",
    "END",
]


def _write(lines, suffix):
    f = tempfile.NamedTemporaryFile(
        "w", suffix=suffix, delete=False)
    f.write("\n".join(lines))
    f.close()
    return f.name


class TestParseFasta:
    def test_roundtrip_multiline(self):
        p = _write([">h1", "ACDE", "FGHI", ">h2", "LMNP"], ".fa")
        try:
            h, s = U.parse_fasta(p)
        finally:
            os.unlink(p)
        assert list(h) == ["h1", "h2"]
        assert list(s) == ["ACDEFGHI", "LMNP"]

    def test_blank_line_does_not_crash(self):
        # trailing/embedded blank lines previously raised IndexError
        p = _write([">h1", "ACDE", "", ">h2", "FGHI", ""], ".fa")
        try:
            h, s = U.parse_fasta(p)
        finally:
            os.unlink(p)
        assert list(s) == ["ACDE", "FGHI"]

    def test_limit(self):
        p = _write([">a", "AA", ">b", "BB", ">c", "CC"], ".fa")
        try:
            h, s = U.parse_fasta(p, limit=2)
        finally:
            os.unlink(p)
        assert len(h) == 2

    def test_omit_chars(self):
        p = _write([">h", "A-C-DE"], ".fa")
        try:
            h, s = U.parse_fasta(p, omit=["-"])
        finally:
            os.unlink(p)
        assert list(s) == ["ACDE"]


class TestParsePdb:
    def test_parses_residues_and_seq(self):
        p = _write(PDB_LINES, ".pdb")
        try:
            xyz, seq = U.parse_PDB_biounits(p, atoms=["N", "CA", "C", "O"],
                                          chain="A")
        finally:
            os.unlink(p)
        assert not isinstance(xyz, str)  # error sentinel is a string
        assert list(seq) == ["AG"]
        assert np.asarray(xyz).shape == (2, 4, 3)

    def test_wrong_chain_returns_error_string(self):
        p = _write(PDB_LINES, ".pdb")
        try:
            xyz, seq = U.parse_PDB_biounits(p, atoms=["CA"], chain="Z")
        finally:
            os.unlink(p)
        assert isinstance(xyz, str)

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            U.parse_PDB_biounits("/nonexistent/x.pdb", atoms=["CA"])

    def test_hetatm_ignored_unless_mse(self):
        lines = PDB_LINES + [
            "HETATM  100  O   HOH A  99       0.000   0.000   0.000  1.00 20.00           O  "]
        p = _write(lines, ".pdb")
        try:
            xyz, seq = U.parse_PDB_biounits(p, atoms=["N", "CA", "C", "O"],
                                            chain="A")
        finally:
            os.unlink(p)
        assert list(seq) == ["AG"]  # waters must not extend the sequence


class TestLossMasking:
    """Contract: an all-zero mask makes the average undefined (0/0).
    Currently NaN — pinned here; a silent 0.0 would hide the empty
    segment entirely."""

    def test_loss_nll_all_zero_mask_is_nan(self):
        S = torch.randint(0, 21, (2, 8))
        log_probs = torch.log_softmax(torch.randn(2, 8, 21), dim=-1)
        mask = torch.zeros(2, 8)
        _, loss_av = U.loss_nll(S, log_probs, mask)
        assert torch.isnan(loss_av)

    def test_loss_nll_normal_case(self):
        S = torch.randint(0, 21, (2, 8))
        log_probs = torch.log_softmax(torch.randn(2, 8, 21), dim=-1)
        mask = torch.ones(2, 8)
        loss, loss_av = U.loss_nll(S, log_probs, mask)
        assert torch.isfinite(loss_av)
        assert loss.shape == S.shape

    def test_scores_all_zero_mask_is_nan(self):
        S = torch.randint(0, 21, (4, 6))
        log_probs = torch.log_softmax(torch.randn(4, 6, 21), dim=-1)
        mask = torch.zeros(4, 6)
        assert torch.isnan(U._scores(S, log_probs, mask)).all()

    def test_scores_mask_selects_positions(self):
        # masking to a single position must equal that position's NLL
        S = torch.tensor([[3]], dtype=torch.long)
        lp = torch.log_softmax(torch.randn(1, 1, 21), dim=-1)
        mask = torch.ones(1, 1)
        sc = U._scores(S, lp, mask)
        assert torch.isclose(sc[0], -lp[0, 0, 3])

    def test_loss_smoothed_all_zero_mask_is_nan(self):
        S = torch.randint(0, 21, (2, 8))
        log_probs = torch.log_softmax(torch.randn(2, 8, 21), dim=-1)
        _, loss_av = U.loss_smoothed(S, log_probs, torch.zeros(2, 8))
        assert torch.isnan(loss_av)


class TestS2Seq:
    def test_mask_drops_positions(self):
        S = torch.tensor([0, 1, 2])  # A C D
        mask = torch.tensor([1., 0., 1.])
        assert U._S_to_seq(S, mask) == "AD"

    def test_full_mask(self):
        S = torch.tensor([0, 20])  # A X
        assert U._S_to_seq(S, torch.ones(2)) == "AX"


class TestGatherHelpers:
    def test_gather_nodes_shapes(self):
        nodes = torch.randn(2, 5, 8)
        idx = torch.randint(0, 5, (2, 5, 3))
        out = U.gather_nodes(nodes, idx)
        assert out.shape == (2, 5, 3, 8)

    def test_gather_nodes_values(self):
        nodes = torch.arange(2 * 3 * 4).float().reshape(2, 3, 4)
        idx = torch.tensor([[[0, 2]]])  # gather pos 0 and 2
        idx = idx.expand(1, 3, 2)
        out = U.gather_nodes(nodes[0:1], idx)
        assert torch.equal(out[0, 0, 0], nodes[0, 0])
        assert torch.equal(out[0, 0, 1], nodes[0, 2])

    def test_gather_edges_gather_pairwise(self):
        E = torch.randn(1, 3, 3, 8)
        idx = torch.tensor([[[0, 1]]]).expand(1, 3, 2)
        out = U.gather_edges(E, idx)
        assert out.shape == (1, 3, 2, 8)
        assert torch.equal(out[0, 0, 1], E[0, 0, 1])
