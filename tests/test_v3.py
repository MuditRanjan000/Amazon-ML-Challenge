import sys
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution.v3_match import norm_addr, norm_name, normalize_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "execution"))
import v3_pipeline as v3  # noqa: E402


def test_normalizers_handle_generator_noise():
    assert norm_name("Crysta1 Lending  PC") == norm_name("Crystal Lending") == "crystal lending"
    assert norm_name("Bombaypower.Com") == "bombaypower"
    assert norm_addr("66 Edgewood Street, Bridgeport, Connecticut") == norm_addr("66 EDGEWOOD ST, BRIDGEPORT, CT")
    r = normalize_records(pd.DataFrame({"entity_id": ["x"], "business_name": ["ಬಾಂಬೆ ಪವರ್"],
                                        "business_address": ["No: 032/2, 2Nd Floor"]}), processes=1).iloc[0]
    assert (r.nums, r.dig, r.n1, r.nonlatin) == ("2 32", "3222", "32", True)


def test_context_and_owner():
    df = pd.DataFrame({"s1": list("aabbc"), "cand": list("xyxzx"), "score": 1.0, "rank": [1, 2, 1, 2, 1],
                       "p": [0.9, 0.2, 0.5, 0.8, 0.7]})
    c = v3.context(df)
    assert np.allclose(c.rec_other, [0.7, 0, 0.9, 0, 0.9]) and list(c.rec_rank) == [1, 1, 3, 1, 2]
    assert list(v3.owner_mask(df.s1.values, df.cand.values, df.p.values)) == [True, True, False, True, False]


def test_efo_picks_expected_f05_optimum():
    s1, q = np.array(list("aaabbc")), np.array([0.9, 0.6, 0.1, 0.05, 0.02, 0.97])
    assert list(v3.efo_mask(s1, q)) == [True, False, False, False, False, True]  # b: empty beats any pick
