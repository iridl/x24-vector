import fbfmaproom

def test_year_label_oneyear():
    assert fbfmaproom.year_label(12.5, 1) == "1961"

def test_year_label_straddle():
    assert fbfmaproom.year_label(12.5, 3) == "1960/61"
