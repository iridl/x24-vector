from cftime import Datetime360Day as DT360

import fbfmaproom

def test_year_label_oneyear():
    assert fbfmaproom.year_label(
        DT360(1961, 6, 1),
        1
    ) == "1961"

def test_year_label_straddle():
    assert fbfmaproom.year_label(
        DT360(1960, 12, 16),
        3
    ) == "1960/61"

def test_from_month_since_360Day():
    assert fbfmaproom.from_month_since_360Day(735.5) == DT360(2021, 4, 16)
