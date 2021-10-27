from cftime import Datetime360Day as DT360
import numpy as np
import pandas as pd

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

def test_generate_tables():
    main_df, summary_df, prob_thresh = fbfmaproom.generate_tables(
        country_key='ethiopia',
        obs_dataset_key='rain',
        season_config={
            'label': 'MAM',
            'target_month': 3.5,
            'length': 3.0,
            'issue_months': [11, 0, 1],
            'year_range': [1983, 2021]
        },
        table_columns=[
            {'id': 'year_label', 'name': 'Year'},
            {'id': 'enso_state', 'name': 'ENSO State'},
            {'id': 'forecast', 'name': 'Forecast, %'},
            {'id': 'obs_rank', 'name': 'Rain Rank'},
            {'id': 'bad_year', 'name': 'Reported Bad Years'}
        ],
        issue_month=2,
        freq=30,
        mode='0',
        geom_key='ET05',
        severity=0,
    )

    # for c in main_df.columns:
    #     print(f'{c}={list(main_df[c].values)}')

    expected_main = pd.DataFrame.from_dict(dict(
        time=[DT360(year, 4, 16) for year in range(2021, 1982, -1)],
        year_label=[
            '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014',
            '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006',
            '2005', '2004', '2003', '2002', '2001', '2000', '1999', '1998',
            '1997', '1996', '1995', '1994', '1993', '1992', '1991', '1990',
            '1989', '1988', '1987', '1986', '1985', '1984', '1983'
        ],
        enso_state=[
            None, 'Neutral', 'El Niño', 'La Niña', 'Neutral', 'El Niño',
            'El Niño', 'Neutral', 'Neutral', 'La Niña', 'La Niña', 'Neutral',
            'Neutral', 'La Niña', 'Neutral', 'Neutral', 'Neutral', 'Neutral',
            'Neutral', 'Neutral', 'Neutral', 'La Niña', 'La Niña', 'El Niño',
            'Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral', 'El Niño',
            'Neutral', 'Neutral', 'La Niña', 'Neutral', 'El Niño', 'Neutral',
            'La Niña', 'Neutral', 'El Niño'
        ],
        forecast=[
            '34.04', '26.84', '34.28', '32.35', '36.43', '31.38', '32.21',
            '33.35', '38.26', '38.26', '37.05', '30.61', '36.07', '41.86',
            '34.10', '42.46', '36.14', '36.93', '35.87', '31.45', '35.50',
            '40.70', '40.95', '28.68', '35.04', '33.82', '35.92', '37.54',
            '29.53', '39.90', '27.96', '28.82', '34.68', '35.84', '31.13',
            '35.42', '38.82', '38.87', '36.09'
        ],
        obs_rank=[
            24., 33., 12., 38.,  8., 32., 16., 13., 35., 15.,  3., 37.,  5.,
            4., 19., 21., 20.,  9., 14., 10., 11.,  6.,  7., 18., 22., 30.,
            31., 25., 26.,  2., 28., 27., 34., 23., 39., 29., 36.,  1., 17.
        ],
        bad_year=[
            '', None, None, None, 'Bad', 'Bad', '', None, None, None, 'Bad',
            None, 'Bad', 'Bad', None, None, 'Bad', '', None, '', None, 'Bad',
            'Bad', None, None, None, None, None, None, 'Bad', '', '', None,
            None, '', None, None, 'Bad', None
        ],
        worst_obs=[
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0
        ],
        worst_pnep=[
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0
        ],
        severity=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0
        ],
    )).set_index("time")
    pd.testing.assert_frame_equal(main_df, expected_main)

    # for c in summary_df.columns:
    #     print(f'{c}={list(summary_df[c].values)}')
    expected_summary = pd.DataFrame.from_dict(dict(
        year_label=['Worthy-action:', 'Act-in-vain:', 'Fail-to-act:',
                    'Worthy-Inaction:', 'Rate:', 'Year'],
        enso_state=[2, 5, 8, 24, '66.67%', 'ENSO State'],
        forecast=[6, 5, 4, 24, '76.92%', 'Forecast, %'],
        obs_rank=[8, 3, 2, 26, '87.18%', 'Rain Rank'],
        bad_year=[None, None, None, None, None, 'Reported Bad Years'],
    ))
    pd.testing.assert_frame_equal(
        summary_df.set_index("year_label"),
        expected_summary.set_index("year_label")
    )

    assert np.isclose(prob_thresh, 37.052727)


def test_pnep_tile_url_callback():
    resp = fbfmaproom.pnep_tile_url_callback.__wrapped__(
        2021, 2, 30, '/fbfmaproom/ethiopia', 'season1'
    )
    assert resp == '/fbfmaproom-tiles/pnep/{z}/{x}/{y}/ethiopia/season1/2021/2/30'

def test_pnep_tiles():
    with fbfmaproom.SERVER.test_client() as client:
        resp = client.get('/fbfmaproom-tiles/pnep/6/40/27/ethiopia/season1/2021/2/30')
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
