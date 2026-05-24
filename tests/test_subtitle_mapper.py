import os
import tempfile
import pandas as pd
import numpy as np

from opera_align.subtitle_mapper import map_subtitles


def test_map_subtitles_basic(tmp_path):
    df = pd.DataFrame({
        'start_time': [0.0, 1.0, 2.0],
        'end_time': [0.5, 1.5, 2.5],
        'text': ['a', 'b', 'c']
    })
    csv_in = tmp_path / "subs.csv"
    csv_out = tmp_path / "mapped.csv"
    df.to_csv(csv_in, index=False)

    mapped_ref = np.array([0.0, 1.0, 2.0, 3.0])
    mapped_stream = np.array([0.1, 1.1, 2.1, 3.1])

    map_subtitles(str(csv_in), mapped_ref, mapped_stream, str(csv_out))
    df_out = pd.read_csv(csv_out)
    assert 'start_time_stream' in df_out.columns
    assert np.allclose(df_out['start_time_stream'].values, np.array([0.1, 1.1, 2.1]))
