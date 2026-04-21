
def _make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["hour"] = index.hour.astype(np.int16)
    # df["minute"] = index.minute.astype(np.int16)
    df["dayofweek"] = index.dayofweek.astype(np.int16)
    # df["dayofyear"] = index.dayofyear.astype(np.int16)
    # cyclical
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    # df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    # df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def _add_weather_features(df: pd.DataFrame, weather_features: List[str]) -> pd.DataFrame:

    out=df.copy()
    for f in weather_features:
        out[f"{f}_added"]=esql.select([f], start=out.index.min(), end=out.index.max(),NN=0).groupby(level='datetime').mean()[f]
    out=out.ffill().bfill()
    print(out)
    return out
    

def _add_diff_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        for k in lags:
            out[f"{c}_diff{k}"] = out[c].diff(k)
    return out


def _add_rolling_features(
    df: pd.DataFrame, cols: List[str], windows: List[int], min_periods: int = 1
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        for w in windows:
            out[f"{c}_ma{w}"] = out[c].rolling(w, min_periods=min_periods).mean()
            out[f"{c}_std{w}"] = out[c].rolling(w, min_periods=min_periods).std()
    return out


def _regional_aggregate(weather: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    weather: index=datetime, columns= features per gridpoint flatten style:
      e.g. u100_<latlon>, v100_<latlon>, wind_speed_100m_<latlon>
    Returns per-timestamp regional aggregate features for each base variable.
    """
    df = weather.copy()
    base_vars = ["u100", "v100", "wind_speed_100m"]
    out = pd.DataFrame(index=df.index)

    for v in base_vars:
        cols = [c for c in df.columns if c.startswith(v + "_")]
        if not cols:
            continue
        block = df[cols]
        if how == "mean":
            out[f"regional_{v}_mean"] = block.mean(axis=1)
            out[f"regional_{v}_std"] = block.std(axis=1)
        elif how == "median":
            out[f"regional_{v}_median"] = block.median(axis=1)
            out[f"regional_{v}_mad"] = (block.sub(block.median(axis=1), axis=0)).abs().median(axis=1)
        else:
            raise ValueError(f"Unknown how={how}")
    
    return out
