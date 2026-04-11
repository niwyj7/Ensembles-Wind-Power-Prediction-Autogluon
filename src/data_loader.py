def _weather_from_esql(
    esql,
    features: List[str],
    start: str,
    end: str,
    nn: int = 0,
) -> pd.DataFrame:
    """
    Pull raw weather from esql and compute u100/v100 + cleanup.
    Output index=datetime, columns include ['latlon', 'wind_speed_100m','u100','v100'].
    """
    df = esql.select(features, start=start, end=end, NN=nn)
    df = df.reset_index()
    df = df.set_index("datetime")
    df = df.drop(columns=["T"], errors="ignore")

    if "wind_speed_100m" in df.columns and "wind_direction_100m" in df.columns:
        ws = df["wind_speed_100m"].astype(float)
        wd = df["wind_direction_100m"].astype(float)
        df["u100"] = ws * np.sin(np.radians(wd))
        df["v100"] = ws * np.cos(np.radians(wd))

    return df

def _pivot_weather(df: pd.DataFrame, locations: Optional[List[str]] = None) -> pd.DataFrame:
    """
    df has index=datetime, includes column latlon and numeric weather cols.
    Returns wide features: <var>_<latlon>
    """
    work = df.copy()
    if locations is not None:
        work = work[work["latlon"].isin(locations)]
    # keep only numeric weather columns (except latlon)
    keep_cols = [c for c in ["wind_speed_100m", "u100", "v100"] if c in work.columns]
    work = work.reset_index()[["datetime", "latlon"] + keep_cols].copy()
    wide = work.pivot(index="datetime", columns="latlon")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.sort_index()
    # ensure 15-minute grid
    start = wide.index.min().floor("D")
    end = wide.index.max().ceil("D") - pd.Timedelta(minutes=15)
    full = pd.date_range(start, end, freq="15min")
    wide = wide.reindex(full)
    wide = wide.interpolate(limit_direction="both")
    return wide

def _make_dataset(
    ed,
    esql,
    features: List[str],
    start: str,
    end: str,
    nn: int = 0,
    locations: Optional[List[str]] = None,
    add_diff: bool = True,
    add_roll: bool = True,
    add_weather: bool = True,
    regional_how: str = "mean",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) aligned on 15-min index.
    """
    # weather raw -> wide
    weather_raw = _weather_from_esql(esql, features, start, end, nn=nn)
    weather_wide = _pivot_weather(weather_raw, locations=locations)

  

    # regional features (computed from wide grid)
    reg = _regional_aggregate(weather_wide, how=regional_how)
    X=_add_diff_features(weather_wide, weather_wide.columns, lags=[2])
    X = _add_weather_features(X, weather_features=["temperature_2m", "surface_pressure", "rain"])

    # base features: regional + (optional) all grid (can be high-dim)
    X = pd.concat([X, reg], axis=1)


    # time features
    X = pd.concat([X, _make_time_features(X.index)], axis=1)

    # diffs / rolling on regional + on regional aggregates only (more stable)
    base_for_engineering = [c for c in reg.columns if c.startswith("regional_")]
    if add_diff:
        X = _add_diff_features(X, base_for_engineering, lags=[1,3,8])
    if add_roll:
        X = _add_rolling_features(X, base_for_engineering, windows=[12, 24])

    # target
    y_df = ed.pull(["wind_n1"], start=start, end=end).copy().dropna()
    y = y_df["wind_n1"].astype(float)

    # align
    idx = X.index.intersection(y.index)
    X = X.loc[idx].copy()
    y = y.loc[idx].copy()
    
    # remove constant cols (can help CatBoost/LGBM)
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    return X, y
