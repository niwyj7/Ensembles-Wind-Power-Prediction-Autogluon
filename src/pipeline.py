
def _train_val_test_split_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    val_days: int = 14,
    test_days: int = 7,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Contiguous split. Assumes 15-min frequency.
    """
    df = X.copy()
    df["wind_n1"] = y

    end = df.index.max()
    test_start = end - pd.Timedelta(days=test_days) + pd.Timedelta(minutes=15)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df.loc[: val_start - pd.Timedelta(minutes=15)]
    val = df.loc[val_start: test_start - pd.Timedelta(minutes=15)]
    test = df.loc[test_start:]

    def xy(d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return d.drop(columns=["wind_n1"]), d["wind_n1"]

    return {"train": xy(train), "val": xy(val), "test": xy(test)}


def run_autogluon_pipeline(
    ed,
    esql,
    features: List[str],
    start: str,
    end: str,
    locations: Optional[List[str]] = None,
    val_days: int = 14,
    test_days: int = 7,
    presets: str = "best_quality",
    time_limit: Optional[int] = 3600,
    save_path: str = "./ag_wind_tabular",
    save_path_mm: str = "./ag_wind_multimodal",
) -> AutoGluonRunResult:
    """
    Trains:
      - TabularPredictor: LGBM, CAT, MLP (and others unless restricted)
      - MultiModalPredictor: TFT + MLP-ish heads for time series (via multimodal timeseries)
    """
    warnings.filterwarnings("ignore")

    X, y = _make_dataset(
        ed=ed,
        esql=esql,
        features=features,
        start=start,
        end=end,
        nn=0,
        locations=locations,
        add_diff=True,
        add_roll=True,
        regional_how="mean",
    )
    splits = _train_val_test_split_by_time(X, y, val_days=val_days, test_days=test_days)
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    Xte, yte = splits["test"]

    # -------------------------
    # Tabular: LGBM / CatBoost / MLP
    # -------------------------
    train_df = Xtr.copy()
    train_df["wind_n1"] = ytr
    val_df = Xva.copy()
    val_df["wind_n1"] = yva

    hyperparameters = {
        "GBM": [
            {"extra_trees": False},
            {"extra_trees": True},
        ],
        "CAT": {},
        "NN_TORCH": {
            "num_epochs": 50,
        },
        # keep others off to focus requested models
        "XGB": [],
        "RF": [],
        "XT": [],
        "KNN": [],
        "LR": [],
        "FASTAI": [],
    }

    tab = TabularPredictor(
        label="wind_n1",
        eval_metric="rmse",
        path=save_path,
        verbosity=2,
    ).fit(
        train_data=train_df,
        tuning_data=val_df,
        presets=presets,
        time_limit=time_limit,
        hyperparameters=hyperparameters,
        num_bag_folds=0,
        num_stack_levels=0,
    )

    lb = tab.leaderboard(val_df, silent=True)
    best_model = tab.model_best

    # test metrics
    y_pred_te = pd.Series(tab.predict(Xte), index=Xte.index)
    metrics_te = _eval_metrics(yte, y_pred_te)

    # best hyperparams (if accessible)
    try:
        info = tab.info()
        best_hparams = info.get("model_info", {}).get(best_model, {}).get("hyperparameters", {})
    except Exception:
        best_hparams = {}

    # feature importance on validation set
    fi = tab.feature_importance(val_df, silent=True)
    fi = fi.sort_values("importance", ascending=False)

    print("\n=== Tabular leaderboard (val) ===")
    display(lb)

    print("\n=== Tabular feature importance (top 50, val) ===")
    display(fi.head(50))

    print("\n=== Tabular best model ===")
    print(best_model)
    print("\n=== Tabular test metrics ===")
    print(metrics_te)

    # -------------------------
    # Ablation study (validate RMSE impact)
    # -------------------------
    # Grouped feature sets: base regional, diffs, rolling, time, full grid
    
    cols = X.columns.tolist()
    groups = {
        "regional_base": [c for c in cols if c.startswith("regional_") and (("_diff" not in c) and ("_ma" not in c) and ("_std" not in c))],
        "regional_diff": [c for c in cols if c.startswith("regional_") and ("_diff" in c)],
        "regional_roll": [c for c in cols if c.startswith("regional_") and (("_ma" in c) or ("_std" in c))],
        "time": [c for c in cols if c in {"hour","minute","dayofweek","dayofyear","hour_sin","hour_cos","doy_sin","doy_cos"}],
        "grid_all": [c for c in cols if (c.startswith("u100_") or c.startswith("v100_") or c.startswith("wind_speed_100m_"))],
        "weather_extra": [c for c in cols if c in {"temperature_2m_added", "surface_pressure_added", "rain_added"}],
        "seperated_diff": [c for c in cols if ("_diff" in c) and not c.startswith("regional_")],
    }

    # baseline uses all
    base_pred = pd.Series(tab.predict(Xva), index=Xva.index)
    base_rmse = _eval_metrics(yva, base_pred)["rmse"]

    ablation_rows = []
    for gname, gcols in groups.items():
        if not gcols:
            continue
        keep = [c for c in cols if c not in set(gcols)]
        Xva_drop = Xva[keep]
        # Use same trained model; missing cols won't be accepted, so we re-train fast with restricted cols.
        # Keep time small: refit with only best model type by selecting hyperparameters accordingly.
        train_small = Xtr[keep].copy()
        train_small["wind_n1"] = ytr
        val_small = Xva[keep].copy()
        val_small["wind_n1"] = yva

        tab_small = TabularPredictor(
            label="wind_n1",
            eval_metric="rmse",
            path=f"{save_path}_ablate_{gname}",
            verbosity=0,
        ).fit(
            train_data=train_small,
            tuning_data=val_small,
            presets="medium_quality",
            time_limit=min(600, time_limit) if time_limit else 600,
            hyperparameters=hyperparameters,
            num_bag_folds=0,
            num_stack_levels=0,
        )
        pred_small = pd.Series(tab_small.predict(Xva[keep]), index=Xva.index)
        rmse_small = _eval_metrics(yva, pred_small)["rmse"]
        ablation_rows.append(
            {
                "group_removed": gname,
                "n_removed": len(gcols),
                "val_rmse_full": base_rmse,
                "val_rmse_after_removal": rmse_small,
                "rmse_delta_positive_worse": rmse_small - base_rmse,
            }
        )

    ablation_df = pd.DataFrame(ablation_rows).sort_values("rmse_delta_positive_worse", ascending=False)
    print("\n=== Ablation study (remove feature group) ===")
    display(ablation_df)

    

    return AutoGluonRunResult(
        tabular_predictor_path=save_path,
        tabular_leaderboard=lb,
        tabular_metrics_test=metrics_te,
        tabular_best_model=best_model,
        tabular_best_hparams=best_hparams,
  
    )
