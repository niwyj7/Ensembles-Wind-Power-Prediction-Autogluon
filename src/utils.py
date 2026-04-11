def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna()
    b = b.dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < 10:
        return np.nan
    return float(a.loc[idx].corr(b.loc[idx]))

def _eval_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[idx]
    y_pred = y_pred.loc[idx]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), 1e-6))))) * 100.0
    corr = _safe_corr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "corr": corr}


@dataclass
class AutoGluonRunResult:
    tabular_predictor_path: str
    tabular_leaderboard: pd.DataFrame
    tabular_metrics_test: Dict[str, float]
    tabular_best_model: str
    tabular_best_hparams: Dict

    multimodal_predictor_path: Optional[str] = None
    multimodal_metrics_test: Optional[Dict[str, float]] = None
