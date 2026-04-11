from src.pipeline import run_autogluon_pipeline

if __name__ == "__main__":
    # Parameters
    features = ['wind_speed_100m', 'wind_direction_100m']
    
    # Execute the experiment
    result = run_autogluon_pipeline(
        ed, esql, features, 
        start='20250101', 
        end='20260410'
    )
    
    print(f"Best Model: {result.tabular_best_model}")
