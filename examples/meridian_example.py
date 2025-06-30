"""Example usage of the Meridian adapter with mmm-eval."""

import pandas as pd
import numpy as np

from mmm_eval import (
    Evaluator,
    MeridianConfig,
    MeridianPriorDistributionSchema,
    MeridianModelSpecSchema,
    MeridianModelSchema,
    MeridianFitSchema,
)
from mmm_eval.data.constants import InputDataframeConstants


def create_sample_data(n_days: int = 100) -> pd.DataFrame:
    """Create sample data for demonstration."""
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Create sample media spend data
    np.random.seed(42)
    data = pd.DataFrame({
        "date": dates,
        "tv_spend": np.random.uniform(1000, 5000, n_days),
        "digital_spend": np.random.uniform(500, 2000, n_days),
        "radio_spend": np.random.uniform(200, 800, n_days),
        "control_var1": np.random.normal(0, 1, n_days),
        "control_var2": np.random.normal(0, 1, n_days),
    })
    
    # Create response variable (simulated)
    response = (
        50  # Base level
        + 0.8 * data["tv_spend"] ** 0.5
        + 1.2 * data["digital_spend"] ** 0.6
        + 0.5 * data["radio_spend"] ** 0.4
        + 0.3 * data["control_var1"]
        + 0.2 * data["control_var2"]
        + np.random.normal(0, 10, n_days)  # Noise
    )
    
    # Create revenue (assuming 1:1 with response for simplicity)
    revenue = response * 1.0
    
    data[InputDataframeConstants.RESPONSE_COL] = response
    data[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL] = revenue
    
    return data


def create_meridian_config() -> MeridianConfig:
    """Create a Meridian configuration."""
    
    # Create prior distribution configuration
    prior_config = MeridianPriorDistributionSchema(
        roi_mu=0.0,  # Log-normal mean
        roi_sigma=1.0,  # Log-normal standard deviation
        name="roi_m"
    )
    
    # Create model specification
    model_spec_config = MeridianModelSpecSchema(
        prior=prior_config,
        media_effects_dist="log_normal",
        hill_before_adstock=False,
        max_lag=8,
        organic_media_prior_type="contribution",
        non_media_treatments_prior_type="contribution",
    )
    
    # Create model configuration
    model_config = MeridianModelSchema(
        date_column="date",
        media_columns=["tv_spend", "digital_spend", "radio_spend"],
        response_column=InputDataframeConstants.RESPONSE_COL,
        control_columns=["control_var1", "control_var2"],
    )
    
    # Create fit configuration
    fit_config = MeridianFitSchema(
        n_chains=4,
        n_adapt=500,
        n_burnin=500,
        n_keep=1000,
        seed=42,
    )
    
    # Create the complete configuration
    config = MeridianConfig(
        meridian_model_config=model_config,
        model_spec_config=model_spec_config,
        fit_config=fit_config,
        revenue_column=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
        response_column=InputDataframeConstants.RESPONSE_COL,
    )
    
    return config


def main():
    """Demonstrate Meridian adapter usage."""
    print("=== Meridian Adapter Example ===\n")
    
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data(n_days=100)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}\n")
    
    # Create Meridian configuration
    print("Creating Meridian configuration...")
    config = create_meridian_config()
    print("Configuration created successfully!\n")
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator(
        data=data,
        test_names=("accuracy", "cross_validation")  # Run basic tests
    )
    print("Evaluator initialized!\n")
    
    # Evaluate Meridian framework
    print("Evaluating Meridian framework...")
    try:
        results = evaluator.evaluate_framework("meridian", config)
        
        print("Evaluation completed successfully!")
        print(f"Number of tests run: {len(results.results)}")
        
        # Display results
        for test_name, test_result in results.results.items():
            print(f"\n{test_name.upper()} Test Results:")
            if hasattr(test_result, 'test_scores') and test_result.test_scores:
                for metric_name, metric_value in test_result.test_scores.__dict__.items():
                    if not metric_name.startswith('_'):
                        print(f"  {metric_name}: {metric_value:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This is expected if Meridian is not installed.")
        print("The adapter will use placeholder implementations for demonstration.\n")
        
        # Demonstrate direct adapter usage
        print("Demonstrating direct adapter usage...")
        from mmm_eval.adapters import get_adapter
        
        try:
            adapter = get_adapter("meridian", config)
            
            # Fit the model
            print("Fitting model...")
            adapter.fit(data)
            print("Model fitted successfully!")
            
            # Make predictions
            print("Making predictions...")
            predictions = adapter.predict(data.head(10))
            print(f"Predictions shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:5]}")
            
            # Get channel ROIs
            print("Getting channel ROIs...")
            rois = adapter.get_channel_roi()
            print("Channel ROIs:")
            for channel, roi in rois.items():
                print(f"  {channel}: {roi:.2f}%")
                
        except Exception as e:
            print(f"Direct adapter usage failed: {e}")


if __name__ == "__main__":
    main() 