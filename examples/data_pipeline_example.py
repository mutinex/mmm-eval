"""
Example of using the DataPipeline for end-to-end data processing.
"""

import pandas as pd
from pathlib import Path
from mmm_eval.data import DataPipeline


def create_sample_data():
    """Create sample MMM data for demonstration."""
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    
    data = []
    for date in dates:
        for channel in ['facebook', 'google', 'tv']:
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'media_channel': channel,
                'media_channel_spend': 1000 + (hash(f"{date}{channel}") % 500),
                'media_channel_revenue': 2000 + (hash(f"{date}{channel}") % 1000),
                'media_channel_volume_contribution': 0.3 + (hash(f"{date}{channel}") % 100) / 1000
            })
    
    return pd.DataFrame(data)


def example_simple_pipeline():
    """Example of simple pipeline usage."""
    print("=== Simple Pipeline Example ===")
    
    # Create sample CSV file
    df = create_sample_data()
    csv_path = Path("sample_mmm_data.csv")
    df.to_csv(csv_path, index=False)
    
    try:
        # Create and run pipeline
        pipeline = DataPipeline(
            data_path=csv_path,
            parse_dates=True,
            require_no_nulls=True
        )
        
        # Run the complete pipeline
        processed_df = pipeline.run()
        
        print(f"Pipeline completed successfully!")
        print(f"Final DataFrame shape: {processed_df.shape}")
        print(f"Date column type: {processed_df['date'].dtype}")
        print(f"Columns: {list(processed_df.columns)}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        # Clean up
        if csv_path.exists():
            csv_path.unlink()
    print()


def example_custom_pipeline():
    """Example with custom pipeline settings."""
    print("=== Custom Pipeline Example ===")
    
    # Create sample CSV file
    df = create_sample_data()
    csv_path = Path("sample_mmm_data.csv")
    df.to_csv(csv_path, index=False)
    
    try:
        # Create pipeline with custom settings
        pipeline = DataPipeline(
            data_path=csv_path,
            parse_dates=True,
            date_column="date",
            require_no_nulls=False,  # Allow nulls
            min_data_size=10  # Lower minimum size
        )
        
        processed_df = pipeline.run()
        print(f"Custom pipeline completed!")
        print(f"DataFrame shape: {processed_df.shape}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        # Clean up
        if csv_path.exists():
            csv_path.unlink()
    print()


if __name__ == "__main__":
    example_simple_pipeline()
    example_custom_pipeline() 