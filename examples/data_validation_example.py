"""
Example of using the clean three-layer data architecture.
"""

import pandas as pd
from pathlib import Path
from mmm_eval.data import DataLoader, DataProcessor, DataValidator


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


def example_three_layer_architecture():
    """Example of the clean three-layer architecture."""
    print("=== Three-Layer Architecture Example ===")
    
    # Create sample CSV file
    df = create_sample_data()
    csv_path = Path("sample_mmm_data.csv")
    df.to_csv(csv_path, index=False)
    
    try:
        # Layer 1: Load data
        print("1. Loading data...")
        loader = DataLoader(csv_path)
        raw_df = loader.load()
        print(f"   Raw data shape: {raw_df.shape}")
        print(f"   Date column type: {raw_df['date'].dtype}")
        
        # Layer 2: Process data
        print("\n2. Processing data...")
        processor = DataProcessor(parse_dates=True, sort_by_date=True)
        processed_df = processor.process(raw_df)
        print(f"   Processed data shape: {processed_df.shape}")
        print(f"   Date column type: {processed_df['date'].dtype}")
        print(f"   Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
        
        # Layer 3: Validate data
        print("\n3. Validating data...")
        validator = DataValidator(require_no_nulls=True, min_data_size=21)
        result = validator.run_validations(processed_df)
        print(f"   Validation passed: {result.is_valid}")
        if result.errors:
            print(f"   Errors: {result.errors}")
        if result.warnings:
            print(f"   Warnings: {result.warnings}")
        
    except Exception as e:
        print(f"Failed: {e}")
    finally:
        # Clean up
        if csv_path.exists():
            csv_path.unlink()
    print()


def example_custom_processing():
    """Example with custom processing settings."""
    print("=== Custom Processing Example ===")
    
    # Create data with multiple date columns
    df = create_sample_data()
    df['created_at'] = df['date']  # Add another date column
    
    # Process with custom settings
    processor = DataProcessor(
        parse_dates=True,
        date_columns=['date', 'created_at'],
        sort_by_date=False  # Don't sort
    )
    
    processed_df = processor.process(df)
    print(f"Processed data shape: {processed_df.shape}")
    print(f"Date columns: {[col for col in processed_df.columns if processed_df[col].dtype == 'datetime64[ns]']}")
    print()


def example_individual_layers():
    """Example using each layer individually."""
    print("=== Individual Layers Example ===")
    
    # Create sample data
    df = create_sample_data()
    
    # Just processing
    processor = DataProcessor(parse_dates=True)
    processed_df = processor.process(df)
    print(f"Just processing: date type = {processed_df['date'].dtype}")
    
    # Just validation
    validator = DataValidator(require_no_nulls=False)
    result = validator.run_validations(df)  # Validate raw data
    print(f"Just validation: passed = {result.is_valid}")
    print()


if __name__ == "__main__":
    example_three_layer_architecture()
    example_custom_processing()
    example_individual_layers() 