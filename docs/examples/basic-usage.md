# Basic Usage Examples

This guide provides practical examples of how to use mmm-eval for different scenarios.

For the sake of simplicity, all examples below exhibit use of PyMC-marketing, but the
takeaways still apply when using other frameworks. For examples of how to configure
a Meridian model, see the example notebook in the `examples/` directory.

## Example 1: Basic Evaluation

```bash
mmm-eval \
  --input-data-path marketing_data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path results/
```

This assumes your data has standard column names and a valid configuration file.

## Example 2: Custom Configuration

Create a configuration file `config.json`:

```json
{
  "pymc_model_config": {
    "date_column": "date_week",
    "channel_columns": ["tv_spend", "digital_spend", "print_spend"],
    "control_columns": ["price", "seasonality", "holiday"],
    "adstock": "GeometricAdstock(l_max=4)",
    "saturation": "LogisticSaturation()",
    "yearly_seasonality": 2
  },
  "fit_config": {
    "target_accept": 0.9,
    "draws": 100,
    "tune": 50,
    "chains": 2,
    "random_seed": 42
  },
  "revenue_column": "revenue",
  "response_column": "sales"
}
```

Run the evaluation:

```bash
mmm-eval \
  --input-data-path marketing_data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./results/
```

## Example 3: Specific Tests Only

Run only certain validation tests:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./results/ \
  --test-names accuracy cross_validation
```

Available tests:
- `accuracy` - Model accuracy using holdout validation
- `cross_validation` - Time series cross-validation
- `refresh_stability` - Model stability over time
- `perturbation` - Sensitivity to data changes

## Example 4: Verbose Output

Get detailed information during execution:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./results/ \
  --verbose
```

## Example 5: Advanced Configuration

For more complex models, use an advanced configuration:

```json
{
  "pymc_model_config": {
    "date_column": "date_week",
    "channel_columns": ["tv_spend", "digital_spend", "print_spend", "radio_spend"],
    "control_columns": ["price", "seasonality", "holiday", "competitor_promo"],
    "adstock": "WeibullAdstock(l_max=6)",
    "saturation": "HillSaturation()",
    "yearly_seasonality": 4,
    "time_varying_intercept": true,
    "time_varying_media": false
  },
  "fit_config": {
    "target_accept": 0.95,
    "draws": 2000,
    "tune": 1000,
    "chains": 4,
    "random_seed": 123,
    "progress_bar": true,
    "return_inferencedata": true
  },
  "revenue_column": "revenue",
  "response_column": "sales"
}
```

Run with verbose output:

```bash
mmm-eval \
  --input-data-path marketing_data.csv \
  --framework pymc-marketing \
  --config-path advanced_config.json \
  --test-names accuracy cross_validation refresh_stability perturbation \
  --output-path ./advanced_results/ \
  --verbose
```

## Data Format Examples

### Basic CSV Structure

```csv
date_week,quantity,revenue,channel_1,channel_2,price,event_1,event_2
2023-01-01,1000,7000,5000,2000,10.99,0,0
2023-01-08,1200,8000,5500,2200,10.99,0,0
2023-01-15,1100,7500,5200,2100,11.99,1,0
2023-01-22,1300,9000,6000,2400,11.99,0,1
2023-01-29,1400,9500,6500,2600,12.99,0,0
2023-02-05,1500,10000,7000,2800,12.99,0,0
2023-02-12,1600,10500,7500,3000,13.99,1,0
2023-02-19,1700,11000,8000,3200,13.99,0,1
2023-02-26,1800,11500,8500,3400,14.99,0,0
2023-03-05,1900,12000,9000,3600,14.99,0,0
```

### With More Channels

```csv
date_week,sales,revenue,tv_spend,digital_spend,print_spend,radio_spend,price,seasonality,holiday
2023-01-01,1000,7000,5000,2000,1000,500,10.99,0.8,0
2023-01-08,1200,8000,5500,2200,1100,550,10.99,0.9,0
2023-01-15,1100,7500,5200,2100,1050,520,11.99,0.7,1
2023-01-22,1300,9000,6000,2400,1200,600,11.99,0.8,0
2023-01-29,1400,9500,6500,2600,1300,650,12.99,0.9,0
```

## Expected Output

After running an evaluation, you'll find a CSV file in your output directory:

```
results/
└── mmm_eval_pymc-marketing_20241201_143022.csv
```

### Sample Results

```csv
test_name,metric_name,metric_value,metric_pass
accuracy,mape,15.0,True
accuracy,r_squared,0.85,True
cross_validation,mape,18.0,True
cross_validation,r_squared,0.82,True
refresh_stability,mean_percentage_change_for_each_channel:channel_1,0.05,True
refresh_stability,mean_percentage_change_for_each_channel:channel_2,0.03,True
refresh_stability,std_percentage_change_for_each_channel:channel_1,0.02,True
refresh_stability,std_percentage_change_for_each_channel:channel_2,0.01,True
perturbation,percentage_change_for_each_channel:channel_1,0.02,True
perturbation,percentage_change_for_each_channel:channel_2,0.01,True
```

## Performance Examples

### Quick Testing

For development and testing:

```bash
mmm-eval \
  --input-data-path small_test_data.csv \
  --framework pymc-marketing \
  --config-path test_config.json \
  --output-path ./test_results/ \
  --test-names accuracy
```

Use minimal sampling parameters in your config:
```json
{
  "fit_config": {
    "draws": 50,
    "tune": 25,
    "chains": 1
  }
}
```

### Production Evaluation

For production use:

```bash
mmm-eval \
  --input-data-path production_data.csv \
  --framework pymc-marketing \
  --config-path production_config.json \
  --output-path ./production_results/ \
  --test-names accuracy cross_validation refresh_stability perturbation \
  --verbose
```

Use robust sampling parameters:
```json
{
  "fit_config": {
    "draws": 2000,
    "tune": 1000,
    "chains": 4,
    "target_accept": 0.95
  }
}
```

## Troubleshooting Examples

### Missing Configuration File

If you get a configuration error:

```bash
# Create a basic config file
cat > config.json << 'EOF'
{
  "pymc_model_config": {
    "date_column": "date_week",
    "channel_columns": ["channel_1", "channel_2"],
    "adstock": "GeometricAdstock(l_max=4)",
    "saturation": "LogisticSaturation()"
  },
  "fit_config": {
    "target_accept": 0.9,
    "draws": 100,
    "tune": 50,
    "chains": 2,
    "random_seed": 42
  },
  "revenue_column": "revenue"
}
EOF

# Run evaluation
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./results/
```

### Data Format Issues

If you get data format errors, check your CSV structure:

```bash
# Check your data format
head -5 data.csv

# Ensure required columns exist
python -c "
import pandas as pd
df = pd.read_csv('data.csv')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('Date range:', df['date_week'].min(), 'to', df['date_week'].max())
"
```

## Next Steps

- Learn about [Data](../user-guide/data.md) for different data structures
- Explore [Configuration](../getting-started/configuration.md) for advanced settings
- Check the [CLI Reference](../user-guide/cli.md) for all available options 