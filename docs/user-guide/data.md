# Data

mmm-eval expects your data to be in a specific format. This guide explains the required structure and provides examples for preparing your marketing mix modeling data.

## Data Requirements

### Quantity Requirements

* **Observations**: At least 40 data points required, but 100+ is highly recommended
* **Frequency**: Consistent frequency with no gaps (weekly is recommended)
* **Media channels**: At least 2-3 channels for meaningful analysis
* **Revenue data**: Required for ROI calculations

### Quality Requirements

#### Completeness
* No missing values in required columns
* Complete time series (no gaps in dates)
* Non-negative values for spend columns

#### Consistency
- Same units throughout (e.g., all spend in same currency)
- Consistent date format
- Consistent naming conventions

#### Reasonableness
- Values within expected ranges
- No obvious outliers or errors
- Logical relationships between variables

## Data Format

mmm-eval accepts CSV and Parquet files with the following structure:

### Required Columns

- **Date column**: Time series data with consistent date format
- **Target column**: The variable you want to predict (e.g., sales, conversions)
- **Revenue column**: Revenue data for calculating ROI and efficiency metrics
- **Media columns**: Marketing channel spend or activity data

### Optional columns

- **Control columns**: Additional variables that may affect the target

!!! info "Meridian Data Inputs"
    In addition to the above, Meridian supports multiple types of controls and media treatments, as well as a geography field.
    For details, see the Meridian notebook in the `examples/` directory for a full walkthrough, as well as
    their documentation [here](https://developers.google.com/meridian/docs/user-guide/supported-data-types-formats?hl=en).

### Column Types

#### Date Column
- **Purpose**: Identifies the time period for each observation
- **Format**: Date in a consistent format (e.g., YYYY-MM-DD, MM/DD/YYYY)
- **Requirements**: 
  - Must be in chronological order
  - No missing dates in the series
  - Consistent format throughout

#### Target Column
- **Purpose**: The dependent variable you want to model (e.g., sales, conversions)
- **Format**: Numeric values
- **Requirements**:
  - No missing values
  - Positive values (for most use cases)
  - Reasonable scale for your business

#### Revenue Column
- **Purpose**: Revenue data for calculating ROI and efficiency metrics
- **Format**: Numeric values
- **Requirements**:
  - No missing values
  - Positive values
  - Same time period as target column
  - Used for ROI calculations and efficiency analysis

#### Media Columns
- **Purpose**: Marketing channel spend or activity data
- **Format**: Numeric values
- **Requirements**:
  - No missing values (use 0 for periods with no spend)
  - Non-negative values
  - Consistent units (e.g., all in dollars, all in thousands)

#### Control Columns
- **Purpose**: Additional variables that may affect the target
- **Format**: Numeric or categorical values
- **Examples**: Price, seasonality indicators, holiday flags, competitor activity

## Example Data Structure

### Basic Example

```csv
date,sales,revenue,tv_spend,digital_spend,print_spend
2023-01-01,1000,7000,5000,2000,1000
2023-01-02,1200,8000,5500,2200,1100
2023-01-03,1100,7500,5200,2100,1050
2023-01-04,1300,9000,6000,2400,1200
2023-01-05,1400,9500,6500,2600,1300
```

### Advanced Example with Controls

```csv
date,sales,revenue,tv_spend,digital_spend,print_spend,price,seasonality,holiday
2023-01-01,1000,7000,5000,2000,1000,10.99,0.8,0
2023-01-02,1200,8000,5500,2200,1100,10.99,0.9,0
2023-01-03,1100,7500,5200,2100,1050,11.99,0.7,1
2023-01-04,1300,9000,6000,2400,1200,11.99,0.8,0
2023-01-05,1400,9500,6500,2600,1300,12.99,0.9,0
```

### Real-World Example

```csv
date,sales,revenue,tv_spend,digital_spend,social_spend,search_spend,email_spend,price,holiday_flag,competitor_promo
2023-01-01,1250,8750,15000,8000,3000,5000,2000,12.99,0,0
2023-01-02,1320,9240,16000,8500,3200,5200,2100,12.99,0,0
2023-01-03,1180,8260,14000,7500,2800,4800,1900,13.99,1,1
2023-01-04,1450,10150,18000,9500,3800,6000,2400,13.99,0,0
2023-01-05,1520,10640,19000,10000,4000,6300,2500,14.99,0,0
```

## Date Formats

Please provide your dates in ISO format, e.g. `YYYY-MM-DD`.

## Validation

mmm-eval performs several validation checks:

### Automatic Validation

1. **Missing values**: Checks for missing data in required columns
2. **Date consistency**: Ensures dates are in chronological order
3. **Data types**: Verifies numeric columns contain valid numbers
4. **Value ranges**: Checks for negative values in spend columns
5. **Revenue consistency**: Ensures revenue data is available and positive

## Data Preparation Tips

### Before Running mmm-eval

#### Clean your data

- Remove any test or dummy data
- Handle missing values appropriately
- Check for and remove outliers if necessary

#### Standardize formats

- Ensure consistent date format
- Use consistent units (e.g., thousands of dollars)
- Standardize column names

#### Validate relationships

- Check that spend and sales have logical relationships
- Verify that revenue data is consistent with sales
- Ensure control variables make sense

### Common Issues and Solutions

#### Missing Values
```csv
# Problem: Missing values in media columns
date,sales,revenue,tv_spend,digital_spend
2023-01-01,1000,7000,5000,
2023-01-02,1200,8000,,2000

# Solution: Fill with zeros or appropriate values
date,sales,revenue,tv_spend,digital_spend
2023-01-01,1000,7000,5000,0
2023-01-02,1200,8000,0,2000
```

#### Inconsistent Date Formats
```csv
# Problem: Mixed date formats
date,sales,revenue
01/01/2023,1000,7000
2023-01-02,1200,8000

# Solution: Standardize to one format
date,sales,revenue
2023-01-01,1000,7000
2023-01-02,1200,8000
```

#### Missing Revenue Data
```csv
# Problem: No revenue column
date,sales,tv_spend,digital_spend
2023-01-01,1000,5000,2000
2023-01-02,1200,5500,2200

# Solution: Add revenue column
date,sales,revenue,tv_spend,digital_spend
2023-01-01,1000,7000,5000,2000
2023-01-02,1200,8000,5500,2200
```

## Best Practices

### Data Collection

- **Consistent timing**: Collect data at the same time each period
- **Complete coverage**: Ensure all channels are captured
- **Quality control**: Implement data validation at source
- **Documentation**: Keep records of any data changes or anomalies

### Data Analysis

- **Start simple**: Begin with basic models before adding complexity
- **Validate assumptions**: Check that your data meets model requirements
- **Monitor quality**: Regularly review data for issues
- **Document decisions**: Keep records of data preparation choices

## Troubleshooting

### Common Error Messages

- **"Missing required column"**: Ensure all required columns are present
- **"Invalid date format"**: Check your date format specification
- **"Negative values in spend columns"**: Replace negative values with zeros
- **"Missing revenue data"**: Add revenue column to your dataset

### Getting Help

If you encounter data format issues:

- Check the [CLI Reference](cli.md) for all available options
- Review the [Examples](examples/basic-usage.md) for similar use cases
- Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support 