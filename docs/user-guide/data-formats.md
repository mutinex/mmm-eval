# Data Formats

mmm-eval expects your data to be in a specific format. This guide explains the required structure and provides examples.

## Required Data Format

mmm-eval accepts CSV files with the following structure:

### Basic Requirements

- **Date column**: Time series data with consistent date format
- **Target column**: The variable you want to predict (e.g., sales, conversions)
- **Media columns**: Marketing channel spend or activity data
- **Control columns** (optional): Additional variables that may affect the target

### Column Types

#### Date Column
- **Purpose**: Identifies the time period for each observation
- **Format**: Date in a consistent format (e.g., YYYY-MM-DD, MM/DD/YYYY)
- **Requirements**: 
  - Must be in chronological order
  - No missing dates in the series
  - Consistent format throughout

#### Target Column
- **Purpose**: The dependent variable you want to model
- **Format**: Numeric values
- **Requirements**:
  - No missing values
  - Positive values (for most use cases)
  - Reasonable scale for your business

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
date,sales,tv_spend,digital_spend,print_spend
2023-01-01,1000,5000,2000,1000
2023-01-02,1200,5500,2200,1100
2023-01-03,1100,5200,2100,1050
2023-01-04,1300,6000,2400,1200
2023-01-05,1400,6500,2600,1300
```

### Advanced Example with Controls

```csv
date,sales,tv_spend,digital_spend,print_spend,price,seasonality,holiday
2023-01-01,1000,5000,2000,1000,10.99,0.8,0
2023-01-02,1200,5500,2200,1100,10.99,0.9,0
2023-01-03,1100,5200,2100,1050,11.99,0.7,1
2023-01-04,1300,6000,2400,1200,11.99,0.8,0
2023-01-05,1400,6500,2600,1300,12.99,0.9,0
```

## Data Requirements

### Minimum Data Requirements

- **Time period**: At least 52 weeks (1 year) of data
- **Frequency**: Weekly or daily data (consistent frequency)
- **Observations**: Minimum 100 data points recommended
- **Media channels**: At least 2-3 channels for meaningful analysis

### Data Quality Requirements

#### Completeness
- No missing values in required columns
- Complete time series (no gaps in dates)
- Consistent data collection methodology

#### Consistency
- Same units throughout (e.g., all spend in same currency)
- Consistent date format
- Consistent naming conventions

#### Reasonableness
- Values within expected ranges
- No obvious outliers or errors
- Logical relationships between variables

## Date Formats

### Supported Formats

mmm-eval supports various date formats:

- `YYYY-MM-DD` (ISO format) - Recommended
- `MM/DD/YYYY`
- `DD-MM-YYYY`
- `YYYY/MM/DD`

### Date Format Specification

If your dates aren't in ISO format, specify the format:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --date-format "%m/%d/%Y"
```

Or in configuration:

```json
{
  "data": {
    "date_format": "%m/%d/%Y"
  }
}
```

## Data Validation

mmm-eval performs several validation checks:

### Automatic Validation

1. **Missing values**: Checks for missing data in required columns
2. **Date consistency**: Ensures dates are in chronological order
3. **Data types**: Verifies numeric columns contain valid numbers
4. **Value ranges**: Checks for negative values in spend columns

### Custom Validation

You can configure additional validation:

```json
{
  "data": {
    "validation": {
      "check_missing_values": true,
      "check_negative_values": true,
      "check_date_range": true,
      "min_date": "2020-01-01",
      "max_date": "2023-12-31"
    }
  }
}
```

## Data Preparation Tips

### Before Running mmm-eval

1. **Clean your data**:
   - Remove any test or dummy data
   - Handle missing values appropriately
   - Check for and remove outliers if necessary

2. **Standardize formats**:
   - Ensure consistent date format
   - Use consistent units (e.g., thousands of dollars)
   - Standardize column names

3. **Validate relationships**:
   - Check that spend and sales have logical relationships
   - Verify that control variables make sense
   - Ensure no data entry errors

### Common Issues and Solutions

#### Missing Values
```csv
# Problem: Missing values in media columns
date,sales,tv_spend,digital_spend
2023-01-01,1000,5000,
2023-01-02,1200,,2000

# Solution: Fill with zeros or appropriate values
date,sales,tv_spend,digital_spend
2023-01-01,1000,5000,0
2023-01-02,1200,0,2000
```

#### Inconsistent Date Formats
```csv
# Problem: Mixed date formats
date,sales
01/01/2023,1000
2023-01-02,1200

# Solution: Standardize to one format
date,sales
2023-01-01,1000
2023-01-02,1200
```

#### Negative Values
```csv
# Problem: Negative spend values
date,sales,tv_spend
2023-01-01,1000,-5000

# Solution: Use absolute values or zero
date,sales,tv_spend
2023-01-01,1000,0
```

## File Size and Performance

### Recommended Limits

- **File size**: Up to 100MB for CSV files
- **Rows**: Up to 10,000 observations
- **Columns**: Up to 50 columns total

### Performance Considerations

- Larger datasets will take longer to process
- More media channels increase computation time
- Complex models (e.g., with many control variables) require more time

## Next Steps

- Learn about [Configuration](getting-started/configuration.md) for customizing data handling
- Check [Examples](examples/basic-usage.md) for practical data preparation
- Review [CLI Reference](cli.md) for command-line options 