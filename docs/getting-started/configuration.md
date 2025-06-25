# Configuration

mmm-eval can be customized through configuration files to control data processing, test parameters, and output settings.

## Configuration File Format

mmm-eval uses JSON configuration files. You can specify a configuration file using the `--config-path` option:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json
```

## Configuration Structure

A complete configuration file has the following structure:

```json
{
  "data": {
    "date_column": "date",
    "target_column": "sales",
    "media_columns": ["tv_spend", "digital_spend", "print_spend"],
    "control_columns": ["price", "seasonality"],
    "date_format": "%Y-%m-%d",
    "validation": {
      "check_missing_values": true,
      "check_negative_values": true,
      "check_date_range": true
    }
  },
  "framework": {
    "pymc_marketing": {
      "model_config": {
        "date_column": "date",
        "target_column": "sales",
        "media_columns": ["tv_spend", "digital_spend", "print_spend"],
        "control_columns": ["price", "seasonality"]
      }
    }
  },
  "tests": {
    "accuracy": {
      "train_test_split": 0.8,
      "random_state": 42,
      "metrics": ["mape", "rmse", "r2", "mae"]
    },
    "cross_validation": {
      "folds": 5,
      "random_state": 42,
      "metrics": ["mape", "rmse", "r2", "mae"]
    },
    "refresh_stability": {
      "refresh_periods": [0.5, 0.75, 0.9],
      "metrics": ["mape", "rmse", "r2", "mae"]
    },
    "perturbation": {
      "perturbation_levels": [0.05, 0.1, 0.15],
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  },
  "output": {
    "format": "json",
    "include_plots": true,
    "plot_format": "png",
    "save_intermediate_results": false
  }
}
```

## Data Configuration

### Basic Data Settings

```json
{
  "data": {
    "date_column": "date",
    "target_column": "sales",
    "media_columns": ["tv_spend", "digital_spend", "print_spend"],
    "control_columns": ["price", "seasonality"]
  }
}
```

- **date_column**: Name of the column containing dates
- **target_column**: Name of the column containing the target variable
- **media_columns**: List of column names for marketing channels
- **control_columns**: List of column names for control variables (optional)

### Date Format

Specify the date format if your dates aren't in ISO format:

```json
{
  "data": {
    "date_format": "%Y-%m-%d"
  }
}
```

Common formats:
- `%Y-%m-%d` - 2023-01-01
- `%m/%d/%Y` - 01/01/2023
- `%d-%m-%Y` - 01-01-2023

### Data Validation

Control data validation settings:

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

## Framework Configuration

### PyMC-Marketing Settings

```json
{
  "framework": {
    "pymc_marketing": {
      "model_config": {
        "date_column": "date",
        "target_column": "sales",
        "media_columns": ["tv_spend", "digital_spend", "print_spend"],
        "control_columns": ["price", "seasonality"],
        "seasonality": {
          "yearly_seasonality": 10,
          "weekly_seasonality": 3
        }
      }
    }
  }
}
```

## Test Configuration

### Accuracy Test

```json
{
  "tests": {
    "accuracy": {
      "train_test_split": 0.8,
      "random_state": 42,
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

- **train_test_split**: Proportion of data for training (0.0 to 1.0)
- **random_state**: Random seed for reproducibility
- **metrics**: List of metrics to calculate

### Cross-Validation Test

```json
{
  "tests": {
    "cross_validation": {
      "folds": 5,
      "random_state": 42,
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

- **folds**: Number of cross-validation folds
- **random_state**: Random seed for reproducibility
- **metrics**: List of metrics to calculate

### Refresh Stability Test

```json
{
  "tests": {
    "refresh_stability": {
      "refresh_periods": [0.5, 0.75, 0.9],
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

- **refresh_periods**: List of proportions for refresh periods
- **metrics**: List of metrics to calculate

### Perturbation Test

```json
{
  "tests": {
    "perturbation": {
      "perturbation_levels": [0.05, 0.1, 0.15],
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

- **perturbation_levels**: List of perturbation levels (as proportions)
- **metrics**: List of metrics to calculate

## Output Configuration

```json
{
  "output": {
    "format": "json",
    "include_plots": true,
    "plot_format": "png",
    "save_intermediate_results": false
  }
}
```

- **format**: Output format ("json" or "csv")
- **include_plots**: Whether to generate plots
- **plot_format**: Plot file format ("png", "pdf", "svg")
- **save_intermediate_results**: Whether to save intermediate test results

## Available Metrics

The following metrics are available for all tests:

- **mape**: Mean Absolute Percentage Error
- **rmse**: Root Mean Square Error
- **r2**: R-squared (coefficient of determination)
- **mae**: Mean Absolute Error
- **mse**: Mean Square Error

## Environment Variables

You can also use environment variables for configuration:

```bash
export MMM_EVAL_DATE_COLUMN=date
export MMM_EVAL_TARGET_COLUMN=sales
export MMM_EVAL_MEDIA_COLUMNS=tv_spend,digital_spend,print_spend
```

## Configuration Precedence

Configuration is applied in the following order (later takes precedence):

1. Default configuration
2. Environment variables
3. Configuration file
4. Command line arguments

## Example Configurations

### Minimal Configuration

```json
{
  "data": {
    "date_column": "date",
    "target_column": "sales",
    "media_columns": ["tv_spend", "digital_spend"]
  }
}
```

### Advanced Configuration

```json
{
  "data": {
    "date_column": "date",
    "target_column": "sales",
    "media_columns": ["tv_spend", "digital_spend", "print_spend"],
    "control_columns": ["price", "seasonality"],
    "date_format": "%Y-%m-%d",
    "validation": {
      "check_missing_values": true,
      "check_negative_values": true
    }
  },
  "tests": {
    "accuracy": {
      "train_test_split": 0.8,
      "random_state": 42
    },
    "cross_validation": {
      "folds": 5,
      "random_state": 42
    }
  },
  "output": {
    "include_plots": true,
    "plot_format": "png"
  }
}
```

## Next Steps

- Learn about [Data Formats](user-guide/data-formats.md) for different data structures
- Explore [Examples](examples/custom-configuration.md) for configuration use cases
- Check the [CLI Reference](user-guide/cli.md) for command-line options 