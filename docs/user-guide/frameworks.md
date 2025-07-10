# Frameworks

mmm-eval supports multiple Marketing Mix Modeling (MMM) frameworks. This guide explains each supported framework and their features.

## Supported Frameworks

### Google Meridian

Google's Meridian framework provides advanced MMM capabilities with support for reach and frequency data.

#### Features

- **Reach and Frequency**: Support for reach/frequency data in addition to spend
- **Organic Media**: Modeling of organic media channels
- **Non-Media Treatments**: Support for non-media variables
- **Geographic Modeling**: Multi-geography support
- **Advanced Priors**: Sophisticated prior distribution system

#### Usage

```bash
mmm-eval --input-data-path data.csv --config-path config.json --framework meridian --output-path results/
```

#### Configuration

Meridian requires specific configuration for:
- Input data builder settings
- Model specification parameters
- Sample posterior configuration

See [Configuration](../getting-started/configuration.md) for detailed setup.

### PyMC-Marketing

PyMC-Marketing is a Bayesian MMM framework built on PyMC.

#### Features

- **Bayesian Inference**: Full posterior distributions
- **Flexible Adstock**: Multiple adstock transformation options
- **Saturation Models**: Various saturation functions
- **Control Variables**: Support for additional regressors
- **Seasonality**: Built-in seasonal modeling

#### Usage

```bash
mmm-eval --input-data-path data.csv --config-path config.json --framework pymc-marketing --output-path results/
```

#### Configuration

PyMC-Marketing requires configuration for:
- Model parameters (adstock, saturation, etc.)
- Fit parameters (chains, draws, etc.)
- Data mapping

See [Configuration](../getting-started/configuration.md) for detailed setup.

## Framework Comparison

| Feature | Meridian | PyMC-Marketing |
|---------|----------|----------------|
| **Reach/Frequency** | ✅ Supported | ❌ Not supported |
| **Organic Media** | ✅ Supported | ❌ Not supported |
| **Geographic** | ✅ Supported | ❌ Not supported |
| **Bayesian** | ✅ Full posterior | ✅ Full posterior |
| **Adstock** | Built-in | Multiple options |
| **Saturation** | Built-in | Multiple options |
| **Control Variables** | ✅ Supported | ✅ Supported |
| **Seasonality** | Built-in | Fourier modes |

## Choosing a Framework

### When to Use Meridian

- **Reach/Frequency Data**: If you have reach and frequency data
- **Organic Media**: If modeling organic channels is important
- **Geographic Analysis**: If you need multi-geography support
- **Advanced Features**: If you need sophisticated prior distributions

### When to Use PyMC-Marketing

- **Standard MMM**: For traditional spend-based modeling
- **Flexible Modeling**: If you need custom adstock/saturation
- **Bayesian Workflow**: If you're familiar with PyMC
- **Open Source**: If you prefer open-source frameworks

## Framework-Specific Considerations

### Meridian

#### Data Requirements

- **Media Channels**: Channel names and spend columns
- **Reach/Frequency**: Optional reach and frequency columns
- **Organic Media**: Optional organic media columns
- **Non-Media**: Optional non-media treatment columns
- **Geography**: Optional geography column

#### Configuration Complexity

Meridian requires more complex configuration due to:
- Prior distribution specification
- Multiple data input types
- Advanced model parameters

### PyMC-Marketing

#### Data Requirements

- **Media Channels**: Spend columns only
- **Control Variables**: Optional control columns
- **Date Column**: Required for time series
- **Target Column**: Response variable

#### Configuration Simplicity

PyMC-Marketing has simpler configuration:
- Standard adstock/saturation choices
- Basic fit parameters
- Straightforward data mapping

## Migration Between Frameworks

### From PyMC-Marketing to Meridian

1. **Add Reach/Frequency Data**: If available
2. **Configure Priors**: Set up PriorDistribution objects
3. **Update Configuration**: Use Meridian-specific config structure
4. **Test Gradually**: Start with basic configuration

### From Meridian to PyMC-Marketing

1. **Remove Reach/Frequency**: Use spend data only
2. **Simplify Configuration**: Use PyMC-Marketing config structure
3. **Adjust Parameters**: Set adstock and saturation
4. **Validate Results**: Compare performance

## Best Practices

### Framework Selection

- **Start Simple**: Begin with PyMC-Marketing for basic needs
- **Add Complexity**: Move to Meridian for advanced features
- **Data-Driven**: Choose based on available data
- **Expertise**: Consider team familiarity

### Configuration Management

- **Version Control**: Track configuration changes
- **Documentation**: Document framework choices
- **Testing**: Test configurations thoroughly
- **Backup**: Keep working configurations

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Check framework-specific requirements
2. **Data Format Issues**: Verify data matches framework expectations
3. **Performance Problems**: Adjust framework-specific parameters
4. **Memory Issues**: Consider data size and model complexity

### Getting Help

- Check [Configuration](../getting-started/configuration.md) for setup
- Review [Examples](../examples/basic-usage.md) for use cases
- Join [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for support 