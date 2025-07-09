# Frameworks

BenjaMMMin supports multiple Marketing Mix Modeling (MMM) frameworks. This guide explains each supported framework and their features.

## Supported Frameworks

### Google Meridian

**Status**: ✅ Fully Supported

[Google Meridian](https://developers.google.com/meridian) is Google's open-source MMM framework that provides fast Bayesian inference and production-ready implementation.

#### Features

- **Fast Bayesian inference**: Optimized MCMC sampling
- **Production-ready**: Designed for large-scale deployments
- **Built-in seasonality**: Automatic seasonal component handling
- **Media effects**: Configurable adstock and saturation functions
- **Control variables**: Support for external factors
- **Uncertainty quantification**: Full Bayesian uncertainty estimates

#### CLI Usage

```bash
benjammmin --input-data-path data.csv --config-path config.json --framework meridian --output-path results/
```

#### Advantages

- **Speed**: Optimized for fast inference on large datasets
- **Production-ready**: Built for enterprise-scale deployments
- **Google-backed**: Supported by Google's MMM expertise
- **Comprehensive**: Full Bayesian uncertainty quantification
- **Scalable**: Handles large datasets efficiently

#### Limitations

- **Less flexible**: More constrained model structure compared to PyMC-Marketing
- **Google-specific**: Optimized for Google's use cases
- **Learning curve**: Different API compared to other frameworks


### PyMC-Marketing

**Status**: ✅ Fully Supported

[PyMC-Marketing](https://www.pymc-marketing.io/en/latest/guide/) is a Bayesian MMM framework built on PyMC that provides robust statistical modeling capabilities.

#### Features

- **Bayesian inference**: Probabilistic modeling with uncertainty quantification
- **Flexible modeling**: Customizable model structures
- **Seasonality handling**: Built-in seasonal components
- **Media effects**: Configurable adstock and saturation functions
- **Control variables**: Support for external factors

#### CLI Usage

```bash
benjammmin --input-data-path data.csv --config-path config.json --framework pymc-marketing --output-path results/
```

#### Advantages

- **Statistical rigor**: Bayesian approach provides uncertainty estimates
- **Flexibility**: Highly customizable model structures
- **Interpretability**: Clear parameter interpretation
- **Robustness**: Handles various data scenarios well

#### Limitations

- **Computational cost**: Can be slower than simpler approaches
- **Complexity**: Requires more expertise to configure optimally
- **Data requirements**: Needs sufficient data for reliable inference

## Framework Comparison

| Feature         | PyMC-Marketing     | Google Meridian     | Robyn*              |
|-----------------|--------------------|-----------------------|---------------------|
| **Type**        | Bayesian           | Bayesian              | Frequentist         |
| **Inference**   | MCMC               | MCMC                  | Ridge Regression + Bootstrapping |
| **Seasonality** | ✅ Built-in        | ✅ Built-in           | ✅ Built-in         |
| **Saturation**  | ✅ Configurable    | ✅ Configurable       | ✅ Configurable     |
| **Controls**    | ✅ Supported       | ✅ Supported          | ✅ Supported        |
| **Uncertainty** | ✅ Full            | ✅ Full               | ⚠️ Approximate via bootstrapping |
| **Speed**       | Fast               | Medium                | Medium              |
| **Complexity**  | Low                | Medium                | Medium              |

*Planned for future releases

### Getting Help

For framework-specific issues:

1. **Check documentation**: Review framework-specific guides
2. **Review examples**: Look at example configurations
3. **Check data**: Ensure data meets requirements
4. **Community support**: Use GitHub discussions for help

## Future Frameworks

### Planned Support

**Robyn**:

- Meta's MMM framework
- Automated model selection
- Built-in validation

### Framework Requests

To request support for additional frameworks:

1. **Create an issue** on GitHub
2. **Provide details** about the framework
3. **Include use cases** and requirements
4. **Contribute** if possible

## Next Steps

- Learn about [Configuration](getting-started/configuration.md) for framework setup
- Check [Examples](examples/basic-usage.md) for framework usage
- Review [Tests](tests.md) for validation approaches 