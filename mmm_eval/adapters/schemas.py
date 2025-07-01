from typing import Annotated, Any

from pydantic import BaseModel, Field, InstanceOf, field_validator
from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation


class PyMCFitSchema(BaseModel):
    """Schema for PyMC Fit Configuration.

    Defaults are all set to None so that the user can provide only the values they want to change.
    If a user does not provide a value, we will let the latest PYMC defaults be used in model instantiation.
    """

    draws: int | None = Field(None, description="Number of posterior samples to draw.")
    tune: int | None = Field(None, description="Number of tuning (warm-up) steps.")
    chains: int | None = Field(None, description="Number of MCMC chains to run.")
    target_accept: float | None = Field(None, ge=0.0, le=1.0, description="Target acceptance rate for the sampler.")
    random_seed: int | None = Field(None, description="Random seed for reproducibility.")
    progressbar: bool = Field(False, description="Whether to display the progress bar.")
    return_inferencedata: bool = Field(True, description="Whether to return arviz.InferenceData.")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "coerce_types_to_string": False,  # Allow type coercion
    }

    @property
    def fit_config_dict_without_non_provided_fields(self) -> dict[str, Any]:
        """Return only non-None values.

           These are the values that are provided by the user.
           We don't want to include the default values as they should be set by the latest PYMC

        Returns
            Dictionary of non-None values

        """
        return {key: value for key, value in self.model_dump().items() if value is not None}


class PyMCModelSchema(BaseModel):
    """Schema for PyMC Config Dictionary."""

    date_column: str = Field(..., description="Column name of the date variable.")
    channel_columns: list[str] = Field(min_length=1, description="Column names of the media channel variables.")
    adstock: InstanceOf[AdstockTransformation] = Field(..., description="Type of adstock transformation to apply.")
    saturation: InstanceOf[SaturationTransformation] = Field(
        ..., description="Type of saturation transformation to apply."
    )
    time_varying_intercept: bool = Field(False, description="Whether to consider time-varying intercept.")
    time_varying_media: bool = Field(False, description="Whether to consider time-varying media contributions.")
    model_config: dict | None = Field(None, description="Model configuration.")
    sampler_config: dict | None = Field(None, description="Sampler configuration.")
    validate_data: bool = Field(True, description="Whether to validate the data before fitting to model")
    control_columns: (
        Annotated[
            list[str],
            Field(
                min_length=1,
                description="Column names of control variables to be added as additional regressors",
            ),
        ]
        | None
    ) = None
    yearly_seasonality: (
        Annotated[
            int,
            Field(gt=0, description="Number of Fourier modes to model yearly seasonality."),
        ]
        | None
    ) = None
    adstock_first: bool = Field(True, description="Whether to apply adstock first.")
    dag: str | None = Field(
        None,
        description="Optional DAG provided as a string Dot format for causal identification.",
    )
    treatment_nodes: list[str] | tuple[str] | None = Field(
        None,
        description="Column names of the variables of interest to identify causal effects on outcome.",
    )
    outcome_node: str | None = Field(None, description="Name of the outcome variable.")

    @field_validator("channel_columns")
    def validate_channel_columns(cls, v):
        """Validate channel columns are not empty.

        Args:
            v: Channel columns value

        Returns:
            Validated value

        Raises:
            ValueError: If channel columns is empty

        """
        if v is not None and not v:
            raise ValueError("channel_columns must not be empty")
        return v

    @field_validator("adstock")
    def validate_adstock(cls, v):
        """Validate adstock component.

        Args:
            v: Adstock value

        Returns:
            Validated value

        Raises:
            ValueError: If adstock is not a valid type

        """
        if v is not None:
            assert isinstance(v, AdstockTransformation)
        return v

    @field_validator("saturation")
    def validate_saturation(cls, v):
        """Validate saturation component.

        Args:
            v: Saturation value

        Returns:
            Validated value

        Raises:
            ValueError: If saturation is not a valid type

        """
        if v is not None:
            assert isinstance(v, SaturationTransformation)
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # Allow extra fields not defined in schema
        "coerce_types_to_string": False,  # Allow type coercion
    }


class PyMCStringConfigSchema(BaseModel):
    """Schema for PyMC Evaluation Config Dictionary."""

    model_config: dict[str, Any] = Field(..., description="Model configuration.")
    fit_config: dict[str, Any] = Field(..., description="Fit configuration.")
    response_column: str = Field(..., description="Name of the target column.")
    revenue_column: str | None = Field(None, description="Name of the revenue column.")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "coerce_types_to_string": False,  # Allow type coercion
    }


# Meridian-specific schemas
class MeridianPriorDistributionSchema(BaseModel):
    """Schema for Meridian prior distribution configuration."""

    roi_mu: float = Field(0.0, description="Mean of the log-normal ROI prior distribution.")
    roi_sigma: float = Field(1.0, ge=0.0, description="Standard deviation of the log-normal ROI prior distribution.")
    name: str = Field("roi_m", description="Name of the ROI parameter.")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "coerce_types_to_string": False,
    }


class MeridianModelSpecSchema(BaseModel):
    """Schema for Meridian ModelSpec configuration."""

    prior: MeridianPriorDistributionSchema = Field(..., description="Prior distribution configuration.")
    media_effects_dist: str = Field("log_normal", description="Distribution type for media effects.")
    hill_before_adstock: bool = Field(False, description="Whether to apply Hill transformation before adstock.")
    max_lag: int | None = Field(8, description="Maximum lag for adstock transformation.")
    unique_sigma_for_each_geo: bool = Field(False, description="Whether to use unique sigma for each geography.")
    media_prior_type: str | None = Field(None, description="Prior type for media variables.")
    rf_prior_type: str | None = Field(None, description="Prior type for reach and frequency variables.")
    paid_media_prior_type: str | None = Field(None, description="Prior type for paid media variables.")
    roi_calibration_period: list[float] | None = Field(None, description="ROI calibration period array.")
    rf_roi_calibration_period: list[float] | None = Field(None, description="Reach and frequency ROI calibration period array.")
    organic_media_prior_type: str = Field("contribution", description="Prior type for organic media variables.")
    organic_rf_prior_type: str = Field("contribution", description="Prior type for organic reach and frequency variables.")
    non_media_treatments_prior_type: str = Field("contribution", description="Prior type for non-media treatment variables.")
    non_media_baseline_values: list[float | str] | None = Field(None, description="Baseline values for non-media variables.")
    knots: int | list[int] | None = Field(None, description="Knots for spline transformations.")
    baseline_geo: int | str | None = Field(None, description="Baseline geography identifier.")
    holdout_id: list[int] | None = Field(None, description="Holdout period identifiers.")
    control_population_scaling_id: list[int] | None = Field(None, description="Control population scaling identifiers.")
    non_media_population_scaling_id: list[int] | None = Field(None, description="Non-media population scaling identifiers.")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "coerce_types_to_string": False,
    }


class MeridianFitSchema(BaseModel):
    """Schema for Meridian fit configuration."""

    n_chains: int | list[int] = Field(4, description="Number of MCMC chains to run.")
    n_adapt: int = Field(500, description="Number of adaptation steps.")
    n_burnin: int = Field(500, description="Number of burn-in steps.")
    n_keep: int = Field(1000, description="Number of posterior samples to keep.")
    current_state: dict[str, Any] | None = Field(None, description="Current state for MCMC sampling.")
    init_step_size: int | None = Field(None, description="Initial step size for MCMC.")
    dual_averaging_kwargs: dict[str, int] | None = Field(None, description="Dual averaging parameters.")
    max_tree_depth: int = Field(10, description="Maximum tree depth for NUTS sampler.")
    max_energy_diff: float = Field(500.0, description="Maximum energy difference for NUTS sampler.")
    unrolled_leapfrog_steps: int = Field(1, description="Number of unrolled leapfrog steps.")
    parallel_iterations: int = Field(10, description="Number of parallel iterations.")
    seed: int | list[int] | None = Field(None, description="Random seed for reproducibility.")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "coerce_types_to_string": False,
    }

    @property
    def fit_config_dict_without_non_provided_fields(self) -> dict[str, Any]:
        """Return only non-None values.

        Returns:
            Dictionary of non-None values

        """
        return {key: value for key, value in self.model_dump().items() if value is not None}


class MeridianModelSchema(BaseModel):
    """Schema for Meridian model configuration."""

    date_column: str = Field(..., description="Column name of the date variable.")
    media_channels: list[str] = Field(min_length=1, description="Column names of the media channel variables.")
    channel_spend_columns: list[str] = Field(min_length=1, description="Column names of the media channel metric variables.")
    channel_impressions_columns: list[str] | None = Field(None, description="Column names of the media channel impressions variables.")
    channel_reach_columns: list[str] | None = Field(None, description="Column names of the media channel reach variables.")
    channel_frequency_columns: list[str] | None = Field(None, description="Column names of the media channel frequency variables.")

    organic_media_columns: list[str] | None = Field(None, description="Column names of the organic media variables.")
    organic_media_channels: list[str] | None = Field(None, description="Channel names of the organic media variables.")
    non_media_treatment_columns: list[str] | None = Field(None, description="Column names of the non-media treatment variables.")
    
    response_column: str = Field(..., description="Column name of the response variable.")
    control_columns: list[str] | None = Field(None, description="Column names of control variables.")
    # Add other model parameters as needed

    @field_validator("media_channels")
    def validate_media_channels(cls, v):
        """Validate media columns are not empty.

        Args:
            v: Media columns value

        Returns:
            Validated value

        Raises:
            ValueError: If media columns is empty

        """
        if v is not None and not v:
            raise ValueError("media_channels must not be empty")
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "coerce_types_to_string": False,
    }


class MeridianStringConfigSchema(BaseModel):
    """Schema for Meridian evaluation config dictionary."""

    model_config: dict[str, Any] = Field(..., description="Model configuration.")
    fit_config: dict[str, Any] = Field(..., description="Fit configuration.")
    response_column: str = Field(..., description="Name of the target column.")
    revenue_column: str | None = Field(None, description="Name of the revenue column.")

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "coerce_types_to_string": False,
    }
