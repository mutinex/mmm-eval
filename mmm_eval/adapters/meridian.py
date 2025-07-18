"""Google Meridian adapter for MMM evaluation."""

import gc
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from meridian.analysis.analyzer import Analyzer
from meridian.data import data_frame_input_data_builder as data_builder
from meridian.model.model import Meridian
from meridian.model.spec import ModelSpec

from mmm_eval.adapters.base import BaseAdapter, PrimaryMediaRegressor
from mmm_eval.adapters.schemas import MeridianInputDataBuilderSchema
from mmm_eval.configs import MeridianConfig
from mmm_eval.data.constants import InputDataframeConstants

logger = logging.getLogger(__name__)

# Meridian requires revenue to be specified in this form, but we accept raw revenue as
# input for continuity between frameworks
REVENUE_PER_KPI_COL = "revenue_per_kpi"


# TODO: move to validation pipeline
def _validate_media_channels(df: pd.DataFrame, config: MeridianConfig) -> None:
    """Validate that media channels have sufficient variation for Meridian modeling.

    Meridian requires media channels to have non-zero spend across time periods and geos
    to estimate meaningful parameters. Channels with zero spend everywhere will cause
    the model to fail or produce unreliable results.

    Args:
        df: Input DataFrame containing media spend data
        config: MeridianConfig object containing media channel specifications

    Raises:
        ValueError: If any media channel has zero spend across all time periods and geos

    """
    input_config = config.input_data_builder_config

    # Check spend columns for each media channel
    for channel, spend_col in zip(input_config.media_channels, input_config.channel_spend_columns, strict=False):
        if spend_col not in df.columns:
            raise ValueError(f"Spend column '{spend_col}' for channel '{channel}' not found in DataFrame")

        # Check if the channel has any non-zero spend
        if df[spend_col].sum() == 0:
            raise ValueError(
                f"Media channel '{channel}' (column '{spend_col}') has zero spend across "
                f"all time periods and geos. This will likely cause Meridian to fail or produce "
                f"unreliable results. Consider removing this channel from the model or "
                f"ensuring it has sufficient spend variation."
            )


def _add_media_to_data_builder(
    df: pd.DataFrame,
    builder: data_builder.DataFrameInputDataBuilder,
    input_data_builder_schema: MeridianInputDataBuilderSchema,
    date_column: str,
) -> data_builder.DataFrameInputDataBuilder:
    """Add paid media metrics to data frame input builder object.

    This function configures media channels in the Meridian data builder based on the available
    media metrics. It supports three different media configurations:

    1. **Reach/Frequency**: When `channel_reach_columns` and `channel_frequency_columns` are provided,
       uses the `with_reach()` method to configure reach and frequency metrics.
    2. **Impressions**: When `channel_impressions_columns` are provided (but no reach/frequency),
       uses the `with_media()` method with impressions as the media metric.
    3. **Spend-only**: When only `channel_spend_columns` are provided, uses the `with_media()` method
       with spend as both the media metric and spend metric.

    The function automatically determines which configuration to use based on the schema:
    - If `channel_reach_columns` is provided, it uses reach/frequency configuration
    - Otherwise, it uses impressions if available, or falls back to spend-only

    Args:
        df: Input DataFrame containing the media data with columns for spend, impressions,
            reach, and/or frequency as specified in the schema.
        builder: DataFrameInputDataBuilder instance to configure with media channels.
        input_data_builder_schema: Schema object containing media channel configuration including
            column names for spend, impressions, reach, frequency, and channel names.
        date_column: Name of the date column in the DataFrame.

    Returns:
        The configured DataFrameInputDataBuilder instance with media channels added.
        This follows the fluent interface pattern, allowing method chaining.

    Note:
        - When using reach/frequency, both `channel_reach_columns` and `channel_frequency_columns`
          must be provided together.
        - When using impressions, `channel_impressions_columns` should be provided.
        - When using spend-only, `media_cols` will be set equal to `media_spend_cols`.
        - The `channel_spend_columns` are always required as they represent the actual
          media spend amounts regardless of the media metric type used.

    """
    if input_data_builder_schema.channel_reach_columns:
        builder = builder.with_reach(
            df,
            reach_cols=input_data_builder_schema.channel_reach_columns,
            frequency_cols=input_data_builder_schema.channel_frequency_columns,
            rf_spend_cols=input_data_builder_schema.channel_spend_columns,
            rf_channels=input_data_builder_schema.media_channels,
            time_col=date_column,
        )
    else:
        media_cols = (
            input_data_builder_schema.channel_impressions_columns or input_data_builder_schema.channel_spend_columns
        )
        builder = builder.with_media(
            df,
            media_cols=media_cols,
            media_spend_cols=input_data_builder_schema.channel_spend_columns,
            media_channels=input_data_builder_schema.media_channels,
            time_col=date_column,
        )
    return builder


def construct_meridian_data_object(df: pd.DataFrame, config: MeridianConfig) -> pd.DataFrame:
    """Construct a Meridian data object from a pandas DataFrame.

    This function transforms a standard DataFrame into the format required by the Meridian
    framework. It handles the conversion of revenue to revenue_per_kpi, and configures
    the data builder with various components including KPI, population, controls,
    media channels (with different types: reach/frequency, impressions, or spend-only),
    organic media, and non-media treatments.

    Args:
        df: Input DataFrame containing the raw data with columns for dates, response,
            revenue, media spend, and other variables as specified in the config.
        config: MeridianConfig object containing the configuration for data transformation
            including column mappings and feature specifications.

    Returns:
        A Meridian data object built from the input DataFrame according to the config.

    Note:
        The function modifies the input DataFrame by:
        - Converting revenue to revenue_per_kpi (revenue / response)
        - Dropping the original revenue column
        - Adding various media and control components based on config
        - Validating that media channels have sufficient variation (non-zero spend)

    Raises:
        ValueError: If a media channel has zero spend across all time periods and geos,
                   as this would cause Meridian to fail or produce unreliable results.

    """
    df = df.copy()

    # Validate media channels have sufficient variation
    _validate_media_channels(df, config)

    # convert from "revenue" to "revenue_per_kpi"
    df[REVENUE_PER_KPI_COL] = (
        df[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL] / df[InputDataframeConstants.RESPONSE_COL]
    )
    df = df.drop(columns=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL)

    input_data_builder_schema = config.input_data_builder_config

    # KPI, population, and control variables
    builder = (
        data_builder.DataFrameInputDataBuilder(kpi_type="non_revenue")
        .with_kpi(df, time_col=config.date_column, kpi_col=InputDataframeConstants.RESPONSE_COL)
        .with_revenue_per_kpi(df, time_col=config.date_column, revenue_per_kpi_col=REVENUE_PER_KPI_COL)
    )
    # population, if provided, needs to be called "population" in the DF
    if "population" in df.columns:
        builder = builder.with_population(df)

    # controls (non-intervenable, e.g. macroeconomics)
    if input_data_builder_schema.control_columns:
        builder = builder.with_controls(
            df, time_col=config.date_column, control_cols=input_data_builder_schema.control_columns
        )

    # paid media
    builder = _add_media_to_data_builder(df, builder, input_data_builder_schema, config.date_column)

    # organic media
    if input_data_builder_schema.organic_media_columns:
        builder = builder.with_organic_media(
            df,
            organic_media_cols=input_data_builder_schema.organic_media_columns,
            organic_media_channels=input_data_builder_schema.organic_media_channels,
            media_time_col=config.date_column,
        )

    # non-media treatments (anything that is "intervenable", e.g. pricing/promotions)
    if input_data_builder_schema.non_media_treatment_columns:
        builder = builder.with_non_media_treatments(
            df,
            non_media_treatment_cols=input_data_builder_schema.non_media_treatment_columns,
            time_col=config.date_column,
        )

    return builder.build()


def construct_holdout_mask(max_train_date: pd.Timestamp, time_index: np.ndarray) -> np.ndarray:
    """Construct a boolean mask for holdout period identification.

    This function creates a boolean mask that identifies which time periods fall into
    the holdout/test period (after the maximum training date). The mask can be used
    to separate training and test data or to filter predictions to only the holdout period.

    Args:
        max_train_date: The maximum date to be considered part of the training period.
            All dates after this will be marked as holdout/test data.
        time_index: Array-like object containing the time index to be masked.
            Can be any format that pandas.to_datetime can convert.

    Returns:
        A boolean array of the same length as time_index, where True indicates
        the time period is in the holdout/test set (after max_train_date).

    Example:
        >>> time_index = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        >>> max_train_date = pd.Timestamp('2023-06-30')
        >>> mask = construct_holdout_mask(max_train_date, time_index)
        >>> # mask will be True for dates from 2023-07-01 onwards

    """
    full_index = pd.to_datetime(time_index)
    test_index = full_index[full_index > max_train_date]

    return full_index.isin(test_index)


class MeridianAdapter(BaseAdapter):
    """Adapter for Google Meridian MMM framework."""

    def __init__(self, config: MeridianConfig):
        """Initialize the Meridian adapter.

        Args:
            config: MeridianConfig object

        """
        self.config = config
        self.input_data_builder_schema = config.input_data_builder_config

        self.date_column = config.date_column
        self.channel_spend_columns = self.input_data_builder_schema.channel_spend_columns

        # Initialize stateful attributes to None/False
        self._reset_state()

    @property
    def media_channels(self) -> list[str]:
        """Return the channel names used by this adapter.

        For Meridian, this returns the human-readable channel names from the config.

        Returns
            List of channel names

        """
        return self.input_data_builder_schema.media_channels

    @property
    def primary_media_regressor_type(self) -> PrimaryMediaRegressor:
        """Return the type of primary media regressors used by the model.

        For Meridian, this is determined by the configuration:
        - If channel_reach_columns is provided: returns PrimaryMediaRegressor.REACH_AND_FREQUENCY
        - If channel_impressions_columns is provided: returns PrimaryMediaRegressor.IMPRESSIONS
        - Otherwise: returns PrimaryMediaRegressor.SPEND

        Returns
            PrimaryMediaRegressor enum value

        """
        if self.input_data_builder_schema.channel_reach_columns:
            return PrimaryMediaRegressor.REACH_AND_FREQUENCY
        if self.input_data_builder_schema.channel_impressions_columns:
            return PrimaryMediaRegressor.IMPRESSIONS
        return PrimaryMediaRegressor.SPEND

    @property
    def primary_media_regressor_columns(self) -> list[str]:
        """Return the primary media regressor columns that should be perturbed in tests.

        For Meridian, this depends on the configuration:
        - If channel_reach_columns is provided: returns empty list (not supported in perturbation tests)
        - If channel_impressions_columns is provided: returns channel_impressions_columns
        - Otherwise: returns channel_spend_columns

        Returns
            List of column names that are used as primary media regressors in the model

        """
        if self.input_data_builder_schema.channel_reach_columns:
            return []  # Not supported in perturbation tests
        if self.input_data_builder_schema.channel_impressions_columns:
            return self.input_data_builder_schema.channel_impressions_columns
        return self.channel_spend_columns

    def get_channel_names(self) -> list[str]:
        """Get the channel names that would be used as the index in get_channel_roi results.

        For Meridian, this returns the media_channels which are the human-readable
        channel names used in the ROI results.

        Returns
            List of channel names

        """
        return self.media_channels

    def copy(self) -> "MeridianAdapter":
        """Create a deep copy of this adapter with all configuration.

        Returns
            A new MeridianAdapter instance with the same configuration

        """
        # Create a deep copy of the input data builder schema
        new_input_data_builder_schema = MeridianInputDataBuilderSchema(
            media_channels=self.input_data_builder_schema.media_channels.copy(),
            channel_spend_columns=self.input_data_builder_schema.channel_spend_columns.copy(),
            channel_impressions_columns=(
                self.input_data_builder_schema.channel_impressions_columns.copy()
                if self.input_data_builder_schema.channel_impressions_columns
                else None
            ),
            channel_reach_columns=(
                self.input_data_builder_schema.channel_reach_columns.copy()
                if self.input_data_builder_schema.channel_reach_columns
                else None
            ),
            channel_frequency_columns=(
                self.input_data_builder_schema.channel_frequency_columns.copy()
                if self.input_data_builder_schema.channel_frequency_columns
                else None
            ),
            control_columns=(
                self.input_data_builder_schema.control_columns.copy()
                if self.input_data_builder_schema.control_columns
                else None
            ),
            organic_media_columns=(
                self.input_data_builder_schema.organic_media_columns.copy()
                if self.input_data_builder_schema.organic_media_columns
                else None
            ),
            organic_media_channels=(
                self.input_data_builder_schema.organic_media_channels.copy()
                if self.input_data_builder_schema.organic_media_channels
                else None
            ),
            non_media_treatment_columns=(
                self.input_data_builder_schema.non_media_treatment_columns.copy()
                if self.input_data_builder_schema.non_media_treatment_columns
                else None
            ),
        )

        # Create a new config
        new_config = MeridianConfig(
            date_column=self.date_column,
            input_data_builder_config=new_input_data_builder_schema,
            model_spec_config=self.config.model_spec_config,
            sample_posterior_config=self.config.sample_posterior_config,
        )

        return MeridianAdapter(new_config)

    def add_channels(self, new_channel_names: list[str]) -> dict[str, list[str]]:
        """Add new channels to the adapter's configuration.

        Args:
            new_channel_names: List of new channel names to add (e.g., ["TV", "Radio"])

        Returns:
            Dictionary mapping channel names to lists of column names that were added for each channel.
            For Meridian, this includes spend columns and potentially impressions columns.

        """
        if self.is_fitted:
            raise RuntimeError("Cannot add channels to a fitted adapter")

        # Check if reach/frequency regressor type is not supported
        if self.primary_media_regressor_type == PrimaryMediaRegressor.REACH_AND_FREQUENCY:
            raise NotImplementedError("Adding channels is not supported for reach and frequency regressor type")

        # Store original column lists to determine what was added
        original_spend_columns = self.input_data_builder_schema.channel_spend_columns.copy()
        original_impressions_columns = self.input_data_builder_schema.channel_impressions_columns.copy() if self.input_data_builder_schema.channel_impressions_columns else []

        # Add to the input data builder schema
        self.input_data_builder_schema.media_channels.extend(new_channel_names)

        # Add spend columns for new channels
        spend_columns = [f"{channel.lower()}_spend" for channel in new_channel_names]
        self.input_data_builder_schema.channel_spend_columns.extend(spend_columns)
        self.channel_spend_columns.extend(spend_columns)

        # Add impressions columns if using impressions regressors
        if self.primary_media_regressor_type == PrimaryMediaRegressor.IMPRESSIONS:
            impressions_columns = [f"{channel.lower()}_impressions" for channel in new_channel_names]
            if self.input_data_builder_schema.channel_impressions_columns:
                self.input_data_builder_schema.channel_impressions_columns.extend(impressions_columns)

        # Determine which columns were actually added for each channel
        added_columns = {}
        for channel_name in new_channel_names:
            channel_columns = []
            
            # Add spend column
            spend_col = f"{channel_name.lower()}_spend"
            if spend_col in self.input_data_builder_schema.channel_spend_columns and spend_col not in original_spend_columns:
                channel_columns.append(spend_col)
            
            # Add impressions column if applicable
            if self.primary_media_regressor_type == PrimaryMediaRegressor.IMPRESSIONS:
                impressions_col = f"{channel_name.lower()}_impressions"
                if (self.input_data_builder_schema.channel_impressions_columns and 
                    impressions_col in self.input_data_builder_schema.channel_impressions_columns and 
                    impressions_col not in original_impressions_columns):
                    channel_columns.append(impressions_col)
            
            added_columns[channel_name] = channel_columns

        return added_columns

    def get_primary_media_regressor_columns_for_channels(self, channel_names: list[str]) -> list[str]:
        """Get the primary media regressor columns for specific channels.

        This method returns the subset of primary_media_regressor_columns that correspond
        to the requested channels.

        Args:
            channel_names: List of channel names to get regressor columns for

        Returns:
            List of column names that are used as primary media regressors for the given channels

        """
        # Get the current primary media regressor columns
        all_regressor_columns = self.primary_media_regressor_columns
        
        # Find which columns correspond to the requested channels
        # This assumes the order of channels matches the order of columns
        channel_to_column_mapping = {}
        for i, channel in enumerate(self.media_channels):
            if i < len(all_regressor_columns):
                channel_to_column_mapping[channel] = all_regressor_columns[i]
        
        # Return the columns for the requested channels
        result = []
        for channel_name in channel_names:
            if channel_name in channel_to_column_mapping:
                result.append(channel_to_column_mapping[channel_name])
        
        return result

    def _reset_state(self) -> None:
        """Reset all stateful attributes to their initial values.

        This method ensures that the adapter starts with a clean state each time
        fit() is called, which is critical for the validation suite to work correctly.
        Only attributes that are set during fit() should be reset here.
        """
        # Explicitly delete trace to free TensorFlow memory
        if hasattr(self, "trace") and self.trace is not None:
            del self.trace
            self.trace = None

        # Clear TensorFlow session and memory
        tf.keras.backend.clear_session()

        # Force garbage collection
        gc.collect()

        # Reset other attributes
        self.training_data = None
        self.max_train_date = None
        self.holdout_mask = None
        self.model = None
        self.analyzer = None
        self.is_fitted = False

    def fit(self, data: pd.DataFrame, max_train_date: pd.Timestamp | None = None) -> None:
        """Fit the Meridian model to data.

        Args:
            data: Training data
            max_train_date: Optional maximum training date for holdout validation

        """
        # Reset state to ensure clean start when new data is provided
        self._reset_state()

        # build Meridian data object
        self.training_data = construct_meridian_data_object(data, self.config)
        self.max_train_date = max_train_date

        model_spec_kwargs = dict(self.config.model_spec_config)

        # if max train date is provided, construct a mask that is True for all dates before max_train_date
        if self.max_train_date:
            self.holdout_mask = construct_holdout_mask(self.max_train_date, self.training_data.kpi.time)
            # model expects a 2D array of shape (n_geos, n_times) so have to duplicate the values across each geo
            holdout_id = np.repeat(self.holdout_mask[None, :], repeats=len(self.training_data.kpi.geo), axis=0)
            # if only a single geo, convert to 1D array
            if holdout_id.shape[0] == 1:
                holdout_id = holdout_id[0, :]
            model_spec_kwargs["holdout_id"] = holdout_id

        # Create and fit the Meridian model
        model_spec = ModelSpec(**model_spec_kwargs)
        self.model = Meridian(
            input_data=self.training_data,  # type: ignore
            model_spec=model_spec,
        )
        self.trace = self.model.sample_posterior(**dict(self.config.sample_posterior_config))

        # used to compute channel contributions for ROI calculations
        self.analyzer = Analyzer(self.model)
        self.is_fitted = True

    def _predict_on_all_data(self) -> np.ndarray:
        """Make predictions on all data provided to fit().

        Returns
            predicted values on data provided to fit().

        """
        if not self.is_fitted or self.analyzer is None:
            raise RuntimeError("Model must be fit before prediction")

        # shape (n_chains, n_draws, n_times)
        preds_tensor = self.analyzer.expected_outcome(aggregate_geos=True, aggregate_times=False, use_kpi=True)
        posterior_mean = np.mean(preds_tensor, axis=(0, 1))
        return posterior_mean

    def predict(self, data: pd.DataFrame | None = None) -> np.ndarray:
        """Make predictions using the fitted model.

        This returns predictions for the entirety of the dataset passed to fit() unless
        `max_train_date` is specified when calling fit(); in that case it only returns
        predictions for the time periods indicated by the holdout mask.

        Note: Meridian doesn't require input data for prediction - it uses the fitted
        model state, so the `data` argument will be ignored if passed.

        Args:
            data: Ignored - Meridian uses the fitted model state for predictions.

        Returns:
            Predicted values

        Raises:
            RuntimeError: If model is not fitted

        """
        posterior_mean = self._predict_on_all_data()

        # if holdout mask is provided, use it to mask the predictions to restrict only to the
        # holdout period
        if self.holdout_mask is not None:
            posterior_mean = posterior_mean[self.holdout_mask]

        return posterior_mean

    def fit_and_predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Fit the Meridian model and make predictions given new input data.

        The full dataset must be passed to `fit()`, since making out-of-sample predictions
        is only possible by way of specifying a holdout mask when sampling from the
        posterior.

        Args:
            train: Training data
            test: Test data

        Returns:
            Predicted values for the test period

        """
        train_and_test = pd.concat([train, test])
        max_train_date = train[self.date_column].squeeze().max()
        self.fit(train_and_test, max_train_date=max_train_date)
        return self.predict()

    def fit_and_predict_in_sample(self, data: pd.DataFrame) -> np.ndarray:
        """Fit the model on data and return predictions for the same data.

        Args:
            data: dataset to train model on and make predictions for

        Returns:
            Predicted values for the training data.

        """
        # no max train date specified, so predictions are all in-sample
        self.fit(data)
        return self._predict_on_all_data()

    def get_channel_roi(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        """Get channel ROI estimates.

        Args:
            start_date: Optional start date for ROI calculation
            end_date: Optional end date for ROI calculation

        Returns:
            Series containing ROI estimates for each channel

        """
        if not self.is_fitted or self.training_data is None or self.analyzer is None:
            raise RuntimeError("Model must be fit before computing ROI")

        # restrict ROI calculation to a particular period if start/end date args are
        # passed
        training_date_index = pd.to_datetime(self.training_data.kpi.time)
        roi_date_index = training_date_index.copy()
        if start_date:
            roi_date_index = roi_date_index[roi_date_index >= start_date]
        if end_date:
            roi_date_index = roi_date_index[roi_date_index < end_date]

        selected_times = [bool(date) for date in training_date_index.isin(roi_date_index)]

        # analyzer.roi() returns a tensor of shape (n_chains, n_draws, n_channels)
        rois_per_channel = np.mean(self.analyzer.roi(selected_times=selected_times), axis=(0, 1))

        rois = {}
        for channel, roi in zip(self.media_channels, rois_per_channel, strict=False):
            rois[channel] = float(roi)
        return pd.Series(rois)
