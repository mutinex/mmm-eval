"""Unit tests for Meridian adapter."""

import pandas as pd
import pytest
from unittest.mock import Mock, patch, ANY
import numpy as np

from mmm_eval.adapters.meridian import construct_meridian_data_object, REVENUE_PER_KPI_COL, MeridianAdapter
from mmm_eval.configs import MeridianConfig
from mmm_eval.adapters.schemas import (
    MeridianInputDataBuilderSchema,
    MeridianModelSpecSchema,
    MeridianSamplePosteriorSchema,
)
from mmm_eval.data.constants import InputDataframeConstants
from meridian.model.prior_distribution import PriorDistribution


class TestConstructMeridianDataObject:
    """Test cases for construct_meridian_data_object function."""

    def setup_method(self):
        """Set up test data and configurations."""
        # Create base test data
        self.base_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            InputDataframeConstants.RESPONSE_COL: [100.0] * 10,
            InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 10,
            "tv_spend": [500.0] * 10,
            "digital_spend": [300.0] * 10,
        })

        # Create mock prior distribution
        self.mock_prior = Mock(spec=PriorDistribution)

        # Create base model spec config
        self.base_model_spec_config = MeridianModelSpecSchema(
            prior=self.mock_prior,
            media_effects_dist="log_normal",
            hill_before_adstock=False,
            max_lag=8,
        )

        # Create base sample posterior config
        self.base_sample_posterior_config = MeridianSamplePosteriorSchema(
            n_chains=1,
            n_adapt=10,
            n_burnin=10,
            n_keep=25,
            seed=42,
        )

    def _create_config(self, input_data_builder_config: MeridianInputDataBuilderSchema) -> MeridianConfig:
        """Helper method to create a MeridianConfig."""
        return MeridianConfig(
            input_data_builder_config=input_data_builder_config,
            model_spec_config=self.base_model_spec_config,
            sample_posterior_config=self.base_sample_posterior_config,
            revenue_column=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
        )

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_basic_functionality(self, mock_builder_class):
        """Test basic functionality with required fields only."""
        # Setup
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv", "digital"],
            channel_spend_columns=["tv_spend", "digital_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(self.base_df, config)

        # Verify
        assert result is not None
        mock_builder_class.assert_called_once_with(kpi_type="non_revenue")
        mock_builder.with_kpi.assert_called_once_with(
            ANY, 
            time_col="date", 
            kpi_col=InputDataframeConstants.RESPONSE_COL
        )
        mock_builder.with_revenue_per_kpi.assert_called_once_with(
            ANY, 
            time_col="date", 
            revenue_per_kpi_col=REVENUE_PER_KPI_COL
        )
        mock_builder.with_media.assert_called_once_with(
            ANY,
            media_cols=["tv_spend", "digital_spend"],
            media_spend_cols=["tv_spend", "digital_spend"],
            media_channels=["tv", "digital"],
            time_col="date",
        )
        mock_builder.build.assert_called_once()

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_revenue_per_kpi_calculation(self, mock_builder_class):
        """Test that revenue_per_kpi is calculated correctly."""
        # Setup
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Create a fresh copy of the DataFrame to avoid test pollution
        test_df = self.base_df.copy()

        # Execute
        construct_meridian_data_object(test_df, config)

        # The DataFrame passed to with_kpi should have the correct columns
        called_df = mock_builder.with_kpi.call_args[0][0]
        expected_revenue_per_kpi = self.base_df[InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL] / self.base_df[InputDataframeConstants.RESPONSE_COL]
        assert REVENUE_PER_KPI_COL in called_df.columns
        pd.testing.assert_series_equal(
            called_df[REVENUE_PER_KPI_COL], 
            expected_revenue_per_kpi, 
            check_names=False
        )
        # The revenue column should be dropped in the DataFrame passed to with_kpi
        assert InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL not in called_df.columns

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_with_population_column(self, mock_builder_class):
        """Test handling when population column is present."""
        # Setup
        df_with_population = self.base_df.copy()
        df_with_population["population"] = [1000000] * 10

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_population.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(df_with_population, config)

        # Verify
        mock_builder.with_population.assert_called_once_with(ANY)

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_without_population_column(self, mock_builder_class):
        """Test handling when population column is not present."""
        # Setup
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(self.base_df, config)

        # Verify
        mock_builder.with_population.assert_not_called()

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_with_control_columns(self, mock_builder_class):
        """Test handling when control columns are provided."""
        # Setup
        df_with_controls = self.base_df.copy()
        df_with_controls["control_var1"] = [0.5] * 10
        df_with_controls["control_var2"] = [0.3] * 10

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_controls.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
            control_columns=["control_var1", "control_var2"],
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(df_with_controls, config)

        # Verify
        mock_builder.with_controls.assert_called_once_with(
            ANY,
            time_col="date",
            control_cols=["control_var1", "control_var2"],
        )

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_without_control_columns(self, mock_builder_class):
        """Test handling when control columns are not provided."""
        # Setup
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
            control_columns=None,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(self.base_df, config)

        # Verify
        mock_builder.with_controls.assert_not_called()

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_with_reach_frequency_media(self, mock_builder_class):
        """Test handling when reach/frequency columns are provided."""
        # Setup
        df_with_rf = self.base_df.copy()
        df_with_rf["tv_reach"] = [1000000] * 10
        df_with_rf["tv_frequency"] = [3.5] * 10

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_reach.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            channel_reach_columns=["tv_reach"],
            channel_frequency_columns=["tv_frequency"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(df_with_rf, config)

        # Verify
        mock_builder.with_reach.assert_called_once_with(
            ANY,
            reach_cols=["tv_reach"],
            frequency_cols=["tv_frequency"],
            rf_spend_cols=["tv_spend"],
            rf_channels=["tv"],
            time_col="date",
        )
        mock_builder.with_media.assert_not_called()

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_with_impressions_media(self, mock_builder_class):
        """Test handling when impressions columns are provided."""
        # Setup
        df_with_impressions = self.base_df.copy()
        df_with_impressions["tv_impressions"] = [5000000] * 10

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            channel_impressions_columns=["tv_impressions"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(df_with_impressions, config)

        # Verify
        mock_builder.with_media.assert_called_once_with(
            ANY,
            media_cols=["tv_impressions"],  # Should use impressions, not spend
            media_spend_cols=["tv_spend"],
            media_channels=["tv"],
            time_col="date",
        )

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_with_spend_only_media(self, mock_builder_class):
        """Test handling when only spend columns are provided (no impressions/reach)."""
        # Setup
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(self.base_df, config)

        # Verify
        mock_builder.with_media.assert_called_once_with(
            ANY,
            media_cols=["tv_spend"],  # Should use spend as media_cols when no impressions
            media_spend_cols=["tv_spend"],
            media_channels=["tv"],
            time_col="date",
        )

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_with_organic_media(self, mock_builder_class):
        """Test handling when organic media columns are provided."""
        # Setup
        df_with_organic = self.base_df.copy()
        df_with_organic["organic_impressions"] = [2000000] * 10

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.with_organic_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            organic_media_columns=["organic_impressions"],
            organic_media_channels=["organic"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(df_with_organic, config)

        # Verify
        mock_builder.with_organic_media.assert_called_once_with(
            ANY,
            organic_media_cols=["organic_impressions"],
            organic_media_channels=["organic"],
            media_time_col="date",
        )

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_without_organic_media(self, mock_builder_class):
        """Test handling when organic media columns are not provided."""
        # Setup
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            organic_media_columns=None,
            organic_media_channels=None,
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(self.base_df, config)

        # Verify
        mock_builder.with_organic_media.assert_not_called()

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_with_non_media_treatments(self, mock_builder_class):
        """Test handling when non-media treatment columns are provided."""
        # Setup
        df_with_treatments = self.base_df.copy()
        df_with_treatments["discount"] = [0.1] * 10
        df_with_treatments["promotion"] = [0.2] * 10

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.with_non_media_treatments.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            non_media_treatment_columns=["discount", "promotion"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(df_with_treatments, config)

        # Verify
        mock_builder.with_non_media_treatments.assert_called_once_with(
            ANY,
            non_media_treatment_cols=["discount", "promotion"],
            time_col="date",
        )

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_without_non_media_treatments(self, mock_builder_class):
        """Test handling when non-media treatment columns are not provided."""
        # Setup
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_media.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            non_media_treatment_columns=None,
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(self.base_df, config)

        # Verify
        mock_builder.with_non_media_treatments.assert_not_called()

    @patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder')
    def test_comprehensive_scenario(self, mock_builder_class):
        """Test a comprehensive scenario with all optional features enabled."""
        # Setup
        df_comprehensive = self.base_df.copy()
        df_comprehensive["population"] = [1000000] * 10
        df_comprehensive["control_var1"] = [0.5] * 10
        df_comprehensive["tv_reach"] = [1000000] * 10
        df_comprehensive["tv_frequency"] = [3.5] * 10
        df_comprehensive["organic_impressions"] = [2000000] * 10
        df_comprehensive["discount"] = [0.1] * 10

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.with_kpi.return_value = mock_builder
        mock_builder.with_revenue_per_kpi.return_value = mock_builder
        mock_builder.with_population.return_value = mock_builder
        mock_builder.with_controls.return_value = mock_builder
        mock_builder.with_reach.return_value = mock_builder
        mock_builder.with_organic_media.return_value = mock_builder
        mock_builder.with_non_media_treatments.return_value = mock_builder
        mock_builder.build.return_value = Mock()

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            channel_reach_columns=["tv_reach"],
            channel_frequency_columns=["tv_frequency"],
            control_columns=["control_var1"],
            organic_media_columns=["organic_impressions"],
            organic_media_channels=["organic"],
            non_media_treatment_columns=["discount"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute
        result = construct_meridian_data_object(df_comprehensive, config)

        # Verify all methods were called
        mock_builder.with_population.assert_called_once_with(ANY)
        mock_builder.with_controls.assert_called_once_with(
            ANY,
            time_col="date",
            control_cols=["control_var1"],
        )
        mock_builder.with_reach.assert_called_once_with(
            ANY,
            reach_cols=["tv_reach"],
            frequency_cols=["tv_frequency"],
            rf_spend_cols=["tv_spend"],
            rf_channels=["tv"],
            time_col="date",
        )
        mock_builder.with_organic_media.assert_called_once_with(
            ANY,
            organic_media_cols=["organic_impressions"],
            organic_media_channels=["organic"],
            media_time_col="date",
        )
        mock_builder.with_non_media_treatments.assert_called_once_with(
            ANY,
            non_media_treatment_cols=["discount"],
            time_col="date",
        )
        mock_builder.build.assert_called_once()

    def test_zero_division_in_revenue_per_kpi(self):
        """Test handling of zero response values in revenue_per_kpi calculation."""
        # Setup
        df_with_zero_response = self.base_df.copy()
        df_with_zero_response.loc[0, InputDataframeConstants.RESPONSE_COL] = 0.0

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        with patch('mmm_eval.adapters.meridian.data_builder.DataFrameInputDataBuilder') as mock_builder_class:
            mock_builder = Mock()
            mock_builder_class.return_value = mock_builder
            mock_builder.with_kpi.return_value = mock_builder
            mock_builder.with_revenue_per_kpi.return_value = mock_builder
            mock_builder.with_media.return_value = mock_builder
            mock_builder.build.return_value = Mock()

            # Execute
            construct_meridian_data_object(df_with_zero_response, config)

            # The DataFrame passed to with_kpi should have the correct columns
            called_df = mock_builder.with_kpi.call_args[0][0]
            # Check for inf or NaN in the revenue_per_kpi column
            assert called_df[REVENUE_PER_KPI_COL].isnull().any() or (called_df[REVENUE_PER_KPI_COL] == float('inf')).any()

    def test_missing_required_columns(self):
        """Test handling when required columns are missing."""
        # Setup
        df_missing_columns = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            # Missing response and revenue columns
        })

        input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        config = self._create_config(input_config)

        # Execute and verify it raises an error
        with pytest.raises(ValueError):
            construct_meridian_data_object(df_missing_columns, config)


class TestMeridianAdapter:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            InputDataframeConstants.RESPONSE_COL: [100.0] * 5,
            InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL: [1000.0] * 5,
            "tv_spend": [500.0] * 5,
        })
        self.mock_prior = Mock(spec=PriorDistribution)
        self.input_config = MeridianInputDataBuilderSchema(
            date_column="date",
            media_channels=["tv"],
            channel_spend_columns=["tv_spend"],
            response_column=InputDataframeConstants.RESPONSE_COL,
        )
        self.model_spec = MeridianModelSpecSchema(
            prior=self.mock_prior,
            media_effects_dist="log_normal",
            hill_before_adstock=False,
            max_lag=8,
        )
        self.posterior_config = MeridianSamplePosteriorSchema(
            n_chains=1,
            n_adapt=2,
            n_burnin=2,
            n_keep=2,
            seed=42,
        )
        self.config = MeridianConfig(
            input_data_builder_config=self.input_config,
            model_spec_config=self.model_spec,
            sample_posterior_config=self.posterior_config,
            revenue_column=InputDataframeConstants.MEDIA_CHANNEL_REVENUE_COL,
        )

    @patch("mmm_eval.adapters.meridian.construct_meridian_data_object")
    @patch("mmm_eval.adapters.meridian.ModelSpec")
    @patch("mmm_eval.adapters.meridian.Meridian")
    @patch("mmm_eval.adapters.meridian.Analyzer")
    def test_fit_sets_state_and_calls_dependencies(self, mock_analyzer, mock_meridian, mock_modelspec, mock_construct):
        mock_training_data = Mock()
        mock_construct.return_value = mock_training_data
        mock_model = Mock()
        mock_meridian.return_value = mock_model
        mock_trace = Mock()
        mock_model.sample_posterior.return_value = mock_trace
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance

        adapter = MeridianAdapter(self.config)
        adapter.fit(self.df)

        mock_construct.assert_called_once_with(self.df, self.config)
        mock_modelspec.assert_called_once()
        mock_meridian.assert_called_once_with(input_data=mock_training_data, model_spec=ANY)
        mock_model.sample_posterior.assert_called_once_with(**dict(self.posterior_config))
        mock_analyzer.assert_called_once_with(mock_model)
        assert adapter.training_data == mock_training_data
        assert adapter.model == mock_model
        assert adapter.trace == mock_trace
        assert adapter.analyzer == mock_analyzer_instance
        assert adapter.is_fitted is True

    @patch("mmm_eval.adapters.meridian.Analyzer")
    def test_predict_returns_posterior_mean_and_applies_holdout_mask(self, mock_analyzer):
        adapter = MeridianAdapter(self.config)
        adapter.is_fitted = True
        adapter.analyzer = mock_analyzer_instance = Mock()
        # Simulate expected_outcome returns (chains, draws, times)
        preds_tensor = np.ones((2, 2, 5)) * 10
        mock_analyzer_instance.expected_outcome.return_value = preds_tensor
        # No holdout mask
        assert np.allclose(adapter.predict(), np.mean(preds_tensor, axis=(0, 1)))
        # With holdout mask
        adapter.holdout_mask = np.array([False, True, True, False, True])
        masked = np.mean(preds_tensor, axis=(0, 1))[adapter.holdout_mask]
        assert np.allclose(adapter.predict(), masked)

    def test_predict_raises_if_not_fitted(self):
        adapter = MeridianAdapter(self.config)
        with pytest.raises(RuntimeError):
            adapter.predict()

    @patch("mmm_eval.adapters.meridian.MeridianAdapter.fit")
    @patch("mmm_eval.adapters.meridian.MeridianAdapter.predict")
    def test_fit_and_predict_calls_fit_and_predict(self, mock_predict, mock_fit):
        adapter = MeridianAdapter(self.config)
        train = self.df.iloc[:3]
        test = self.df.iloc[3:]
        mock_predict.return_value = np.array([1, 2])
        result = adapter.fit_and_predict(train, test)
        mock_fit.assert_called_once()
        mock_predict.assert_called_once()
        assert np.all(result == np.array([1, 2]))

    @patch("mmm_eval.adapters.meridian.Analyzer")
    def test_get_channel_roi_returns_series(self, mock_analyzer):
        adapter = MeridianAdapter(self.config)
        adapter.is_fitted = True
        adapter.analyzer = mock_analyzer_instance = Mock()
        # Simulate roi returns (chains, draws, channels)
        roi_tensor = np.ones((2, 2, 1)) * 5.0
        mock_analyzer_instance.roi.return_value = roi_tensor
        adapter.media_channels = ["tv"]
        # Simulate training_data.kpi.time
        adapter.training_data = Mock()
        adapter.training_data.kpi.time = pd.date_range("2023-01-01", periods=5)
        # No date filtering
        result = adapter.get_channel_roi()
        assert isinstance(result, pd.Series)
        assert result["tv"] == 5.0
        # With date filtering
        result = adapter.get_channel_roi(start_date=pd.Timestamp("2023-01-03"))
        assert isinstance(result, pd.Series)

    def test_get_channel_roi_raises_if_not_fitted(self):
        adapter = MeridianAdapter(self.config)
        with pytest.raises(RuntimeError):
            adapter.get_channel_roi()

    def test_fit_resets_state(self):
        # Fit once, then change state, then fit again and check reset
        with patch("mmm_eval.adapters.meridian.construct_meridian_data_object", return_value=Mock()), \
             patch("mmm_eval.adapters.meridian.ModelSpec"), \
             patch("mmm_eval.adapters.meridian.Meridian") as mock_meridian, \
             patch("mmm_eval.adapters.meridian.Analyzer"):
            adapter = MeridianAdapter(self.config)
            adapter.fit(self.df)
            adapter.model = "not a model"
            adapter.is_fitted = False
            adapter.fit(self.df)
            assert adapter.model != "not a model"
            assert adapter.is_fitted is True 