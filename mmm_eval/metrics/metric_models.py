

from enum import Enum
from typing import List

from pydantic import BaseModel

from mmm_eval.metrics.threshold_constants import AccuracyThresholdConstants, CrossValidationThresholdConstants


class MetricNamesBase(Enum):
    """Base class for metric name enums"""
    
    @classmethod
    def metrics_to_list(cls) -> List[str]:
        """
        Convert the enum to a list of strings.
        """
        return [member.value for member in cls]

class AccuracyMetricNames(MetricNamesBase):
    """Define the names of the accuracy metrics"""
    MAPE = "mape"
    R_SQUARED = "r_squared"

class CrossValidationMetricNames(MetricNamesBase):
    """Define the names of the cross-validation metrics"""
    MEAN_MAPE = "mean_mape"
    STD_MAPE = "std_mape"
    MEAN_R_SQUARED = "mean_r_squared"
    STD_R_SQUARED = "std_r_squared"

class StabilityMetricNames(MetricNamesBase):
    """Define the names of the stability metrics"""
    pass

class MetricResults(BaseModel):
    """Define the results of the metrics"""
    
    def check_test_passed(self) -> bool:
        """
        Check if the tests passed.
        """
        
        raise NotImplementedError("Child classes must implement test_passed()")


class AccuracyMetricResults(MetricResults):

    """Define the results of the accuracy metrics"""

    mape: float
    r_squared: float

    def check_test_passed(self) -> bool:
        """
        Check if the tests passed.
        """
        return (
            self.mape <= AccuracyThresholdConstants.MAPE
            and self.r_squared > AccuracyThresholdConstants.R_SQUARED
        )
    
class CrossValidationMetricResults(MetricResults):

    """Define the results of the cross-validation metrics"""

    mean_mape: float
    std_mape: float
    mean_r_squared: float
    std_r_squared: float

    def check_test_passed(self) -> bool:
        """
        Check if the tests passed.
        """
        return (
            self.mean_mape <= CrossValidationThresholdConstants.MEAN_MAPE
            and self.std_mape <= CrossValidationThresholdConstants.STD_MAPE
            and self.mean_r_squared >= CrossValidationThresholdConstants.MEAN_R_SQUARED
        )