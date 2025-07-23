from enum import StrEnum

PLOT_WIDTH = 500
DATE_COL = "date"
FREQ_COL = "date_frequency"
FREQ_DEFAULT = "W-MON"
BCP_COLS = ["brand", "category", "product"]
EXCEPT_COLUMNS = [
    "data_source_slug",
    "etl_file_name",
    "etl_load_datetime_utc",
    "file_id",
    "file_loaded_datetime_utc",
    "record_id",
    "time_period_id",
]
EXCEPT_COLUMN_MAP = {
    "clv_snapshot": [],
    "brand_snapshot": ["legacy_convoy_metrics"],
    "category_snapshot": ["currency_code_original", "legacy_convoy_metrics", "metric_value_original"],
    "competitor_spend_snapshot": ["legacy_convoy_metrics", "currency_code_original", "spend_original"],
    "discounts_snapshot": [
        "currency_code_original",
        "discounts_value_original",
        "discounted_retail_price_original",
        "legacy_convoy_metrics",
        "rebates_value_original",
    ],
    "earned_media_snapshot": ["legacy_convoy_metrics"],
    "events_snapshot": ["legacy_convoy_metrics"],
    "offers_snapshot": [
        "currency_code_original",
        "legacy_convoy_metrics",
        "offer_value_original",
    ],
    "owned_media_snapshot": ["legacy_convoy_metrics", "purpose"],
    "paid_media_snapshot": [
        "convoy_brand",
        "convoy_category",
        "convoy_product",
        "currency_code_original",
        "fees_original",
        "spend_original",
    ],
    "pricing_snapshot": ["legacy_convoy_metrics", "currency_code_original", "retail_price_original"],
    "pricing_auto_snapshot": [],
    "pricing_beverages_snapshot": [],
    "pricing_energy_snapshot": [],
    "sales_snapshot": ["currency_code_original", "value_original"],
    "scan_data_snapshot": ["currency_code_original", "retail_value_original"],
    "sponsorship_snapshot": [
        "asset_cost_original",
        "currency_code_original",
        "end_date",
        "legacy_convoy_metrics",
    ],
    "trade_marketing_snapshot": [
        "currency_code_original",
        "end_date",
        "legacy_convoy_metrics",
        "metric_value_original",
    ],
    "key_value_items_snapshot": [],
    "onboarding_datamart_base": [],
    "pricing_banking_snapshot": [],
}

class DateFrequencyAlias(StrEnum):
    B = "B"  # Business Days
    D = "D"  # Daily
    M = "M"  # Monthly (deprecated in future pandas)
    ME = "ME"  # Monthly
    MS = "MS"  # Month start
    W_SUN = "W-SUN"  # Weekly on Sunday
    W_MON = "W-MON"  # Weekly on Monday
    W_TUE = "W-TUE"  # Weekly on Tuesday
    W_WED = "W-WED"  # Weekly on Wednesday
    W_THU = "W-THU"  # Weekly on Thursday
    W_FRI = "W-FRI"  # Weekly on Friday
    W_SAT = "W-SAT"  # Weekly on Saturday

    @classmethod
    def daily_frequencies(cls):
        """List of daily frequencies."""
        return [cls.B, cls.D]

    @classmethod
    def weekly_frequencies(cls):
        """List of weekly frequencies."""
        return [cls.W_SUN, cls.W_MON, cls.W_TUE, cls.W_WED, cls.W_THU, cls.W_FRI, cls.W_SAT]

    @classmethod
    def monthly_frequencies(cls):
        """List of monthly frequencies."""
        return [cls.M, cls.MS, cls.ME]

    def to_weekday(self):
        """Map weekly frequencies to their associated weekday numbers."""
        try:
            return {
                self.W_MON.value: 0,
                self.W_TUE.value: 1,
                self.W_WED.value: 2,
                self.W_THU.value: 3,
                self.W_FRI.value: 4,
                self.W_SAT.value: 5,
                self.W_SUN.value: 6,
            }[self.value]
        except KeyError as err:
            raise KeyError(f"{self.value} is not a weekly frequency.") from err

    @classmethod
    def days_between_weekdays(cls, freq_a: str, freq_b: str):
        """Calculate the shortest number of days offset from freq_b to freq_a.

        For any permutation of freq_a (freq_a = W-MON in this example):
            W-SUN -> W-MON +1
            W-MON -> W-MON 0
            W-TUE -> W-MON -1
            W-WED -> W-MON -2
            W-THU -> W-MON -3
            W-FRI -> W-MON +3
            W-SAT -> W-MON +2
        """
        diff = DateFrequencyAlias(freq_b).to_weekday() - DateFrequencyAlias(freq_a).to_weekday()
        if diff < -3:
            diff += 7
        return diff