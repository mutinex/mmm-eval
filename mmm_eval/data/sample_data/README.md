# Sample Data Files

This directory contains sample data files used for testing.

## geo_media.xlsx

This file contains sample data for Meridian framework testing.

### How to obtain the file:

1. Download the file from the Meridian GitHub repository:
   ```
   wget https://github.com/google/meridian/raw/main/meridian/data/simulated_data/xlsx/geo_media.xlsx
   ```

2. Place the downloaded file in this directory (`mmm_eval/data/sample_data/geo_media.xlsx`)

### File description:
- Source: Meridian GitHub repository
- Format: Excel (.xlsx)
- Contains: Simulated media mix modeling data with geographic and media channel information
- Used by: `generate_meridian_data()` function in `synth_data_generator.py` 