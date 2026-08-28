# HAB Early-Warning System

Final-year CSE project for Harmful Algal Bloom early-warning prediction in Vembanad Lake.

## Ryder / Integration Track

Current implementation includes:

- Input schema validation
- Satellite/environmental date alignment
- No-future-image leakage check
- Provisional chlorophyll-a-based bloom labels
- 5-day-ahead target creation
- Chronological train/test split
- Random Forest baseline model
- Accuracy, precision, recall, F1 and confusion matrix

## Important

Current baseline metrics were generated using mock data only to verify that the software pipeline works.

They are not scientific project results.

Real project inputs will later be supplied as:

### Satellite
`data/sentinel2/processed/sentinel_index.csv`

Columns:

`date,image_path,cloud_percentage`

### Environmental
`data/environmental/processed/environmental_dataset.csv`

Columns:

`date,lat,lon,chlorophyll_a,sst,rainfall,wind_speed`
