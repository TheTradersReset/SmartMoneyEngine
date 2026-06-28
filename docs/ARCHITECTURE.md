# SmartMoneyEngine Architecture

## Core Layer

Responsible for:

- Loading CSV files
- Validation
- Cleaning
- Preparing Data

---

## Modules

### data_loader.py

Loads CSV into Pandas DataFrame.

### validators.py

Responsible for:

- Required Columns
- Data Types
- Missing Values
- Duplicate Rows
- OHLC Validation
- Date Validation

### cleaner.py

Responsible for:

- Remove duplicates
- Fill missing values
- Sorting
- Reset Index

### preprocessor.py

Responsible for final dataframe preparation before indicators.