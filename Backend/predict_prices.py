from pathlib import Path

import polars as pl


# --------------------------------------------------
# Setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
CSV_PATH = DATA_DIR / "pp-complete.csv"

OUTPUT_PATH = DATA_DIR / "house_price_predictions.csv"

columns = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "new_build",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status",
]


# --------------------------------------------------
# Read only the data we need
# --------------------------------------------------

df = (
    pl.scan_csv(
        CSV_PATH,
        has_header=False,
        new_columns=columns,
    )
    .select([
        "price",
        "date_of_transfer",
        "postcode",
        "paon",
        "saon",
        "street",
    ])
    .with_columns([
        pl.col("price")
        .cast(pl.Float64),

        pl.col("date_of_transfer")
        .str.to_datetime(strict=False),

        pl.col("postcode")
        .str.extract(
            r"^([A-Z]+\d+[A-Z]?)",
            1
        )
        .alias("postcode_district"),
    ])
)


# --------------------------------------------------
# Annual district prices
# --------------------------------------------------

annual = (
    df
    .with_columns(
        pl.col("date_of_transfer")
        .dt.year()
        .alias("year")
    )
    .group_by([
        "postcode_district",
        "year",
    ])
    .agg(
        pl.col("price")
        .mean()
        .alias("average_price")
    )
    .sort([
        "postcode_district",
        "year",
    ])
    .collect()
)


# --------------------------------------------------
# Annual growth
# --------------------------------------------------

annual = annual.with_columns(
    (
        pl.col("average_price")
        / pl.col("average_price")
        .shift(1)
        - 1
    )
    .over("postcode_district")
    .alias("growth")
)


# --------------------------------------------------
# Fit regression using Polars
#
# growth = intercept + slope * year
# --------------------------------------------------

models = (
    annual
    .filter(
        pl.col("growth").is_not_null()
    )
    .group_by("postcode_district")
    .agg([
        pl.len().alias("n"),

        pl.col("year")
        .mean()
        .alias("mean_year"),

        pl.col("growth")
        .mean()
        .alias("mean_growth"),
    ])
    .join(
        annual
        .filter(
            pl.col("growth").is_not_null()
        )
        .with_columns([
            (
                pl.col("year")
                - pl.col("year")
                .mean()
                .over("postcode_district")
            )
            .alias("x"),

            (
                pl.col("growth")
                - pl.col("growth")
                .mean()
                .over("postcode_district")
            )
            .alias("y"),
        ])
        .group_by("postcode_district")
        .agg([
            (
                pl.col("x")
                * pl.col("y")
            )
            .sum()
            .alias("xy"),

            (
                pl.col("x")
                * pl.col("x")
            )
            .sum()
            .alias("xx"),
        ]),
        on="postcode_district",
        how="left",
    )
    .with_columns(
        (
            pl.col("xy")
            / pl.col("xx")
        )
        .alias("slope")
    )
    .with_columns(
        (
            pl.col("mean_growth")
            - pl.col("slope")
            * pl.col("mean_year")
        )
        .alias("intercept")
    )
    .select([
        "postcode_district",
        "intercept",
        "slope",
    ])
)


print("District models:")
print(models)


# --------------------------------------------------
# Final year
# --------------------------------------------------

final_year = annual["year"].max()

print(f"Final year: {final_year}")


# --------------------------------------------------
# Get latest transaction for every property
#
# No global sort required.
# --------------------------------------------------

properties = (
    df
    .filter(
        pl.col("postcode").is_not_null()
    )
    .group_by([
        "postcode",
        "paon",
        "saon",
        "street",
    ])
    .agg([
        pl.col("price")
        .sort_by("date_of_transfer")
        .last()
        .alias("last_price"),

        pl.col("date_of_transfer")
        .max()
        .alias("last_sale_date"),

        pl.col("postcode_district")
        .first()
        .alias("postcode_district"),
    ])
    .with_columns(
        pl.col("last_sale_date")
        .dt.year()
        .alias("last_sale_year")
    )
    .collect()
)


# --------------------------------------------------
# Join district models
# --------------------------------------------------

predictions = (
    properties
    .join(
        models,
        on="postcode_district",
        how="left",
    )
)


# --------------------------------------------------
# Predicted growth at last sale year
# --------------------------------------------------

predictions = predictions.with_columns(
    (
        pl.col("intercept")
        + pl.col("slope")
        * pl.col("last_sale_year")
    )
    .alias("starting_growth")
)


# --------------------------------------------------
# Predicted price
# --------------------------------------------------

predictions = predictions.with_columns(
    pl.when(
        pl.col("last_sale_year") >= final_year
    )
    .then(
        pl.col("last_price")
    )
    .otherwise(
        pl.col("last_price")
        * (
            1 + pl.col("starting_growth")
        )
        ** (
            final_year
            - pl.col("last_sale_year")
        )
    )
    .round(0)
    .alias("predicted_price")
)


# --------------------------------------------------
# Output
# --------------------------------------------------

predictions = predictions.select([
    "postcode",
    "paon",
    "saon",
    "street",
    "postcode_district",
    "last_price",
    "last_sale_year",
    "predicted_price",
])


# Save as CSV
predictions.write_csv(OUTPUT_PATH)

print(f"Saved predictions to: {OUTPUT_PATH}")