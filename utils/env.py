from pyspark.sql import SparkSession


def ensure_database(catalog_name: str, schema_name: str) -> None:
    """
    Ensure the Unity Catalog schema exists inside an existing catalog and set it as context.

    The catalog itself must already exist and be configured with storage
    (via the UI or a separate SQL deployment step). This function will:
      - Fail fast with a clear error if the catalog is missing or inaccessible.
      - Create the schema if needed.
      - Set the current catalog and schema.
    """
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("No active SparkSession found")

    # Require catalog to exist; if not, surface a clear message instead of a low-level UC error.
    try:
        spark.sql(f"USE CATALOG {catalog_name}")
    except Exception as e:
        raise RuntimeError(
            f"Unity Catalog catalog '{catalog_name}' does not exist or is not accessible. "
            "Please create it in the Databricks UI or with "
            f"CREATE CATALOG {catalog_name} MANAGED LOCATION '<storage-path>'."
        ) from e

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
    spark.sql(f"USE SCHEMA {schema_name}")


def build_config(
    catalog_name: str,
    schema_name: str,
    yfinance_start: str,
    yfinance_end: str,
    model_name: str,
    model_date: str,
    mc_executors: int,
    mc_volatility_days: int,
    mc_runs: int,
    table_stocks: str,
    table_indicators: str,
    table_volatility: str,
    table_mc_market: str,
    table_mc_trials: str,
) -> dict:
    """
    Build a UC-aware config dictionary from notebook parameters.
    """
    base = f"{catalog_name}.{schema_name}"
    return {
        "yfinance": {
            "mindate": yfinance_start,
            "maxdate": yfinance_end,
        },
        "model": {
            "name": model_name,
            "date": model_date,
        },
        "database": {
            "catalog": catalog_name,
            "schema": schema_name,
            "tables": {
                "stocks": f"{base}.{table_stocks}",
                "indicators": f"{base}.{table_indicators}",
                "volatility": f"{base}.{table_volatility}",
                "mc_market": f"{base}.{table_mc_market}",
                "mc_trials": f"{base}.{table_mc_trials}",
            },
        },
        "monte-carlo": {
            "executors": mc_executors,
            "volatility": mc_volatility_days,
            "runs": mc_runs,
        },
    }


def load_config_from_widgets(dbutils) -> dict:
    """
    Convenience helper to build the config dictionary directly from
    Databricks notebook widgets populated via job base_parameters.
    """
    def _get_or_default(name: str, default: str) -> str:
        """
        Try to read a widget; if it doesn't exist (e.g., interactive run),
        create it with the given default and return that default.
        """
        try:
            return dbutils.widgets.get(name)
        except Exception:
            dbutils.widgets.text(name, default)
            return default

    # Defaults here should stay in sync with databricks.yml variables
    catalog_name = _get_or_default("catalog_name", "areese_demo_catalog")
    schema_name = _get_or_default("schema_name", "value_at_risk")
    yfinance_start = _get_or_default("yfinance_start", "2023-01-01")
    yfinance_end = _get_or_default("yfinance_end", "2025-12-31")
    model_name = _get_or_default("model_name", "value_at_risk")
    model_date = _get_or_default("model_date", "2025-12-31")
    mc_executors = int(_get_or_default("mc_executors", "20"))
    mc_volatility_days = int(_get_or_default("mc_volatility_days", "90"))
    mc_runs = int(_get_or_default("mc_runs", "32000"))
    table_stocks = _get_or_default("table_stocks", "market_data")
    table_indicators = _get_or_default("table_indicators", "market_indicators")
    table_volatility = _get_or_default("table_volatility", "market_volatility")
    table_mc_market = _get_or_default("table_mc_market", "monte_carlo_market")
    table_mc_trials = _get_or_default("table_mc_trials", "monte_carlo_trials")

    return build_config(
        catalog_name=catalog_name,
        schema_name=schema_name,
        yfinance_start=yfinance_start,
        yfinance_end=yfinance_end,
        model_name=model_name,
        model_date=model_date,
        mc_executors=mc_executors,
        mc_volatility_days=mc_volatility_days,
        mc_runs=mc_runs,
        table_stocks=table_stocks,
        table_indicators=table_indicators,
        table_volatility=table_volatility,
        table_mc_market=table_mc_market,
        table_mc_trials=table_mc_trials,
    )


def configure_mlflow_experiment(dbutils) -> None:
    """
    Configure the MLflow experiment for the current user.
    """
    import mlflow

    username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    mlflow.set_experiment(f"/Users/{username}/value_at_risk")


def load_portfolio(path: str = "config/portfolio.json"):
    """
    Load the portfolio definition from JSON as a pandas DataFrame.
    """
    import pandas as pd

    return pd.read_json(path, orient="records")


def load_market_indicators(path: str = "config/indicators.json"):
    """
    Load the market indicators mapping from JSON.
    """
    import json

    with open(path, "r") as f:
        return json.load(f)


