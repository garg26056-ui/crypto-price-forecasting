import requests
import numpy as np
import pandas as pd
import streamlit as st

COINGECKO_LIST_URL = "https://api.coingecko.com/api/v3/coins/list"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_CHART_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

@st.cache(allow_output_mutation=True)
def load_coin_list():
    response = requests.get(COINGECKO_LIST_URL, timeout=20)
    response.raise_for_status()
    coin_data = response.json()
    return pd.DataFrame(coin_data)

@st.cache(allow_output_mutation=True)
def fetch_market_chart(coin_id: str, days: int = 2, vs_currency: str = "usd"):
    params = {"vs_currency": vs_currency, "days": days, "interval": "hourly"}
    response = requests.get(COINGECKO_CHART_URL.format(id=coin_id), params=params, timeout=20)
    response.raise_for_status()
    return response.json()

@st.cache(allow_output_mutation=True)
def fetch_current_price(coin_id: str, vs_currency: str = "usd"):
    params = {"ids": coin_id, "vs_currencies": vs_currency}
    response = requests.get(COINGECKO_PRICE_URL, params=params, timeout=10)
    response.raise_for_status()
    prices = response.json()
    return prices.get(coin_id, {}).get(vs_currency)


def find_coin_id(query: str, coins_df: pd.DataFrame):
    if not query:
        return None

    normalized = query.strip().lower()
    exact_id = coins_df[coins_df["id"] == normalized]
    if not exact_id.empty:
        return normalized

    exact_symbol = coins_df[coins_df["symbol"] == normalized]
    if len(exact_symbol) == 1:
        return exact_symbol.iloc[0]["id"]

    exact_name = coins_df[coins_df["name"].str.lower() == normalized]
    if not exact_name.empty:
        return exact_name.iloc[0]["id"]

    symbol_matches = coins_df[coins_df["symbol"].str.lower() == normalized]
    if not symbol_matches.empty:
        return symbol_matches.iloc[0]["id"]

    name_matches = coins_df[coins_df["name"].str.contains(normalized, case=False, na=False)]
    if not name_matches.empty:
        return name_matches.iloc[0]["id"]

    return None


def build_price_dataframe(market_data: dict) -> pd.DataFrame:
    prices = market_data.get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def predict_future_price(prices: np.ndarray) -> float:
    if len(prices) < 3:
        raise ValueError("Need at least 3 historical prices to predict.")
    x = np.arange(len(prices))
    slope, intercept = np.polyfit(x, prices, 1)
    next_index = len(prices)
    return float(slope * next_index + intercept)


def main():
    st.set_page_config(page_title="Crypto Price Predictor", layout="wide")
    st.title("Real-Time Crypto Price Prediction")
    st.write(
        "Enter any cryptocurrency name, symbol, or CoinGecko ID and get a live price feed plus a short-term price estimate."
    )

    coins_df = load_coin_list()
    common_coins = ["bitcoin", "ethereum", "cardano", "ripple", "solana", "dogecoin", "binancecoin"]
    top_choice = st.selectbox("Choose a popular crypto", common_coins, index=0)
    user_input = st.text_input("Search crypto by name or symbol", top_choice)

    coin_id = find_coin_id(user_input, coins_df)
    if coin_id is None:
        st.error("Could not match the entered crypto name or symbol. Please try a different value.")
        return

    with st.spinner(f"Loading market data for {coin_id}..."):
        try:
            current_price = fetch_current_price(coin_id)
            chart_data = fetch_market_chart(coin_id, days=2)
        except requests.RequestException as exc:
            st.error(f"Failed to fetch data from CoinGecko: {exc}")
            return

    if current_price is None or not chart_data.get("prices"):
        st.error("No price data available for this coin right now.")
        return

    df = build_price_dataframe(chart_data)
    predicted_price = None
    try:
        predicted_price = predict_future_price(df["price"].values)
    except ValueError:
        predicted_price = None

    col1, col2 = st.columns(2)
    col1.metric(label="Current Price (USD)", value=f"${current_price:,.4f}")
    if predicted_price is not None:
        delta = predicted_price - current_price
        delta_str = f"${delta:,.4f}"
        col2.metric(label="Predicted Next Hour Price", value=f"${predicted_price:,.4f}", delta=delta_str)
    else:
        col2.write("Not enough history to generate a prediction.")

    st.subheader("Historical price chart")
    st.write("DataFrame columns:", df.columns.tolist())

    st.markdown("---")
    st.subheader("Model details")
    st.write(
        "This app uses CoinGecko market data and a simple linear trend fit over the last 48 hours of hourly prices. "
        "The prediction is a short-term estimate, not financial advice."
    )

    with st.expander("Raw fetched values"):
        st.write(df.tail(8))


if __name__ == "__main__":
    main()
