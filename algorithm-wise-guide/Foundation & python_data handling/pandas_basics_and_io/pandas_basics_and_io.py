from io import StringIO
import json
import sqlite3

import pandas as pd


def pandas_objects_demo():
    city_sales = pd.Series(
        [120, 95, 140],
        index=["Delhi", "Pune", "Jaipur"],
        name="weekly_sales",
    )

    orders = pd.DataFrame(
        {
            "city": ["Delhi", "Pune", "Jaipur"],
            "sales": [120, 95, 140],
            "returns": [4, 6, 3],
        }
    )
    return city_sales, orders


def csv_example():
    csv_text = """order_id,customer,amount
1,Ana,120
2,Raj,340
3,Sam,180
"""
    return pd.read_csv(StringIO(csv_text))


def json_sql_api_examples():
    payload = [
        {"customer": "Ana", "amount": 120, "city": "Delhi"},
        {"customer": "Raj", "amount": 340, "city": "Pune"},
        {"customer": "Sam", "amount": 180, "city": "Jaipur"},
    ]
    json_df = pd.json_normalize(json.loads(json.dumps(payload)))

    with sqlite3.connect(":memory:") as connection:
        json_df.to_sql("orders", connection, index=False)
        sql_df = pd.read_sql_query(
            "SELECT customer, amount FROM orders WHERE amount >= 180",
            connection,
        )

    return json_df, sql_df


def html_table_example():
    html_text = """
    <table>
        <tr><th>month</th><th>signups</th></tr>
        <tr><td>Jan</td><td>34</td></tr>
        <tr><td>Feb</td><td>51</td></tr>
        <tr><td>Mar</td><td>48</td></tr>
    </table>
    """
    return pd.read_html(StringIO(html_text))[0]


if __name__ == "__main__":
    city_sales, orders = pandas_objects_demo()
    print("Series example:")
    print(city_sales)
    print()

    print("DataFrame example:")
    print(orders)
    print()

    print("CSV example:")
    print(csv_example())
    print()

    json_df, sql_df = json_sql_api_examples()
    print("JSON to DataFrame:")
    print(json_df)
    print()

    print("SQL query result:")
    print(sql_df)
    print()

    print("HTML table example:")
    print(html_table_example())

