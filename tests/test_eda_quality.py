import pandas as pd

from skyulf_mlflow_library.eda import generate_quality_report, get_text_insights


def test_generate_quality_report_basic():
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 40, 35],
            "salary": [50000, 60000, 55000, None, 70000],
            "city": ["NYC", "LA", "NYC", "SF", "LA"],
            "notes": [
                "Customer reported great service",
                "Repeat client",
                "Customer reported great service",
                "New account",
                "VIP segment customer",
            ],
        }
    )

    report = generate_quality_report(df)

    assert report["success"] is True
    assert report["summary"]["rows"] == 5
    assert any(col["name"] == "city" for col in report["columns"])
    assert report["recommendations"]
    assert report["quality_score"] >= 0


def test_get_text_insights_flags():
    series = pd.Series([
        "HELLO",
        "hello",
        "hello",
        "HELLO",
        "hello",
    ])
    insights = get_text_insights(series, "greeting")
    assert insights["data_category"] == "text"
    assert "quality_flags" in insights
