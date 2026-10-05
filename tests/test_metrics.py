from services.analysis import calculate_metrics


def test_derives_all_metrics():
    out = calculate_metrics(
        {"spend": 100, "revenue": 400, "clicks": 50, "impressions": 1000, "conversions": 5}
    )
    assert out["roas"] == 4.0
    assert out["ctr"] == 5.0
    assert out["cpc"] == 2.0
    assert out["conversion_rate"] == 10.0


def test_only_spend_and_revenue_derives_roas_only():
    out = calculate_metrics({"spend": 100, "revenue": 200})
    assert out["roas"] == 2.0
    assert not {"ctr", "cpc", "conversion_rate"} & out.keys()


def test_zero_values_do_not_divide_by_zero():
    out = calculate_metrics({"spend": 0, "revenue": 100, "clicks": 0, "impressions": 0})
    assert not {"roas", "ctr", "cpc"} & out.keys()


def test_does_not_mutate_input():
    data = {"spend": 100, "revenue": 200}
    calculate_metrics(data)
    assert data == {"spend": 100, "revenue": 200}
