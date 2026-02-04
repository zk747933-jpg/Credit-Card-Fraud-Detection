def rule_based_check(amount, hour, location_change, device_change, txn_count_24h):
    risk_score = 0

    # High amount + new location
    if amount > 25000 and location_change == 1:
        risk_score += 0.4

    # Late night transactions
    if hour < 5:
        risk_score += 0.2

    # New device
    if device_change == 1:
        risk_score += 0.2

    # Too many transactions
    if txn_count_24h > 10:
        risk_score += 0.3

    # Cap risk score at 1
    return min(risk_score, 1.0)
