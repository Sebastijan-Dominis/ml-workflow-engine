def iso_no_colon(dt):
    return dt.isoformat(timespec="seconds").replace(":", "-")