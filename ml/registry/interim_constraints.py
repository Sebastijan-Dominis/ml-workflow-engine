import numpy as np

from ml.data.config.schemas.constants import BorderValue

MIN_CONSTRAINTS = {
    "lead_time": BorderValue(value=0, op="gte"),
    "arrival_date_year": BorderValue(value=2015, op="gte"),
    "arrival_date_week_number": BorderValue(value=1, op="gte"),
    "arrival_date_day_of_month": BorderValue(value=1, op="gte"),
    "stays_in_weekend_nights": BorderValue(value=0, op="gte"),
    "stays_in_week_nights": BorderValue(value=0, op="gte"),
    "adults": BorderValue(value=0, op="gte"),
    "children": BorderValue(value=0, op="gte"),
    "babies": BorderValue(value=0, op="gte"),
    "previous_cancellations": BorderValue(value=0, op="gte"),
    "previous_bookings_not_canceled": BorderValue(value=0, op="gte"),
    "booking_changes": BorderValue(value=0, op="gte"),
    "days_in_waiting_list": BorderValue(value=0, op="gte"),
    "adr": BorderValue(value=0, op="gte"),
    "required_car_parking_spaces": BorderValue(value=0, op="gte"),
    "total_of_special_requests": BorderValue(value=0, op="gte")
}

MAX_CONSTRAINTS = {
    "arrival_date_week_number": BorderValue(value=53, op="lte"),
    "arrival_date_day_of_month": BorderValue(value=31, op="lte"),
}

ALLOWED_VALUES_CONSTRAINTS = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "arrival_date_month": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ],
    "meal": ["BB", "HB", "FB", "SC", "Undefined"],
    "market_segment": [
        "Aviation", "Direct", "Complementary", "Corporate", "Groups", "Offline TA/TO", "Online TA", "Undefined"
    ],
    "distribution_channel": [
        "Direct", "Corporate", "GDS", "TA/TO", "Undefined"
    ],
    "deposit_type": ["No Deposit", "Refundable", "Non Refund"],
    "customer_type": [
        "Contract", "Transient", "Transient-Party", "Group"
    ],
    "reservation_status": ["Canceled", "Check-Out", "No-Show"],
    "reserved_room_type": [
        "A", "B", "C", "D", "E", "F", "G", "H", "L", "P"
    ],
    "assigned_room_type": [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "P"
    ],
    "agent": [str(float(i)) for i in range(1, 1000)],
    "company": [str(float(i)) for i in range(1, 1000)],
    "country": [
        'PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN', 'ARG', 'POL',
        'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'SWE', 'AUS', 'EST',
        'CZE', 'BRA', 'FIN', 'MOZ', 'BWA', 'LUX', 'SVN', 'ALB', 'IND', 'CHN', 'MEX', 'MAR',
        'UKR', 'SMR', 'LVA', 'PRI', 'SRB', 'CHL', 'AUT', 'BLR', 'LTU', 'TUR', 'ZAF', 'AGO',
        'ISR', 'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA', 'KOR', 'CRI', 'HUN', 'ARE', 'TUN', 'JAM',
        'HRV', 'HKG', 'IRN', 'GEO', 'AND', 'GIB', 'URY', 'JEY', 'CAF', 'CYP', 'COL', 'GGY',
        'KWT', 'NGA', 'MDV', 'VEN', 'SVK', 'FJI', 'KAZ', 'PAK', 'IDN', 'LBN', 'PHL', 'SEN',
        'SYC', 'AZE', 'BHR', 'NZL', 'THA', 'DOM', 'MKD', 'MYS', 'ARM', 'JPN', 'LKA', 'CUB',
        'CMR', 'BIH', 'MUS', 'COM', 'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP', 'BDI',
        'SAU', 'VNM', 'PLW', 'QAT', 'EGY', 'PER', 'MLT', 'MWI', 'ECU', 'MDG', 'ISL', 'UZB',
        'NPL', 'BHS', 'MAC', 'TGO', 'TWN', 'DJI', 'STP', 'KNA', 'ETH', 'IRQ', 'HND', 'RWA',
        'KHM', 'MCO', 'BGD', 'IMN', 'TJK', 'NIC', 'BEN', 'VGB', 'TZA', 'GAB', 'GHA', 'TMP',
        'GLP', 'KEN', 'LIE', 'GNB', 'MNE', 'UMI', 'MYT', 'FRO', 'MMR', 'PAN', 'BFA', 'LBY',
        'MLI', 'NAM', 'BOL', 'PRY', 'BRB', 'ABW', 'AIA', 'SLV', 'DMA', 'PYF', 'GUY', 'LCA',
        'ATA', 'GTM', 'ASM', 'MRT', 'NCL', 'KIR', 'SDN', 'ATF', 'SLE', 'LAO'
    ]
}
