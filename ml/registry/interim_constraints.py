import numpy as np

MIN_CONSTRAINTS = {
    "lead_time": 0,
    "arrival_date_year": 2015,
    "arrival_date_week_number": 1,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 0,
    "adults": 0,
    "children": 0,
    "babies": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "adr": 0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "agent": 1,
    "company": 1,
}

MAX_CONSTRAINTS = {
    "arrival_date_week_number": 53,
    "arrival_date_day_of_month": 31,
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
    "agent": [np.nan] + [str(i) for i in range(1, 1000)],
    "company": [np.nan] + [str(i) for i in range(1, 1000)],
    "country": [
        'PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', np.nan, 'ROU', 'NOR', 'OMN', 'ARG', 'POL'
        'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'SWE', 'AUS', 'EST'
        'CZE', 'BRA', 'FIN', 'MOZ', 'BWA', 'LUX', 'SVN', 'ALB', 'IND', 'CHN', 'MEX', 'MAR'
        'UKR', 'SMR', 'LVA', 'PRI', 'SRB', 'CHL', 'AUT', 'BLR', 'LTU', 'TUR', 'ZAF', 'AGO'
        'ISR', 'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA', 'KOR', 'CRI', 'HUN', 'ARE', 'TUN', 'JAM'
        'HRV', 'HKG', 'IRN', 'GEO', 'AND', 'GIB', 'URY', 'JEY', 'CAF', 'CYP', 'COL', 'GGY'
        'KWT', 'NGA', 'MDV', 'VEN', 'SVK', 'FJI', 'KAZ', 'PAK', 'IDN', 'LBN', 'PHL', 'SEN'
        'SYC', 'AZE', 'BHR', 'NZL', 'THA', 'DOM', 'MKD', 'MYS', 'ARM', 'JPN', 'LKA', 'CUB'
        'CMR', 'BIH', 'MUS', 'COM', 'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP', 'BDI'
        'SAU', 'VNM', 'PLW', 'QAT', 'EGY', 'PER', 'MLT', 'MWI', 'ECU', 'MDG', 'ISL', 'UZB'
        'NPL', 'BHS', 'MAC', 'TGO', 'TWN', 'DJI', 'STP', 'KNA', 'ETH', 'IRQ', 'HND', 'RWA'
        'KHM', 'MCO', 'BGD', 'IMN', 'TJK', 'NIC', 'BEN', 'VGB', 'TZA', 'GAB', 'GHA', 'TMP'
        'GLP', 'KEN', 'LIE', 'GNB', 'MNE', 'UMI', 'MYT', 'FRO', 'MMR', 'PAN', 'BFA', 'LBY'
        'MLI', 'NAM', 'BOL', 'PRY', 'BRB', 'ABW', 'AIA', 'SLV', 'DMA', 'PYF', 'GUY', 'LCA'
        'ATA', 'GTM', 'ASM', 'MRT', 'NCL', 'KIR', 'SDN', 'ATF', 'SLE', 'LAO'
    ]
}