# Feature set logic

This is a list of all of the possible input columns (not including operators/derived features) that may appear in each set. 

## booking_context_features:

### Description

They describe the booking itself, independent of the guest's past.

### List

- hotel
- lead_time
- arrival_date
- arrival_date_year
- arrival_date_month
- arrival_date_week_number
- arrival_date_day_of_month
- stays_in_weekend_nights
- stays_in_week_nights
- meal
- deposit_type
- days_in_waiting_list

## party_composition_features:

### Description

They answer the question of who is traveling.

### List

- adults
- children
- babies

## customer_history_features

### Description

They describe what the customer has done before.

### List

- is_repeated_guest
- previous_cancellations
- previous_bookings_not_cancelled
- customer_type

## pricing_features

### Description

They describe the pricing of the booking.

### List

- adr
- required_car_parking_spaces
- total_of_special_requests

## channel_and_agent_features

### Description

They describe how the booking came in.

### List

- country
- agent
- market_segment
- distribution_channel

## room_allocation_features

### Description

They describe hotel-side allocation and constraints.

### List

- reserved_room_type
- assigned_room_type
