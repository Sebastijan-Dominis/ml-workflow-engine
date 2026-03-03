"""Registry of datasets requiring row-id generation and their handlers."""

from ml.data.processed.processing.hotel_bookings.add_row_id import \
    AddRowIDToHotelBookings

ROW_ID_REQUIRED = [
    "hotel_bookings"
]

ROW_ID_FUNCTIONS = {
    "hotel_bookings": AddRowIDToHotelBookings().add_row_id
}