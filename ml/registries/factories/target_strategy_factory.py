"""Registry mapping target-name/version keys to target strategy classes."""

from ml.targets.adr.v1 import AdrTargetV1
from ml.targets.cancellation.v1 import CancellationTargetV1
from ml.targets.lead_time.v1 import LeadTimeTargetV1
from ml.targets.no_show.v1 import NoShowTargetV1
from ml.targets.repeated_guest.v1 import RepeatedGuestTargetV1
from ml.targets.room_upgrade.v1 import RoomUpgradeTargetV1
from ml.targets.special_requests.v1 import SpecialRequestsTargetV1

TARGET_STRATEGIES = {
    ("adr", "v1"): AdrTargetV1,
    ("lead_time", "v1"): LeadTimeTargetV1,
    ("no_show", "v1"): NoShowTargetV1,
    ("is_repeated_guest", "v1"): RepeatedGuestTargetV1,
    ("room_upgrade", "v1"): RoomUpgradeTargetV1,
    ("total_of_special_requests", "v1"): SpecialRequestsTargetV1,
    ("is_canceled", "v1"): CancellationTargetV1,
}