from .cbbaPolicy import CBBAPolicy
from .dstaPolicy import DSTAPolicy
from .lstaPolicy import LSTAPolicy
from .lrcaPolicy import LRCAPolicy

ta_policy_dict = {
    # policy
    'cbba': CBBAPolicy,
    'dsta': DSTAPolicy,
    'lsta': LSTAPolicy,
    'lrca': LRCAPolicy,
    # learning

}
