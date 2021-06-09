# Expose models and model managers
# Ambiguity of join-row complaints multi-class classification
# Ambiguity of join-row complaints binary classification
# Ambiguity of multi-class classification single-class-count complaints
# Expose ambiguity calculations
# Ambiguity of binary classification count complaints
import logging


# Expose Fixers
from .fixer import AutoFixer, Fixer, OracleFixer
from .manager import ModelManager
from .models import LogReg, SimpleCNN

# Expose the processor interface
from .processor import ComplaintRet, Processor

# Expose our rankers
from .rankers import (
    InfluenceRanker,
    LossRanker,
    SelfLossInfluenceRanker,
    TiresiasRanker,
)

logging.basicConfig(level=logging.INFO)

# Fixes single-class-counts : classes >=2
# from .tiresias.tiresias_count import (
#     minimal_single_count_fix_multi,
#     minimal_mutli_count_fix_multi,
#     minimal_set_count_fix,
# )

# Fixes for join counts: Special 2-class case
# from .tiresias.tiresias_join_count_binary import minimal_fix_join_count

# Fixes for join counts: classes >=2
# from .tiresias.tiresias_join_count_multi import minimal_fix_join_count_multi

# Fixes for join rows: classes >=2
# from .tiresias.tiresias_join_rows import minimal_join_rows_fix_multi

# Expose Tiresias related stuff of the package
# The general Tiresias selector interface
