"""Runner for adversarial tests."""
import logging
from typing import Optional

from outhad_contextkit.memory.privacy.adversarial.test_suite import AdversarialTestSuite

logger = logging.getLogger(__name__)


class AdversarialTestRunner:
    """Manages running adversarial tests at intervals."""
    
    def __init__(self, memory_instance, interval: int = 100):
        self.memory = memory_instance
        self.interval = interval
        self.operation_count = 0
        self.test_suite = AdversarialTestSuite(memory_instance)
        self.last_report = None
    
    def should_run_test(self) -> bool:
        """Check if it's time to run adversarial tests."""
        self.operation_count += 1
        return self.operation_count % self.interval == 0
    
    def run_if_needed(self, user_id: str = "test_user") -> Optional[dict]:
        """Run adversarial tests if interval reached."""
        if self.should_run_test():
            logger.info(f"Running adversarial tests (operation {self.operation_count})")
            report = self.test_suite.run_test_suite(user_id)
            self.last_report = report
            
            # Log summary
            logger.info(
                f"Adversarial Test Report: "
                f"{report['passed']}/{report['total_tests']} passed, "
                f"{report['leaked_attacks']} attacks leaked sensitive data"
            )
            
            return report
        
        return None

