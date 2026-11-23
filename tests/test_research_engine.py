import tempfile
import unittest
from pathlib import Path

from research_engine import (
    WorkflowSettings,
    load_memories,
    run_integrated_workflow,
)


class IntegratedWorkflowTests(unittest.TestCase):
    def test_offline_run_verifies_findings_and_persists_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "research_memory.json"
            settings = WorkflowSettings(
                online_llm=False,
                web_search=False,
                memory_path=memory_path,
            )

            first_run = run_integrated_workflow("Quantum computing", settings)
            first_records = load_memories(memory_path)
            second_run = run_integrated_workflow("Quantum computing", settings)

            self.assertTrue(first_run["report"].startswith("# Research report:"))
            self.assertTrue(first_run["verification"]["verified_findings"])
            self.assertFalse(first_run["verification"]["unsupported_claims"])
            self.assertEqual(
                first_run["run_status"]["llm_mode"],
                "OFFLINE DEMO MODE",
            )
            self.assertIn(
                "offline local sample",
                first_run["run_status"]["source_modes"],
            )
            self.assertEqual(
                len(first_records),
                first_run["memory_saved_count"],
            )
            self.assertEqual(load_memories(memory_path), first_records)
            self.assertEqual(
                second_run["memory_saved_count"],
                first_run["memory_saved_count"],
            )

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_integrated_workflow("  ")


if __name__ == "__main__":
    unittest.main()
