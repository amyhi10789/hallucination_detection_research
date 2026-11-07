import unittest

from hallucination_detection.paraphrases import SUPPORTED_RELATIONS, make_question_variants


class ParaphraseTests(unittest.TestCase):
    def test_every_supported_relation_produces_four_distinct_variants(self) -> None:
        for relation in SUPPORTED_RELATIONS:
            with self.subTest(relation=relation):
                variants = make_question_variants(
                    original_question=f"Original question about Test Subject for {relation}?",
                    subject="Test Subject",
                    relation=relation,
                )
                self.assertEqual(len(variants), 4)
                self.assertEqual(len({variant.question.lower() for variant in variants}), 4)
                self.assertTrue(all("test subject" in variant.question.lower() for variant in variants))
                self.assertEqual(variants[0].variant_id, "original")

    def test_unsupported_relation_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            make_question_variants(
                original_question="When did the event occur?",
                subject="the event",
                relation="date",
            )


if __name__ == "__main__":
    unittest.main()
