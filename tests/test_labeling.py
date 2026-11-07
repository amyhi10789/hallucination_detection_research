import unittest

from hallucination_detection.labeling import label_answer, normalize_answer


class LabelingTests(unittest.TestCase):
    def test_normalization_ignores_articles_case_and_punctuation(self) -> None:
        self.assertEqual(normalize_answer("The New York City!"), "new york city")

    def test_alias_inside_concise_sentence_is_correct(self) -> None:
        result = label_answer("It was New York City.", ["New York City", "NYC"])
        self.assertEqual(result.label, "correct")
        self.assertEqual(result.matched_alias, "New York City")
        self.assertFalse(result.needs_manual_review)

    def test_abstention_is_separate(self) -> None:
        result = label_answer("I'm not sure.", ["New York City"])
        self.assertEqual(result.label, "abstain")
        self.assertFalse(result.needs_manual_review)

    def test_mismatch_is_flagged_for_manual_review(self) -> None:
        result = label_answer("Boston", ["New York City"])
        self.assertEqual(result.label, "incorrect")
        self.assertTrue(result.needs_manual_review)

    def test_generic_single_word_alias_does_not_match_long_wrong_answer(self) -> None:
        result = label_answer(
            "He belongs to the Evangelical Lutheran Church in Germany.",
            ["Catholic Church", "Roman Catholic Church", "Church"],
        )
        self.assertEqual(result.label, "incorrect")
        self.assertIsNone(result.matched_alias)
        self.assertTrue(result.needs_manual_review)

    def test_short_scaffold_can_match_single_word_answer(self) -> None:
        result = label_answer("It is actor.", ["actor"])
        self.assertEqual(result.label, "correct")


if __name__ == "__main__":
    unittest.main()
