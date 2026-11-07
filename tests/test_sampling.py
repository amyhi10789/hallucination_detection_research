import unittest
from collections import Counter, defaultdict

from hallucination_detection.sampling import build_paired_pilot, parse_answer_list


class SamplingTests(unittest.TestCase):
    @staticmethod
    def synthetic_rows() -> list[dict[str, object]]:
        relations = ["occupation", "place of birth", "genre", "father", "country", "producer"]
        rows: list[dict[str, object]] = []
        for index in range(36):
            relation = relations[index % len(relations)]
            subject = f"Subject {index}"
            rows.append(
                {
                    "id": index,
                    "subj": subject,
                    "prop": relation,
                    "obj": f"Answer {index}",
                    "s_pop": index + 1,
                    "question": f"Original question about {subject}?",
                    "s_wiki_title": subject,
                    "possible_answers": f'["Answer {index}", "Alias {index}"]',
                }
            )
        return rows

    def test_answer_list_parser(self) -> None:
        self.assertEqual(parse_answer_list('["A", "B"]'), ["A", "B"])
        self.assertEqual(parse_answer_list(["A", "B"]), ["A", "B"])
        self.assertEqual(parse_answer_list("A"), ["A"])

    def test_builds_five_fact_twenty_prompt_paired_pilot(self) -> None:
        rows, summary = build_paired_pilot(self.synthetic_rows(), base_fact_count=5, seed=7)
        self.assertEqual(len(rows), 20)
        self.assertEqual(summary["base_fact_count"], 5)
        self.assertEqual(summary["prompt_count"], 20)
        self.assertEqual(summary["band_counts_by_fact"], {"low": 2, "middle": 1, "high": 2})

        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["fact_id"])].append(row)
        self.assertEqual(len(grouped), 5)
        self.assertTrue(all(len(group) == 4 for group in grouped.values()))
        for group in grouped.values():
            self.assertEqual({row["variant_id"] for row in group}, {"original", "lexical", "syntactic", "concise"})
            self.assertEqual(len({row["question"] for row in group}), 4)
            self.assertEqual(len({row["popularity_band"] for row in group}), 1)

    def test_sampling_is_reproducible(self) -> None:
        first, _ = build_paired_pilot(self.synthetic_rows(), base_fact_count=5, seed=99)
        second, _ = build_paired_pilot(self.synthetic_rows(), base_fact_count=5, seed=99)
        self.assertEqual(first, second)
        self.assertEqual(Counter(row["fact_id"] for row in first), Counter(row["fact_id"] for row in second))

    def test_ambiguous_single_word_non_geographic_subject_is_excluded(self) -> None:
        rows = self.synthetic_rows()
        rows.append(
            {
                "id": 999,
                "subj": "Time",
                "prop": "composer",
                "obj": "Composer",
                "s_pop": 0,
                "question": "Who composed Time?",
                "possible_answers": '["Composer"]',
            }
        )
        pilot, _ = build_paired_pilot(rows, base_fact_count=5, seed=7)
        self.assertNotIn("popqa-999", {row["fact_id"] for row in pilot})

    def test_disambiguated_wikipedia_title_is_used_in_every_variant(self) -> None:
        rows = self.synthetic_rows()[:3]
        rows[0]["subj"] = "One Day"
        rows[0]["s_wiki_title"] = "One Day (2011 film)"
        rows[0]["question"] = "Original question about One Day?"
        pilot, _ = build_paired_pilot(rows, base_fact_count=3, seed=7)
        selected = [row for row in pilot if row["fact_id"] == "popqa-0"]
        self.assertEqual(len(selected), 4)
        self.assertTrue(all("One Day (2011 film)" in row["question"] for row in selected))


if __name__ == "__main__":
    unittest.main()
