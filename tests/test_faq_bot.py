import unittest
from io import StringIO
from pathlib import Path

from faq_bot import FAQMatcher, FAQValidationError, load_faqs, preprocess


class FAQDataTests(unittest.TestCase):
    def test_loads_required_and_optional_columns(self):
        faqs = load_faqs(
            StringIO("Question,Answer\nWhere is the library?,On campus.\n")
        )
        self.assertEqual(list(faqs.columns), [
            "question", "answer", "alternate_questions"
        ])
        self.assertEqual(faqs.iloc[0]["question"], "Where is the library?")

    def test_rejects_missing_required_columns(self):
        with self.assertRaisesRegex(FAQValidationError, "Missing required"):
            load_faqs(StringIO("question\nWhere is the library?\n"))

    def test_rejects_duplicate_questions(self):
        with self.assertRaisesRegex(FAQValidationError, "unique"):
            load_faqs(
                StringIO("question,answer\nWhere is the library?,A\n"
                         "WHERE IS THE LIBRARY?,B\n")
            )

    def test_drops_incomplete_rows_and_rejects_empty_data(self):
        with self.assertRaisesRegex(FAQValidationError, "at least one"):
            load_faqs(StringIO("question,answer\n,No question\n"))


class FAQMatchingTests(unittest.TestCase):
    def setUp(self):
        self.matcher = FAQMatcher(
            [
                {
                    "question": "How can I apply for admission?",
                    "answer": "Apply on the admissions portal.",
                    "alternate_questions": "How do I apply to the college?",
                },
                {
                    "question": "What are the library hours?",
                    "answer": "Check the library notice board.",
                    "alternate_questions": "",
                },
            ]
        )

    def test_matches_alternative_wording(self):
        result = self.matcher.match("How do I apply to the college?")
        self.assertTrue(result.matched)
        self.assertEqual(result.question, "How can I apply for admission?")
        self.assertEqual(result.answer, "Apply on the admissions portal.")

    def test_rejects_unrelated_questions(self):
        result = self.matcher.match("How do I bake a chocolate cake?")
        self.assertFalse(result.matched)
        self.assertIn("couldn't find a confident match", result.answer)

    def test_handles_blank_questions(self):
        result = self.matcher.match("   ")
        self.assertFalse(result.matched)
        self.assertEqual(result.score, 0)

    def test_rejects_invalid_confidence_threshold(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.matcher.match("Where is the library?", threshold=1.5)

    def test_preprocessing_is_case_insensitive_and_removes_noise(self):
        self.assertEqual(preprocess("LIBRARIES, please!"), preprocess("libraries"))

    def test_rejects_questions_without_shared_content_words(self):
        result = self.matcher.match("How do I bake a chocolate cake?")
        self.assertFalse(result.matched)
        self.assertEqual(result.score, 0)


class ProjectFAQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faq_path = Path(__file__).resolve().parents[1] / "faqs.csv"
        cls.faqs = load_faqs(faq_path)
        cls.matcher = FAQMatcher(cls.faqs.to_dict(orient="records"))

    def test_course_questions_select_the_right_catalog(self):
        cases = {
            "What courses are available?": "What courses are offered?",
            "Can u say the course names": "What courses are offered?",
            "Do you have B.Tech in AI and data science?":
                "What courses are available in engineering colleges?",
            "Is there a BA in psychology?":
                "What courses are available in arts colleges?",
            "What commerce courses are available?":
                "What courses are available in arts and science colleges?",
        }
        for question, expected_faq in cases.items():
            with self.subTest(question=question):
                result = self.matcher.match(question)
                self.assertTrue(result.matched, result)
                self.assertEqual(result.question, expected_faq)

    def test_sample_student_questions_are_covered(self):
        questions = [
            "How can I apply for admission?",
            "Does the college provide placement training?",
            "Are scholarships available?",
            "Is hostel accommodation available?",
            "When will semester exams be conducted?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(self.matcher.match(question).matched)

    def test_uploaded_file_can_be_loaded_more_than_once(self):
        source = StringIO("question,answer\nWhere is the library?,On campus.\n")
        self.assertEqual(len(load_faqs(source)), 1)
        self.assertEqual(len(load_faqs(source)), 1)


if __name__ == "__main__":
    unittest.main()
