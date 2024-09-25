import unittest
from app import PhraseSimilarityApp


class TestPhraseSimilarityApp(unittest.TestCase):
    def test_similarity(self):
        app = PhraseSimilarityApp('pycode/vectors.csv', 'pycode/phrases.csv')
        phrase, distance = app.find_similarity("sample phrase")
        self.assertTrue(phrase)
        self.assertGreaterEqual(distance, 0)


if __name__ == '__main__':
    unittest.main()
