import os
import unittest
from semantic_search import load_chunks,build_vectorizer,search


class TestSemanticSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks=load_chunks()
        cls.vectorizer,cls.matrix=build_vectorizer(cls.chunks)

    def test_empty_query_handling(self):
        """Checks that empty or whitespace queries return empty results without crashing."""
        self.assertEqual(search("",self.vectorizer,self.matrix,self.chunks),[])
        self.assertEqual(search("   ",self.vectorizer,self.matrix,self.chunks),[])
        self.assertEqual(search(None,self.vectorizer,self.matrix,self.chunks),[])

    def test_search_accuracy(self):
        """Verifies that sample clinic queries return relevant documents at rank 1."""
        test_cases=[
            ("Puppy vaccination schedule core shots","dog_vaccination_schedule.txt"),
            ("Foods to avoid feeding cats chocolate garlic","cat_nutrition_guide.txt"),
            ("Emergency signs difficulty breathing collapse","pet_emergency_signs.txt"),
            ("Dental care brushing teeth gum disease","pet_dental_care.txt"),
            ("Flea and tick prevention collar treatment","flea_tick_prevention.txt"),
            ("Puppy crate house training routine","puppy_house_training.txt")
        ]

        for query,expected_source in test_cases:
            results=search(query,self.vectorizer,self.matrix,self.chunks,top_k=3)
            self.assertTrue(len(results) > 0,f"No results returned for: {query}")
            top_source=results[0]["source"]
            self.assertEqual(
                top_source,
                expected_source,
                f"For query '{query}', expected top source {expected_source} but got {top_source}"
            )

    def test_top_k_parameter(self):
        """Verifies top_k limits the number of matches returned."""
        query="puppy health care"
        res_1=search(query,self.vectorizer,self.matrix,self.chunks,top_k=1)
        res_3=search(query,self.vectorizer,self.matrix,self.chunks,top_k=3)
        self.assertEqual(len(res_1),1)
        self.assertEqual(len(res_3),3)

    def test_score_validity(self):
        """Verifies that similarity scores are valid floats between 0 and 1."""
        results=search("pet nutrition",self.vectorizer,self.matrix,self.chunks,top_k=3)
        for r in results:
            self.assertGreater(r["score"],0.0)
            self.assertLessEqual(r["score"],1.0)


if __name__=="__main__":
    unittest.main(verbosity=2)
