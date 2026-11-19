import os
import unittest
from semantic_search import load_chunks,build_vectorizer
from agent import answer_question,build_prompt,LOG_FILE


class TestPawPalsAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks=load_chunks()
        cls.vectorizer,cls.matrix=build_vectorizer(cls.chunks)

    def test_invalid_input(self):
        """Tests that empty or whitespace queries trigger the input validation fallback."""
        res_empty=answer_question("",self.vectorizer,self.matrix,self.chunks)
        self.assertEqual(res_empty["status"],"INVALID_INPUT")
        self.assertIn("Please enter a valid",res_empty["answer"])

        res_space=answer_question("    ",self.vectorizer,self.matrix,self.chunks)
        self.assertEqual(res_space["status"],"INVALID_INPUT")

    def test_no_context_fallback(self):
        """Tests that questions outside the clinic's scope trigger the no-context fallback."""
        unrelated_query="How to repair an electric car battery in rain?"
        res=answer_question(unrelated_query,self.vectorizer,self.matrix,self.chunks)
        self.assertEqual(res["status"],"NO_CONTEXT")
        self.assertIn("don't have information on that specific topic",res["answer"])

    def test_successful_retrieval_and_citation(self):
        """Tests that clinic queries retrieve the right document and include citations."""
        query="puppy core vaccines and booster timeline"
        res=answer_question(query,self.vectorizer,self.matrix,self.chunks)
        self.assertIn(res["status"],["SUCCESS","FALLBACK"])
        self.assertTrue(len(res["sources"]) > 0)
        self.assertIn("dog_vaccination_schedule.txt",res["sources"])
        self.assertIn("dog_vaccination_schedule.txt",res["answer"])

    def test_prompt_construction(self):
        """Tests that build_prompt properly formats clinic context and citation rules."""
        sample_chunks=[
            {"source": "sample_doc.txt","text": "Sample pet advice for testing."}
        ]
        sys_p,user_p=build_prompt("How to care for pet?",sample_chunks)
        self.assertIn("PawPals Pet Clinic",sys_p)
        self.assertIn("sample_doc.txt",user_p)
        self.assertIn("Sample pet advice for testing.",user_p)

    def test_logging_mechanism(self):
        """Verifies that queries produce log records in agent.log."""
        self.assertTrue(os.path.exists(LOG_FILE))
        with open(LOG_FILE,"r",encoding="utf-8") as f:
            logs=f.read()
        self.assertIn("[INFO]",logs)


if __name__=="__main__":
    unittest.main(verbosity=2)
