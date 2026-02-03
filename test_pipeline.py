"""
Unit Tests for Medical NLP Pipeline
"""

import unittest
import json
from medical_nlp_pipeline import MedicalNLPPipeline


class TestMedicalNLPPipeline(unittest.TestCase):
    """Test cases for Medical NLP Pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.pipeline = MedicalNLPPipeline()
        cls.sample_transcript = """
        Physician: Good morning, Ms. Jones. How are you feeling today?
        Patient: I'm doing better, but I still have some neck pain.
        Physician: I understand you had a car accident. Can you tell me more?
        Patient: Yes, it was in September. I hit my head and had back pain for weeks.
        Physician: Did you seek treatment?
        Patient: Yes, I had physiotherapy sessions and took painkillers.
        """
    
    def test_extract_patient_info(self):
        """Test patient information extraction"""
        info = self.pipeline.extract_patient_info(self.sample_transcript)
        self.assertIn("Patient_Name", info)
        self.assertEqual(info["Patient_Name"], "Jones")
    
    def test_extract_symptoms(self):
        """Test symptom extraction"""
        symptoms = self.pipeline.extract_symptoms(self.sample_transcript)
        self.assertIsInstance(symptoms, list)
        self.assertTrue(len(symptoms) > 0)
        # Check if common symptoms are detected
        symptom_text = " ".join(symptoms).lower()
        self.assertTrue(any(term in symptom_text for term in ['neck', 'back', 'pain']))
    
    def test_extract_diagnosis(self):
        """Test diagnosis extraction"""
        diagnosis = self.pipeline.extract_diagnosis(self.sample_transcript)
        self.assertIsInstance(diagnosis, str)
        self.assertTrue(len(diagnosis) > 0)
    
    def test_extract_treatment(self):
        """Test treatment extraction"""
        treatments = self.pipeline.extract_treatment(self.sample_transcript)
        self.assertIsInstance(treatments, list)
        # Should detect physiotherapy and painkillers
        treatment_text = " ".join(treatments).lower()
        self.assertTrue('physiotherapy' in treatment_text or 'painkiller' in treatment_text)
    
    def test_summarize_medical_report(self):
        """Test complete medical summary generation"""
        summary = self.pipeline.summarize_medical_report(self.sample_transcript)
        
        # Check all required keys are present
        required_keys = [
            "Patient_Name", "Symptoms", "Diagnosis", 
            "Treatment", "Current_Status", "Prognosis"
        ]
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check data types
        self.assertIsInstance(summary["Symptoms"], list)
        self.assertIsInstance(summary["Treatment"], list)
        self.assertIsInstance(summary["Diagnosis"], str)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis"""
        test_dialogues = [
            "I'm worried about my recovery",
            "I'm feeling much better now",
            "The pain is manageable"
        ]
        
        for dialogue in test_dialogues:
            result = self.pipeline.analyze_sentiment(dialogue)
            
            # Check required keys
            self.assertIn("Sentiment", result)
            self.assertIn("Intent", result)
            
            # Check sentiment values
            self.assertIn(result["Sentiment"], ["Anxious", "Neutral", "Reassured"])
    
    def test_generate_soap_note(self):
        """Test SOAP note generation"""
        soap_note = self.pipeline.generate_soap_note(self.sample_transcript)
        
        # Check all SOAP sections are present
        required_sections = ["Subjective", "Objective", "Assessment", "Plan"]
        for section in required_sections:
            self.assertIn(section, soap_note)
        
        # Check structure of each section
        self.assertIsInstance(soap_note["Subjective"], dict)
        self.assertIsInstance(soap_note["Objective"], dict)
        self.assertIsInstance(soap_note["Assessment"], dict)
        self.assertIsInstance(soap_note["Plan"], dict)
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        keywords = self.pipeline.extract_keywords(self.sample_transcript)
        self.assertIsInstance(keywords, list)
        self.assertTrue(len(keywords) > 0)
        self.assertTrue(all(isinstance(kw, str) for kw in keywords))
    
    def test_empty_transcript(self):
        """Test handling of empty transcript"""
        summary = self.pipeline.summarize_medical_report("")
        self.assertIsInstance(summary, dict)
    
    def test_invalid_sentiment_input(self):
        """Test sentiment analysis with edge cases"""
        # Empty string
        result = self.pipeline.analyze_sentiment("")
        self.assertIn("Sentiment", result)
        
        # Very short text
        result = self.pipeline.analyze_sentiment("Pain.")
        self.assertIn("Sentiment", result)
    
    def test_json_serialization(self):
        """Test that outputs can be serialized to JSON"""
        summary = self.pipeline.summarize_medical_report(self.sample_transcript)
        soap_note = self.pipeline.generate_soap_note(self.sample_transcript)
        
        # Should not raise exception
        try:
            json.dumps(summary)
            json.dumps(soap_note)
            serializable = True
        except:
            serializable = False
        
        self.assertTrue(serializable)


class TestDataTypes(unittest.TestCase):
    """Test data type consistency"""
    
    def setUp(self):
        self.pipeline = MedicalNLPPipeline()
        self.transcript = "Patient: I have pain. Doctor: Let's examine you."
    
    def test_symptom_list_type(self):
        """Ensure symptoms are returned as list"""
        result = self.pipeline.extract_symptoms(self.transcript)
        self.assertIsInstance(result, list)
    
    def test_treatment_list_type(self):
        """Ensure treatments are returned as list"""
        result = self.pipeline.extract_treatment(self.transcript)
        self.assertIsInstance(result, list)
    
    def test_diagnosis_string_type(self):
        """Ensure diagnosis is returned as string"""
        result = self.pipeline.extract_diagnosis(self.transcript)
        self.assertIsInstance(result, str)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.pipeline = MedicalNLPPipeline()
    
    def test_very_long_transcript(self):
        """Test with very long transcript"""
        long_transcript = "Patient: Pain. " * 1000
        summary = self.pipeline.summarize_medical_report(long_transcript)
        self.assertIsInstance(summary, dict)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special_transcript = "Patient: I have pain!!! @#$%"
        summary = self.pipeline.summarize_medical_report(special_transcript)
        self.assertIsInstance(summary, dict)
    
    def test_mixed_case_transcript(self):
        """Test with mixed case input"""
        mixed_case = "PATIENT: I HAVE PAIN. Doctor: how are you?"
        summary = self.pipeline.summarize_medical_report(mixed_case)
        self.assertIsInstance(summary, dict)
    
    def test_no_patient_name(self):
        """Test transcript without patient name"""
        no_name = "Doctor: How are you? Patient: I have pain."
        info = self.pipeline.extract_patient_info(no_name)
        self.assertIn("Patient_Name", info)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMedicalNLPPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestDataTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
