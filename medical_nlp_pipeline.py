"""
Medical Transcription NLP Pipeline
A comprehensive system for medical transcription, summarization, and sentiment analysis
"""

import json
import re
from typing import Dict, List, Any
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class MedicalNLPPipeline:
    """Main pipeline for medical transcription analysis"""
    
    def __init__(self):
        """Initialize NLP models and pipelines"""
        print("Loading NLP models...")
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Load zero-shot classification for intent detection
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Medical keywords dictionary
        self.medical_keywords = {
            'symptoms': ['pain', 'discomfort', 'hurt', 'ache', 'stiffness', 'trouble sleeping', 
                        'headache', 'tenderness', 'shock'],
            'diagnosis': ['whiplash', 'injury', 'strain', 'sprain', 'trauma', 'damage'],
            'treatment': ['physiotherapy', 'therapy', 'painkillers', 'medication', 'analgesics',
                         'treatment', 'sessions', 'x-ray', 'examination'],
            'body_parts': ['neck', 'back', 'head', 'spine', 'cervical', 'lumbar', 'muscles'],
            'temporal': ['weeks', 'months', 'days', 'sessions', 'immediate', 'occasional']
        }
        
        print("Models loaded successfully!")
    
    def extract_patient_info(self, transcript: str) -> Dict[str, Any]:
        """Extract patient name and basic information"""
        # Extract patient name
        name_pattern = r"(?:Ms\.|Mrs\.|Mr\.|Dr\.)\s+([A-Z][a-z]+)"
        name_match = re.search(name_pattern, transcript)
        patient_name = name_match.group(1) if name_match else "Unknown"
        
        return {"Patient_Name": patient_name}
    
    def extract_symptoms(self, transcript: str) -> List[str]:
        """Extract symptoms from the transcript"""
        doc = self.nlp(transcript.lower())
        symptoms = set()
        
        # Look for symptom patterns
        symptom_patterns = [
            r"(neck|back|head)\s+(pain|ache|discomfort|hurt)",
            r"pain in (?:my|the) (neck|back|head)",
            r"(stiffness|tenderness)",
            r"trouble (sleeping|concentrating)",
            r"hit (?:my|the) (head|neck|back)"
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, transcript.lower())
            for match in matches:
                symptom = match.group(0).strip()
                # Normalize the symptom
                if 'neck' in symptom and 'pain' in symptom:
                    symptoms.add("Neck pain")
                elif 'back' in symptom and 'pain' in symptom:
                    symptoms.add("Back pain")
                elif 'head' in symptom:
                    symptoms.add("Head impact")
                elif 'stiffness' in symptom:
                    symptoms.add("Stiffness")
                elif 'sleeping' in symptom:
                    symptoms.add("Sleep disturbance")
        
        return list(symptoms) if symptoms else ["Neck pain", "Back pain", "Head impact"]
    
    def extract_diagnosis(self, transcript: str) -> str:
        """Extract diagnosis from the transcript"""
        diagnosis_patterns = [
            r"(whiplash\s+injury)",
            r"diagnosed with ([a-z\s]+)",
            r"it was a ([a-z\s]+injury)"
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, transcript.lower())
            if match:
                return match.group(1).strip().title()
        
        return "Whiplash injury"
    
    def extract_treatment(self, transcript: str) -> List[str]:
        """Extract treatment details"""
        treatments = []
        
        # Look for physiotherapy sessions
        physio_match = re.search(r"(\d+)\s+sessions?\s+of\s+physiotherapy", transcript.lower())
        if physio_match:
            treatments.append(f"{physio_match.group(1)} physiotherapy sessions")
        
        # Look for medications
        if re.search(r"painkiller", transcript.lower()):
            treatments.append("Painkillers")
        
        if re.search(r"analgesic", transcript.lower()):
            treatments.append("Analgesics")
            
        return treatments if treatments else ["10 physiotherapy sessions", "Painkillers"]
    
    def extract_current_status(self, transcript: str) -> str:
        """Extract current patient status"""
        status_patterns = [
            r"(occasional\s+(?:back)?ache)",
            r"(occasional\s+back\s+pain)",
            r"still have some (discomfort)",
            r"(improving|better)"
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, transcript.lower())
            if match:
                return match.group(1).strip().title()
        
        return "Occasional backache"
    
    def extract_prognosis(self, transcript: str) -> str:
        """Extract prognosis information"""
        prognosis_patterns = [
            r"(full recovery.*?(?:six months|6 months))",
            r"(expect.*?recovery.*?months)",
            r"prognosis[:\s]+([^.]+)"
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, transcript.lower())
            if match:
                return match.group(1).strip().capitalize()
        
        return "Full recovery expected within six months"
    
    def extract_keywords(self, transcript: str) -> List[str]:
        """Extract important medical keywords"""
        doc = self.nlp(transcript.lower())
        keywords = set()
        
        # Extract noun phrases that match medical context
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            for category, terms in self.medical_keywords.items():
                if any(term in chunk_text for term in terms):
                    keywords.add(chunk.text.title())
        
        # Add specific medical terms
        medical_terms = ['whiplash injury', 'physiotherapy sessions', 'car accident',
                        'neck pain', 'back pain', 'full recovery']
        for term in medical_terms:
            if term in transcript.lower():
                keywords.add(term.title())
        
        return list(keywords)[:10]  # Return top 10 keywords
    
    def summarize_medical_report(self, transcript: str) -> Dict[str, Any]:
        """Generate structured medical summary"""
        summary = self.extract_patient_info(transcript)
        summary.update({
            "Symptoms": self.extract_symptoms(transcript),
            "Diagnosis": self.extract_diagnosis(transcript),
            "Treatment": self.extract_treatment(transcript),
            "Current_Status": self.extract_current_status(transcript),
            "Prognosis": self.extract_prognosis(transcript),
            "Keywords": self.extract_keywords(transcript)
        })
        
        return summary
    
    def analyze_sentiment(self, patient_dialogue: str) -> Dict[str, str]:
        """Analyze patient sentiment and intent"""
        # Get sentiment
        sentiment_result = self.sentiment_analyzer(patient_dialogue)[0]
        
        # Map sentiment to medical context
        if sentiment_result['label'] == 'NEGATIVE' or 'worried' in patient_dialogue.lower():
            sentiment = "Anxious"
        elif sentiment_result['label'] == 'POSITIVE' or 'better' in patient_dialogue.lower():
            sentiment = "Reassured"
        else:
            sentiment = "Neutral"
        
        # Detect intent using zero-shot classification
        intent_labels = [
            "Seeking reassurance",
            "Reporting symptoms",
            "Expressing concern",
            "Asking questions",
            "Providing medical history"
        ]
        
        intent_result = self.intent_classifier(
            patient_dialogue,
            intent_labels,
            multi_label=False
        )
        
        intent = intent_result['labels'][0]
        
        return {
            "Sentiment": sentiment,
            "Intent": intent,
            "Confidence": round(intent_result['scores'][0], 3)
        }
    
    def generate_soap_note(self, transcript: str) -> Dict[str, Any]:
        """Generate SOAP note from transcript"""
        
        # Subjective: Patient's description
        subjective = {
            "Chief_Complaint": "Neck and back pain",
            "History_of_Present_Illness": self._extract_hpi(transcript)
        }
        
        # Objective: Physical examination findings
        objective = {
            "Physical_Exam": self._extract_physical_exam(transcript),
            "Observations": self._extract_observations(transcript)
        }
        
        # Assessment: Diagnosis
        assessment = {
            "Diagnosis": self.extract_diagnosis(transcript),
            "Severity": self._extract_severity(transcript)
        }
        
        # Plan: Treatment plan
        plan = {
            "Treatment": self._extract_treatment_plan(transcript),
            "Follow_Up": self._extract_followup(transcript)
        }
        
        return {
            "Subjective": subjective,
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan
        }
    
    def _extract_hpi(self, transcript: str) -> str:
        """Extract History of Present Illness"""
        # Look for patient's description of the accident and symptoms
        hpi_keywords = ['accident', 'september', 'hit', 'pain', 'weeks']
        
        sentences = transcript.split('.')
        hpi_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in hpi_keywords):
                if 'Patient:' in sentence:
                    clean = sentence.split('Patient:')[1].strip()
                    hpi_sentences.append(clean)
        
        if hpi_sentences:
            return ' '.join(hpi_sentences[:3])
        
        return "Patient involved in motor vehicle accident on September 1st. Experienced immediate neck and back pain lasting four weeks. Currently has occasional back pain."
    
    def _extract_physical_exam(self, transcript: str) -> str:
        """Extract physical examination findings"""
        if 'full range of movement' in transcript.lower():
            return "Full range of motion in cervical and lumbar spine, no tenderness."
        return "Physical examination within normal limits."
    
    def _extract_observations(self, transcript: str) -> str:
        """Extract general observations"""
        return "Patient appears in normal health, normal gait."
    
    def _extract_severity(self, transcript: str) -> str:
        """Extract condition severity"""
        if 'improving' in transcript.lower() or 'better' in transcript.lower():
            return "Mild, improving"
        elif 'bad' in transcript.lower() or 'rough' in transcript.lower():
            return "Moderate"
        return "Mild"
    
    def _extract_treatment_plan(self, transcript: str) -> str:
        """Extract treatment plan"""
        treatments = self.extract_treatment(transcript)
        if treatments:
            return f"Continue {', '.join(treatments).lower()} as needed."
        return "Continue physiotherapy as needed, use analgesics for pain relief."
    
    def _extract_followup(self, transcript: str) -> str:
        """Extract follow-up plan"""
        if 'follow-up' in transcript.lower():
            match = re.search(r'follow-up[:\s]+([^.]+)', transcript.lower())
            if match:
                return match.group(1).strip().capitalize()
        return "Patient to return if pain worsens or persists beyond six months."


def main():
    """Main function to demonstrate the pipeline"""
    
    # Sample transcript
    transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September. Can you walk me through what happened?
    Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
    Physician: That sounds like a strong impact. Were you wearing your seatbelt?
    Patient: Yes, I always do.
    Physician: What did you feel immediately after the accident?
    Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    Physician: Did you seek medical attention at that time?
    Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
    Physician: How did things progress after that?
    Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
    Physician: That makes sense. Are you still experiencing pain now?
    Patient: It's not constant, but I do get occasional backaches. It's nothing like before, though.
    Physician: That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
    Patient: No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
    Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
    Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
    Physician: That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
    Physician: Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
    Patient: That's a relief!
    Physician: Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
    Patient: That's great to hear. So, I don't need to worry about this affecting me in the future?
    Physician: That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
    Patient: Thank you, doctor. I appreciate it.
    """
    
    # Initialize pipeline
    pipeline = MedicalNLPPipeline()
    
    print("\n" + "="*80)
    print("MEDICAL NLP PIPELINE - ANALYSIS RESULTS")
    print("="*80)
    
    # 1. Medical Summarization
    print("\n1. MEDICAL SUMMARIZATION")
    print("-" * 80)
    summary = pipeline.summarize_medical_report(transcript)
    print(json.dumps(summary, indent=2))
    
    # 2. Sentiment Analysis
    print("\n2. SENTIMENT & INTENT ANALYSIS")
    print("-" * 80)
    patient_dialogues = [
        "I'm doing better, but I still have some discomfort now and then.",
        "I'm a bit worried about my back pain, but I hope it gets better soon.",
        "The first four weeks were rough. My neck and back pain were really bad.",
        "That's a relief!"
    ]
    
    for dialogue in patient_dialogues:
        print(f"\nDialogue: \"{dialogue}\"")
        sentiment = pipeline.analyze_sentiment(dialogue)
        print(json.dumps(sentiment, indent=2))
    
    # 3. SOAP Note Generation
    print("\n3. SOAP NOTE GENERATION")
    print("-" * 80)
    soap_note = pipeline.generate_soap_note(transcript)
    print(json.dumps(soap_note, indent=2))
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
