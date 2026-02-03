#!/usr/bin/env python3
"""
Command-line utility for Medical NLP Pipeline
Usage: python cli_tool.py [command] [options]
"""

import argparse
import json
import sys
from pathlib import Path
from medical_nlp_pipeline import MedicalNLPPipeline


def read_transcript_file(filepath: str) -> str:
    """Read transcript from file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def save_output(data: dict, output_file: str, format: str = 'json'):
    """Save output to file"""
    try:
        if format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(str(data))
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error saving output: {e}")
        sys.exit(1)


def command_summarize(args):
    """Summarize medical transcript"""
    print("Loading NLP pipeline...")
    pipeline = MedicalNLPPipeline()
    
    # Read transcript
    if args.input:
        transcript = read_transcript_file(args.input)
    elif args.text:
        transcript = args.text
    else:
        print("Error: Either --input or --text must be provided")
        sys.exit(1)
    
    print("Generating medical summary...")
    summary = pipeline.summarize_medical_report(transcript)
    
    # Output
    if args.output:
        save_output(summary, args.output, 'json')
    else:
        print("\nMedical Summary:")
        print(json.dumps(summary, indent=2))


def command_sentiment(args):
    """Analyze sentiment of patient dialogue"""
    print("Loading NLP pipeline...")
    pipeline = MedicalNLPPipeline()
    
    # Get dialogue
    if args.input:
        dialogue = read_transcript_file(args.input)
    elif args.text:
        dialogue = args.text
    else:
        print("Error: Either --input or --text must be provided")
        sys.exit(1)
    
    print("Analyzing sentiment...")
    result = pipeline.analyze_sentiment(dialogue)
    
    # Output
    if args.output:
        save_output(result, args.output, 'json')
    else:
        print("\nSentiment Analysis:")
        print(json.dumps(result, indent=2))


def command_soap(args):
    """Generate SOAP note"""
    print("Loading NLP pipeline...")
    pipeline = MedicalNLPPipeline()
    
    # Read transcript
    if args.input:
        transcript = read_transcript_file(args.input)
    elif args.text:
        transcript = args.text
    else:
        print("Error: Either --input or --text must be provided")
        sys.exit(1)
    
    print("Generating SOAP note...")
    soap_note = pipeline.generate_soap_note(transcript)
    
    # Output
    if args.output:
        save_output(soap_note, args.output, 'json')
    else:
        print("\nSOAP Note:")
        print(json.dumps(soap_note, indent=2))


def command_full(args):
    """Run complete pipeline"""
    print("Loading NLP pipeline...")
    pipeline = MedicalNLPPipeline()
    
    # Read transcript
    if args.input:
        transcript = read_transcript_file(args.input)
    elif args.text:
        transcript = args.text
    else:
        print("Error: Either --input or --text must be provided")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Running Complete Medical NLP Pipeline")
    print("="*80)
    
    # 1. Medical Summary
    print("\n1. Generating Medical Summary...")
    summary = pipeline.summarize_medical_report(transcript)
    print("✓ Medical summary generated")
    
    # 2. Extract patient dialogues for sentiment analysis
    print("\n2. Analyzing Patient Sentiment...")
    patient_dialogues = []
    for line in transcript.split('\n'):
        if 'Patient:' in line:
            dialogue = line.split('Patient:')[1].strip()
            if dialogue:
                patient_dialogues.append(dialogue)
    
    sentiments = []
    for dialogue in patient_dialogues[:5]:  # Analyze first 5
        sentiment = pipeline.analyze_sentiment(dialogue)
        sentiments.append(sentiment)
    print(f"✓ Analyzed {len(sentiments)} patient dialogues")
    
    # 3. SOAP Note
    print("\n3. Generating SOAP Note...")
    soap_note = pipeline.generate_soap_note(transcript)
    print("✓ SOAP note generated")
    
    # Combine results
    results = {
        "medical_summary": summary,
        "sentiment_analysis": sentiments,
        "soap_note": soap_note
    }
    
    # Output
    if args.output:
        save_output(results, args.output, 'json')
    else:
        print("\n" + "="*80)
        print("COMPLETE RESULTS")
        print("="*80)
        print(json.dumps(results, indent=2))


def command_batch(args):
    """Process multiple transcripts"""
    print("Loading NLP pipeline...")
    pipeline = MedicalNLPPipeline()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all text files
    transcript_files = list(input_dir.glob('*.txt'))
    
    if not transcript_files:
        print(f"No .txt files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(transcript_files)} transcript files")
    
    for i, file_path in enumerate(transcript_files, 1):
        print(f"\nProcessing {i}/{len(transcript_files)}: {file_path.name}")
        
        # Read transcript
        transcript = read_transcript_file(str(file_path))
        
        # Process
        summary = pipeline.summarize_medical_report(transcript)
        soap_note = pipeline.generate_soap_note(transcript)
        
        # Save outputs
        output_base = output_dir / file_path.stem
        save_output(summary, f"{output_base}_summary.json", 'json')
        save_output(soap_note, f"{output_base}_soap.json", 'json')
    
    print(f"\n✓ Processed all {len(transcript_files)} files")
    print(f"Results saved to: {output_dir}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Medical NLP Pipeline - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize a transcript
  python cli_tool.py summarize --input transcript.txt --output summary.json
  
  # Analyze sentiment
  python cli_tool.py sentiment --text "I'm worried about my recovery"
  
  # Generate SOAP note
  python cli_tool.py soap --input transcript.txt --output soap_note.json
  
  # Run full pipeline
  python cli_tool.py full --input transcript.txt --output results.json
  
  # Batch process multiple files
  python cli_tool.py batch --input-dir ./transcripts --output-dir ./results
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Summarize command
    parser_summarize = subparsers.add_parser('summarize', help='Generate medical summary')
    parser_summarize.add_argument('--input', '-i', help='Input transcript file')
    parser_summarize.add_argument('--text', '-t', help='Direct text input')
    parser_summarize.add_argument('--output', '-o', help='Output file (JSON)')
    
    # Sentiment command
    parser_sentiment = subparsers.add_parser('sentiment', help='Analyze patient sentiment')
    parser_sentiment.add_argument('--input', '-i', help='Input text file')
    parser_sentiment.add_argument('--text', '-t', help='Direct text input')
    parser_sentiment.add_argument('--output', '-o', help='Output file (JSON)')
    
    # SOAP command
    parser_soap = subparsers.add_parser('soap', help='Generate SOAP note')
    parser_soap.add_argument('--input', '-i', help='Input transcript file')
    parser_soap.add_argument('--text', '-t', help='Direct text input')
    parser_soap.add_argument('--output', '-o', help='Output file (JSON)')
    
    # Full pipeline command
    parser_full = subparsers.add_parser('full', help='Run complete pipeline')
    parser_full.add_argument('--input', '-i', help='Input transcript file')
    parser_full.add_argument('--text', '-t', help='Direct text input')
    parser_full.add_argument('--output', '-o', help='Output file (JSON)')
    
    # Batch command
    parser_batch = subparsers.add_parser('batch', help='Process multiple transcripts')
    parser_batch.add_argument('--input-dir', '-i', required=True, help='Input directory with transcripts')
    parser_batch.add_argument('--output-dir', '-o', required=True, help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'summarize':
            command_summarize(args)
        elif args.command == 'sentiment':
            command_sentiment(args)
        elif args.command == 'soap':
            command_soap(args)
        elif args.command == 'full':
            command_full(args)
        elif args.command == 'batch':
            command_batch(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
