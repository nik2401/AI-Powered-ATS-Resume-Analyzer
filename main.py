from src.parser import pdf_parser, job_desc
from src.utils import remove_Stopwords, get_candidate_name, get_candidate_mobile_number, extract_email, get_candidate_skills
from src.professional_ats_engine import run_professional_ats_analysis, ATSScore
from datetime import datetime
import os
import sys
import json

class ProfessionalATSReport:
    """Generate comprehensive ATS reports like real systems"""
    
    def __init__(self, candidate_name: str, ats_score: ATSScore):
        self.candidate_name = candidate_name
        self.ats_score = ats_score
        self.timestamp = datetime.now()

    def print_executive_summary(self):
        """Print executive summary like real ATS dashboards"""
        
        print("\n" + "="*70)
        print("🏢 PROFESSIONAL ATS ANALYSIS REPORT")
        print("="*70)
        print(f"📊 Candidate: {self.candidate_name}")
        print(f"📅 Analysis Date: {self.timestamp.strftime('%B %d, %Y at %H:%M')}")
        print(f"🎯 Overall ATS Score: {self.ats_score.overall_score:.1%}")
        print(f"📈 Confidence Level: {self.ats_score.confidence:.1%}")
        
        # Score interpretation
        score = self.ats_score.overall_score
        if score >= 0.80:
            status = "🟢 EXCELLENT - Likely to pass initial ATS screening"
            recommendation = "✅ Ready to apply! Your resume meets most ATS requirements."
        elif score >= 0.65:
            status = "🟡 GOOD - May pass ATS with minor improvements"
            recommendation = "⚡ A few optimizations will significantly boost your ATS score."
        elif score >= 0.45:
            status = "🟠 MODERATE - Needs optimization for ATS success"
            recommendation = "📝 Several improvements needed to pass ATS screening effectively."
        else:
            status = "🔴 NEEDS WORK - Unlikely to pass ATS in current state"
            recommendation = "🔧 Major revisions required for ATS compatibility."
        
        print(f"\n🚦 ATS Status: {status}")
        print(f"💡 Recommendation: {recommendation}")

    def print_detailed_breakdown(self):
        """Print detailed section-by-section analysis"""
        
        print(f"\n" + "="*70)
        print("📊 DETAILED SECTION ANALYSIS")
        print("="*70)
        
        # Section scores with visual indicators
        sections = [
            ("Keywords & Skills", self.ats_score.keyword_score, "🎯"),
            ("Resume Format", self.ats_score.format_score, "📄"),
            ("Experience Match", self.ats_score.experience_score, "💼"),
            ("Education Match", self.ats_score.education_score, "🎓")
        ]
        
        for section_name, score, icon in sections:
            bar = self._generate_progress_bar(score)
            grade = self._score_to_grade(score)
            
            print(f"{icon} {section_name:<20} {bar} {score:.1%} ({grade})")
        
        # Keyword analysis breakdown
        keyword_data = self.ats_score.breakdown.get('keyword_analysis', {})
        if keyword_data:
            print(f"\n🎯 KEYWORD ANALYSIS:")
            print(f"   ✅ Matched Skills: {len(keyword_data.get('matched_skills', []))}")
            print(f"   ❌ Missing Skills: {len(keyword_data.get('missing_skills', []))}")
            print(f"   📈 Match Rate: {keyword_data.get('match_rate', 0):.1%}")
        
        # Format compliance
        format_data = self.ats_score.breakdown.get('format_analysis', {})
        if format_data:
            ats_friendly = "✅ Yes" if format_data.get('ats_friendly', False) else "❌ No"
            print(f"\n📄 FORMAT COMPLIANCE:")
            print(f"   ATS-Friendly: {ats_friendly}")
            print(f"   Compliance Score: {format_data.get('compliance_score', 0):.1%}")

    def print_actionable_recommendations(self):
        """Print specific recommendations"""
        
        print(f"\n" + "="*70)
        print("🚀 ACTIONABLE RECOMMENDATIONS")
        print("="*70)
        
        if not self.ats_score.recommendations:
            print("🎉 Excellent! Your resume is well-optimized for ATS systems.")
            return
        
        print("📋 Priority Actions (ranked by impact):")
        for i, recommendation in enumerate(self.ats_score.recommendations[:8], 1):
            print(f"   {i}. {recommendation}")
        
        # Quick wins section
        print(f"\n⚡ QUICK WINS (15 minutes or less):")
        
        keyword_data = self.ats_score.breakdown.get('keyword_analysis', {})
        missing_skills = keyword_data.get('missing_skills', [])
        
        if missing_skills:
            print(f"   • Add to Skills section: {', '.join(missing_skills[:5])}")
            print(f"   • Update job titles with relevant keywords")
            print(f"   • Include industry-specific terms")
        
        format_data = self.ats_score.breakdown.get('format_analysis', {})
        if not format_data.get('ats_friendly', True):
            print(f"   • Remove graphics, tables, or unusual formatting")
            print(f"   • Use standard section headers (Experience, Skills, Education)")
            print(f"   • Save as .docx or .pdf format")

    def print_keyword_optimization(self):
        """Print detailed keyword optimization guide"""
        
        keyword_data = self.ats_score.breakdown.get('keyword_analysis', {})
        if not keyword_data:
            return
            
        print(f"\n" + "="*70)
        print("🔑 KEYWORD OPTIMIZATION GUIDE")
        print("="*70)
        
        matched_skills = keyword_data.get('matched_skills', [])
        missing_skills = keyword_data.get('missing_skills', [])
        
        if matched_skills:
            print(f"✅ KEYWORDS FOUND ({len(matched_skills)}):")
            for skill in matched_skills[:10]:
                print(f"   • {skill}")
        
        if missing_skills:
            print(f"\n❌ CRITICAL MISSING KEYWORDS ({len(missing_skills)}):")
            
            # Categorize missing skills
            technical_skills = []
            soft_skills = []
            tools = []
            
            for skill in missing_skills:
                skill_lower = skill.lower()
                if any(tech in skill_lower for tech in ['python', 'java', 'sql', 'aws', 'react']):
                    technical_skills.append(skill)
                elif any(soft in skill_lower for soft in ['communication', 'leadership', 'teamwork']):
                    soft_skills.append(skill)
                else:
                    tools.append(skill)
            
            if technical_skills:
                print(f"   🔧 Technical: {', '.join(technical_skills[:5])}")
            
            if tools:
                print(f"   🛠️ Tools/Software: {', '.join(tools[:5])}")
            
            if soft_skills:
                print(f"   🤝 Soft Skills: {', '.join(soft_skills[:3])}")
            
            print(f"\n💡 WHERE TO ADD THESE KEYWORDS:")
            print(f"   1. Skills section (exact match)")
            print(f"   2. Job descriptions (naturally integrated)")
            print(f"   3. Summary/Objective (if relevant)")
            print(f"   4. Project descriptions")

    def save_detailed_report(self, filename: str = None):
        """Save comprehensive ATS report to file"""
        
        if not filename:
            safe_name = self.candidate_name.replace(' ', '_').replace(',', '')
            filename = f"ATS_Report_{safe_name}_{self.timestamp.strftime('%Y%m%d_%H%M')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("PROFESSIONAL ATS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Executive Summary
            f.write(f"Candidate: {self.candidate_name}\n")
            f.write(f"Analysis Date: {self.timestamp.strftime('%B %d, %Y at %H:%M')}\n")
            f.write(f"Overall ATS Score: {self.ats_score.overall_score:.1%}\n")
            f.write(f"Confidence Level: {self.ats_score.confidence:.1%}\n\n")
            
            # Section Scores
            f.write("SECTION BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Keywords & Skills: {self.ats_score.keyword_score:.1%}\n")
            f.write(f"Resume Format: {self.ats_score.format_score:.1%}\n")
            f.write(f"Experience Match: {self.ats_score.experience_score:.1%}\n")
            f.write(f"Education Match: {self.ats_score.education_score:.1%}\n\n")
            
            # Recommendations
            f.write("ACTIONABLE RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(self.ats_score.recommendations, 1):
                # Clean emoji for text file
                clean_rec = rec.encode('ascii', 'ignore').decode('ascii')
                f.write(f"{i}. {clean_rec}\n")
            
            # Keyword details
            keyword_data = self.ats_score.breakdown.get('keyword_analysis', {})
            if keyword_data:
                f.write(f"\nKEYWORD ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                
                matched = keyword_data.get('matched_skills', [])
                missing = keyword_data.get('missing_skills', [])
                
                f.write(f"Matched Skills ({len(matched)}):\n")
                for skill in matched:
                    f.write(f"  - {skill}\n")
                
                f.write(f"\nMissing Skills ({len(missing)}):\n")
                for skill in missing:
                    f.write(f"  - {skill}\n")
        
        return filename

    def _generate_progress_bar(self, score: float, width: int = 20) -> str:
        """Generate visual progress bar"""
        filled = int(score * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "D"

def get_resume_file():
    """Get resume file from user input"""
    print("\n" + "="*60)
    print("📄 PROFESSIONAL ATS RESUME ANALYZER")
    print("="*60)
    print("Upload your resume for comprehensive ATS analysis")
    
    # List PDF files in current directory
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if pdf_files:
        print(f"\n📁 Found PDF files in current directory:")
        for i, file in enumerate(pdf_files, 1):
            print(f"  {i}. {file}")
        
        print(f"  {len(pdf_files) + 1}. Enter custom file path")
        
        while True:
            try:
                choice = input(f"\nSelect resume file (1-{len(pdf_files) + 1}): ").strip()
                
                if choice == str(len(pdf_files) + 1):
                    file_path = input("Enter full path to your PDF resume: ").strip()
                    if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
                        return file_path
                    else:
                        print("❌ File not found or not a PDF. Please try again.")
                        continue
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(pdf_files):
                    return pdf_files[choice_num - 1]
                else:
                    print(f"❌ Please enter a number between 1 and {len(pdf_files) + 1}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
    else:
        print("📂 No PDF files found in current directory.")
        file_path = input("Enter full path to your PDF resume: ").strip()
        
        if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
            return file_path
        else:
            print("❌ File not found or not a PDF. Exiting.")
            return None

def main():
    """Main application flow"""
    
    print("🚀 Welcome to Professional ATS Analyzer")
    print("   Analyze your resume like Fortune 500 companies do!")
    
    # Get resume file
    pdf_file = get_resume_file()
    if not pdf_file:
        print("❌ No valid resume file provided. Exiting.")
        sys.exit(1)
    
    print(f"✅ Selected: {pdf_file}")
    
    try:
        # Extract resume text
        print("\n🔍 Extracting resume content...")
        resume_text = pdf_parser(pdf_file)
        
        if not resume_text or len(resume_text.strip()) < 100:
            print("❌ Could not extract sufficient text from PDF. Please check the file.")
            sys.exit(1)
        
        # Extract candidate details
        candidate_name = get_candidate_name(resume_text)
        candidate_phone = get_candidate_mobile_number(resume_text)
        candidate_email = extract_email(resume_text)
        
        print(f"👤 Analyzing resume for: {candidate_name}")
        
    except Exception as e:
        print(f"❌ Error processing resume: {e}")
        sys.exit(1)
    
    # Get job description
    print(f"\n" + "="*60)
    print("📝 JOB DESCRIPTION INPUT")
    print("="*60)
    print("Paste the complete job description you're targeting:")
    
    job_description = job_desc()
    
    if not job_description or len(job_description.strip()) < 50:
        print("❌ Job description too short. Please provide a detailed job posting.")
        sys.exit(1)
    
    # Start professional analysis
    print(f"\n🔄 Initializing Professional ATS Analysis...")
    print("   This may take 30-60 seconds for comprehensive analysis...")
    
    try:
        # Extract skills
        clean_resume, clean_jd = remove_Stopwords(resume_text, job_description)
        resume_skills = get_candidate_skills(clean_resume)
        jd_skills = get_candidate_skills(clean_jd)
        
        print(f"📊 Skills extracted - Resume: {len(resume_skills)}, Job: {len(jd_skills)}")
        
        # Run professional ATS analysis
        ats_result = run_professional_ats_analysis(
            resume_text=resume_text,
            job_description=job_description,
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
        
        # Generate comprehensive report
        report = ProfessionalATSReport(str(candidate_name), ats_result)
        
        # Display results
        report.print_executive_summary()
        
        # Ask for detailed analysis
        show_details = input(f"\nShow detailed section-by-section analysis? (y/n): ").strip().lower()
        if show_details in ['y', 'yes']:
            report.print_detailed_breakdown()
            report.print_actionable_recommendations()
            report.print_keyword_optimization()
        
        # Save report option
        save_report = input(f"\nSave comprehensive ATS report to file? (y/n): ").strip().lower()
        if save_report in ['y', 'yes']:
            try:
                filename = report.save_detailed_report()
                print(f"✅ Report saved as: {filename}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")
        
        # Final summary
        print(f"\n" + "="*70)
        print("🎯 ANALYSIS COMPLETE")
        print("="*70)
        
        score = ats_result.overall_score
        if score >= 0.8:
            print("🟢 RESULT: Your resume is ATS-optimized and ready for applications!")
            print("💡 TIP: Tailor keywords for each specific job application.")
        elif score >= 0.65:
            print("🟡 RESULT: Good foundation! A few improvements will boost your ATS score.")
            print("💡 TIP: Focus on the priority recommendations above.")
        elif score >= 0.45:
            print("🟠 RESULT: Moderate ATS compatibility. Optimization needed for better results.")
            print("💡 TIP: Address format and keyword gaps systematically.")
        else:
            print("🔴 RESULT: Major ATS optimization needed before applying.")
            print("💡 TIP: Start with format compliance, then add missing keywords.")
        
        print(f"\n🚀 Best of luck with your job applications!")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        print("Please check your input files and try again.")
        sys.exit(1)
    
    # Wait before closing
    input(f"\nPress Enter to exit...")

if __name__ == "__main__":
    main()