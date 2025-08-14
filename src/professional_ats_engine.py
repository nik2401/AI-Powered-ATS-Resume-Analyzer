# professional_ats_engine.py - Fixed Real ATS System Implementation

import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import pandas as pd
from datetime import datetime
import json

# Load models with error handling
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")  # Fallback

try:
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
except:
    embedding_model = None  # Will skip semantic analysis if unavailable

@dataclass
class ATSScore:
    overall_score: float
    keyword_score: float
    format_score: float
    experience_score: float
    education_score: float
    confidence: float
    breakdown: Dict
    recommendations: List[str]

class ProfessionalATSEngine:
    def __init__(self):
        """Initialize ATS engine with industry-standard components"""
        self.section_weights = {
            'keywords': 0.40,      # Increased - most critical
            'format': 0.15,        # Reduced - less critical than keywords
            'experience': 0.25,
            'education': 0.15,
            'soft_skills': 0.05
        }
        
        # Comprehensive skill databases (actual skills, not random phrases)
        self.valid_technical_skills = {
            # Programming & Development
            'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'laravel',
            'html', 'css', 'sass', 'typescript', 'jquery', 'bootstrap',
            
            # Data & Analytics
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
            'tableau', 'power bi', 'excel', 'r', 'stata', 'spss', 'matplotlib', 'seaborn',
            'hadoop', 'spark', 'kafka', 'airflow', 'databricks',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
            'ci/cd', 'terraform', 'ansible', 'linux', 'bash', 'powershell',
            
            # Other Technical
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum', 'kanban',
            'git', 'jira', 'confluence', 'slack', 'teams'
        }
        
        self.valid_soft_skills = {
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical thinking',
            'project management', 'time management', 'adaptability', 'creativity', 'collaboration',
            'presentation', 'negotiation', 'mentoring', 'strategic thinking', 'decision making'
        }
        
        # SKILL RELATIONSHIP MAPPING - NEW!
        self.skill_relationships = {
            'computer science': [
                'software development', 'software engineer', 'programming', 'coding',
                'python', 'java', 'javascript', 'algorithms', 'data structures',
                'software design', 'system design', 'backend development', 'frontend development',
                'web development', 'mobile development', 'computer engineering'
            ],
            'data science': [
                'data analysis', 'machine learning', 'statistics', 'python', 'r',
                'data visualization', 'pandas', 'numpy', 'scikit-learn', 'tensorflow',
                'statistical analysis', 'predictive modeling', 'data mining'
            ],
            'cloud computing': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'cloud services',
                'cloud infrastructure', 'devops', 'ci/cd', 'containerization'
            ],
            'analytical thinking': [
                'data analysis', 'problem solving', 'statistical analysis', 'research',
                'critical thinking', 'data-driven decision making', 'quantitative analysis'
            ],
            'software engineering': [
                'software development', 'programming', 'coding', 'system design',
                'software architecture', 'agile', 'scrum', 'version control', 'git'
            ]
        }
        
        # Industry-specific skill mapping
        self.industry_skills = {
            'technology': {
                'core': ['programming', 'software development', 'coding', 'debugging', 'testing'],
                'tools': ['python', 'java', 'react', 'aws', 'docker', 'git', 'sql'],
                'concepts': ['agile', 'scrum', 'ci/cd', 'microservices', 'api', 'cloud computing']
            },
            'data_science': {
                'core': ['data analysis', 'machine learning', 'statistics', 'data visualization'],
                'tools': ['python', 'r', 'sql', 'tableau', 'pandas', 'scikit-learn', 'tensorflow'],
                'concepts': ['statistical modeling', 'predictive analytics', 'big data', 'etl']
            },
            'finance': {
                'core': ['financial analysis', 'risk management', 'financial modeling', 'accounting'],
                'tools': ['excel', 'bloomberg', 'sql', 'python', 'tableau', 'sap'],
                'concepts': ['portfolio management', 'derivatives', 'compliance', 'regulations']
            },
            'marketing': {
                'core': ['digital marketing', 'campaign management', 'brand management'],
                'tools': ['google analytics', 'hubspot', 'salesforce', 'adobe creative suite'],
                'concepts': ['seo', 'sem', 'social media marketing', 'content strategy']
            },
            'healthcare': {
                'core': ['patient care', 'clinical skills', 'medical knowledge'],
                'tools': ['emr systems', 'epic', 'cerner', 'medical devices'],
                'concepts': ['hipaa compliance', 'clinical protocols', 'patient safety']
            },
            'general': {
                'core': ['communication', 'teamwork', 'problem solving'],
                'tools': ['microsoft office', 'email', 'phone'],
                'concepts': ['customer service', 'time management', 'organization']
            }
        }

    def extract_skills_from_text(self, text: str, is_job_description: bool = False) -> List[str]:
        """Extract actual skills from text using intelligent parsing"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        # 1. Direct technical skill matching (exact matches)
        for skill in self.valid_technical_skills:
            if skill in text_lower:
                # Avoid partial word matches (e.g., "java" in "javascript")
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found_skills.append(skill)
        
        # 2. Extract programming languages and frameworks with variations
        tech_patterns = {
            'python': r'\b(python|py)\b',
            'javascript': r'\b(javascript|js|node\.?js)\b',
            'java': r'\b(java)\b(?!script)',  # Java but not JavaScript
            'c++': r'\b(c\+\+|cpp)\b',
            'c#': r'\b(c#|c-sharp|csharp)\b',
            'sql': r'\b(sql|mysql|postgresql|postgres)\b',
            'html': r'\b(html|html5)\b',
            'css': r'\b(css|css3)\b',
            'react': r'\b(react|reactjs|react\.js)\b',
            'angular': r'\b(angular|angularjs)\b',
            'vue': r'\b(vue|vuejs|vue\.js)\b',
            'django': r'\b(django)\b',
            'flask': r'\b(flask)\b',
            'tensorflow': r'\b(tensorflow|tf)\b',
            'pytorch': r'\b(pytorch|torch)\b',
            'scikit-learn': r'\b(scikit-learn|sklearn|sci-kit learn)\b',
            'pandas': r'\b(pandas|pd)\b',
            'numpy': r'\b(numpy|np)\b',
            'matplotlib': r'\b(matplotlib|plt)\b',
            'tableau': r'\b(tableau)\b',
            'power bi': r'\b(power bi|powerbi|pbi)\b',
            'excel': r'\b(excel|microsoft excel)\b',
            'aws': r'\b(aws|amazon web services)\b',
            'azure': r'\b(azure|microsoft azure)\b',
            'docker': r'\b(docker|containerization)\b',
            'kubernetes': r'\b(kubernetes|k8s)\b',
            'git': r'\b(git|github|gitlab)\b',
            'machine learning': r'\b(machine learning|ml|artificial intelligence|ai)\b',
            'data science': r'\b(data science|data scientist)\b',
            'data analysis': r'\b(data analysis|data analytics|analytics)\b',
            'deep learning': r'\b(deep learning|neural networks)\b',
            'computer science': r'\b(computer science|cs|computer engineering)\b',
            'software development': r'\b(software development|software engineer|software developer)\b',
            'web development': r'\b(web development|web developer|frontend|backend|full.?stack)\b',
            'database': r'\b(database|databases|dbms)\b',
            'api': r'\b(api|rest api|restful|graphql)\b',
            'cloud computing': r'\b(cloud|cloud computing|cloud services)\b',
        }
        
        for skill_name, pattern in tech_patterns.items():
            if re.search(pattern, text_lower):
                found_skills.append(skill_name)
        
        # 3. Extract domain expertise and education
        domain_patterns = {
            'data visualization': r'\b(data visualization|visualization|charts|dashboards)\b',
            'statistical analysis': r'\b(statistical analysis|statistics|statistical modeling)\b',
            'time series': r'\b(time series|forecasting|time series analysis)\b',
            'natural language processing': r'\b(nlp|natural language processing|text analysis)\b',
            'computer vision': r'\b(computer vision|cv|image processing)\b',
            'project management': r'\b(project management|scrum|agile|kanban)\b',
            'data mining': r'\b(data mining|data extraction)\b',
            'business intelligence': r'\b(business intelligence|bi)\b',
            'etl': r'\b(etl|extract transform load|data pipeline)\b',
            'computer science': r'\b(computer science|cs degree|computer engineering|software engineering degree|b\.?s\.? computer science|bachelor.*computer science|master.*computer science)\b',
            'analytical thinking': r'\b(analytical thinking|analytical skills|critical thinking|problem solving|analytical mindset)\b',
            'communication': r'\b(communication skills|written communication|verbal communication|presentation skills|interpersonal skills)\b',
            'cloud computing': r'\b(cloud computing|cloud services|cloud infrastructure|cloud platforms)\b',
        }
        
        for skill_name, pattern in domain_patterns.items():
            if re.search(pattern, text_lower):
                found_skills.append(skill_name)
        
        # 4. Extract soft skills (only clear mentions)
        for soft_skill in self.valid_soft_skills:
            pattern = r'\b' + re.escape(soft_skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(soft_skill)
        
        # 5. Remove duplicates and return
        return list(set(found_skills))

    def clean_and_validate_skills(self, raw_skills: List[str], context_text: str = "") -> List[str]:
        """Clean legacy extracted skills and add smart extraction"""
        
        # First, use smart extraction from context
        smart_extracted = self.extract_skills_from_text(context_text) if context_text else []
        
        # Then, clean the raw skills list
        validated_skills = []
        
        if raw_skills:
            for skill in raw_skills:
                if not skill or not isinstance(skill, str):
                    continue
                    
                skill_clean = skill.lower().strip()
                
                # Skip if empty or too short/long
                if len(skill_clean) < 2 or len(skill_clean) > 50:
                    continue
                
                # Expanded non-skills list
                non_skills = {
                    'experience', 'years', 'work', 'company', 'team', 'project', 'role', 'position',
                    'responsible', 'manage', 'develop', 'create', 'implement', 'support', 'ensure',
                    'nursing home', 'employee benefit', 'special education', 'high school',
                    'united states', 'new york', 'los angeles', 'san francisco', 'chicago',
                    'perspective', 'collaborative', 'software developers', 'time', 'data',
                    'analysis', 'working', 'using', 'experience working', 'strong', 'good',
                    'excellent', 'proficient', 'familiar', 'knowledge', 'understanding',
                    'ability', 'skills', 'including', 'various', 'multiple', 'several'
                }
                
                if skill_clean in non_skills:
                    continue
                
                # Skip common words
                common_words = {
                    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'
                }
                
                if skill_clean in common_words:
                    continue
                
                # Skip sentence fragments
                if len(skill_clean.split()) > 3:
                    continue
                
                # Only keep if it's a recognized skill
                is_valid = (
                    skill_clean in self.valid_technical_skills or
                    skill_clean in self.valid_soft_skills or
                    any(re.search(r'\b' + re.escape(tech_skill) + r'\b', skill_clean) 
                        for tech_skill in self.valid_technical_skills) or
                    any(re.search(r'\b' + re.escape(soft_skill) + r'\b', skill_clean) 
                        for soft_skill in self.valid_soft_skills)
                )
                
                if is_valid:
                    validated_skills.append(skill_clean)
        
        # Combine smart extraction with cleaned raw skills
        all_skills = list(set(smart_extracted + validated_skills))
        return all_skills

    def analyze_resume(self, resume_text: str, job_description: str, 
                      resume_skills: List[str], jd_skills: List[str]) -> ATSScore:
        """Complete ATS analysis with proper validation"""
        
        print("🔍 ATS ANALYSIS IN PROGRESS...")
        print("="*50)
        
        # 1. Smart skill extraction (prioritize intelligent extraction over raw input)
        print("🧠 Extracting skills intelligently...")
        smart_resume_skills = self.extract_skills_from_text(resume_text)
        smart_jd_skills = self.extract_skills_from_text(job_description, is_job_description=True)
        
        # 2. Clean legacy skills and combine with smart extraction
        clean_resume_skills = self.clean_and_validate_skills(resume_skills, resume_text)
        clean_jd_skills = self.clean_and_validate_skills(jd_skills, job_description)
        
        # 3. Merge smart and cleaned skills (prioritize smart extraction)
        final_resume_skills = list(set(smart_resume_skills + clean_resume_skills))
        final_jd_skills = list(set(smart_jd_skills + clean_jd_skills))
        
        print(f"🧹 Smart Extraction - Resume: {len(smart_resume_skills)}, Job: {len(smart_jd_skills)}")
        print(f"🔧 Total Skills - Resume: {len(final_resume_skills)}, Job: {len(final_jd_skills)}")
        
        if final_resume_skills:
            print(f"📝 Resume Skills Sample: {', '.join(final_resume_skills[:8])}")
        if final_jd_skills:
            print(f"🎯 Job Skills Sample: {', '.join(final_jd_skills[:8])}")
        
        # 4. Parse sections
        resume_sections = self._parse_resume_sections(resume_text)
        jd_sections = self._parse_job_sections(job_description)
        
        # 5. Detect industry and job level
        industry = self._detect_industry_advanced(job_description)
        seniority_level = self._detect_seniority(job_description, resume_text)
        
        print(f"📊 Detected Industry: {industry.title()}")
        print(f"📈 Seniority Level: {seniority_level.title()}")
        
        # 6. Check job-resume relevance (CRITICAL FIX)
        relevance_score = self._calculate_job_relevance(resume_text, job_description, industry)
        print(f"🎯 Job Relevance: {relevance_score:.1%}")
        
        # 7. Multi-stage scoring
        scores = {}
        
        # Apply relevance penalty for mismatched jobs
        relevance_penalty = max(0.1, relevance_score)  # Minimum 10% if completely irrelevant
        
        # Stage 1: Keyword Matching (40% weight) - FIXED
        scores['keywords'] = self._advanced_keyword_analysis(
            final_resume_skills, final_jd_skills, resume_text, job_description, industry
        ) * relevance_penalty
        
        # Stage 2: Format Compliance (15% weight)
        scores['format'] = self._format_compliance_check(resume_text, resume_sections)
        
        # Stage 3: Experience Relevance (25% weight) - FIXED
        scores['experience'] = self._experience_analysis(
            resume_sections.get('experience', ''), jd_sections.get('requirements', ''),
            seniority_level, industry
        ) * relevance_penalty
        
        # Stage 4: Education Match (15% weight)
        scores['education'] = self._education_analysis(
            resume_sections.get('education', ''), jd_sections.get('requirements', '')
        )
        
        # Stage 5: Soft Skills (5% weight)
        scores['soft_skills'] = self._soft_skills_analysis(resume_text, job_description)
        
        # 8. Calculate weighted overall score
        overall_score = sum(scores[key] * self.section_weights[key] for key in scores)
        
        # 9. Apply final relevance check
        if relevance_score < 0.3:  # Very low relevance
            overall_score *= 0.5  # Severe penalty
        
        # 10. Calculate confidence
        confidence = self._calculate_confidence(resume_sections, scores, relevance_score)
        
        # 11. Generate breakdown
        breakdown = self._generate_breakdown(
            scores, final_resume_skills, final_jd_skills, industry, relevance_score
        )
        
        # 12. Generate recommendations
        recommendations = self._generate_recommendations(
            scores, breakdown, seniority_level, relevance_score
        )
        
        return ATSScore(
            overall_score=overall_score,
            keyword_score=scores['keywords'],
            format_score=scores['format'],
            experience_score=scores['experience'],
            education_score=scores['education'],
            confidence=confidence,
            breakdown=breakdown,
            recommendations=recommendations
        )

    def _calculate_job_relevance(self, resume_text: str, job_description: str, industry: str) -> float:
        """Calculate how relevant the resume is to the job - ENHANCED VERSION"""
        
        resume_lower = resume_text.lower()
        jd_lower = job_description.lower()
        
        # Get industry-specific keywords with higher weights for core skills
        if industry in self.industry_skills:
            core_keywords = self.industry_skills[industry]['core']
            tool_keywords = self.industry_skills[industry]['tools'] 
            concept_keywords = self.industry_skills[industry]['concepts']
        else:
            core_keywords = self.industry_skills['general']['core']
            tool_keywords = self.industry_skills['general']['tools']
            concept_keywords = self.industry_skills['general']['concepts']
        
        # Weight different types of matches
        core_matches = sum(2 for keyword in core_keywords if keyword in resume_lower)  # 2x weight
        tool_matches = sum(1.5 for keyword in tool_keywords if keyword in resume_lower)  # 1.5x weight  
        concept_matches = sum(1 for keyword in concept_keywords if keyword in resume_lower)  # 1x weight
        
        # Calculate weighted score
        total_possible = len(core_keywords) * 2 + len(tool_keywords) * 1.5 + len(concept_keywords) * 1
        actual_score = (core_matches + tool_matches + concept_matches) / total_possible if total_possible > 0 else 0.1
        
        # Boost for direct job-resume keyword overlap
        resume_words = set(self._extract_important_words(resume_lower))
        jd_words = set(self._extract_important_words(jd_lower))
        
        # Focus on technical terms and skills
        tech_terms = {'python', 'sql', 'machine', 'learning', 'data', 'analysis', 'science', 
                     'aws', 'azure', 'cloud', 'docker', 'git', 'database', 'programming',
                     'software', 'development', 'engineering', 'analytics', 'statistics'}
        
        resume_tech = resume_words & tech_terms
        jd_tech = jd_words & tech_terms
        
        if jd_tech:
            tech_overlap = len(resume_tech & jd_tech) / len(jd_tech)
            actual_score += tech_overlap * 0.3  # 30% boost for technical overlap
        
        # Industry bonus - if both resume and job are clearly in the same field
        if industry != 'general':
            industry_indicators = {
                'data_science': ['data', 'science', 'analytics', 'machine', 'learning'],
                'technology': ['software', 'development', 'programming', 'technical'],
                'finance': ['financial', 'investment', 'banking', 'trading'],
            }
            
            if industry in industry_indicators:
                indicators = industry_indicators[industry]
                resume_industry_score = sum(1 for ind in indicators if ind in resume_lower)
                jd_industry_score = sum(1 for ind in indicators if ind in jd_lower)
                
                if resume_industry_score >= 2 and jd_industry_score >= 2:
                    actual_score += 0.2  # 20% industry alignment bonus
        
        final_relevance = min(1.0, actual_score)
        return max(0.05, final_relevance)  # Minimum 5%

    def _extract_important_words(self, text: str) -> List[str]:
        """Extract important domain-specific words"""
        # Remove common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in stop_words][:20]  # Top 20 important words

    def _detect_industry_advanced(self, job_description: str) -> str:
        """Advanced industry detection with better accuracy"""
        jd_lower = job_description.lower()
        
        industry_indicators = {
            'data_science': [
                'data scientist', 'machine learning', 'data analysis', 'analytics',
                'python', 'sql', 'statistics', 'modeling', 'algorithms', 'big data'
            ],
            'technology': [
                'software engineer', 'developer', 'programming', 'coding', 'technical',
                'java', 'javascript', 'react', 'api', 'database', 'cloud'
            ],
            'finance': [
                'financial analyst', 'investment', 'banking', 'finance', 'trading',
                'portfolio', 'risk management', 'bloomberg', 'financial modeling'
            ],
            'healthcare': [
                'nurse', 'doctor', 'medical', 'clinical', 'patient', 'healthcare',
                'hospital', 'treatment', 'diagnosis', 'medical records'
            ],
            'marketing': [
                'marketing', 'advertising', 'campaign', 'brand', 'social media',
                'seo', 'digital marketing', 'content', 'promotion'
            ]
        }
        
        # Score each industry
        industry_scores = {}
        for industry, keywords in industry_indicators.items():
            score = sum(2 if keyword in jd_lower else 0 for keyword in keywords)
            industry_scores[industry] = score
        
        # Return highest scoring industry or general
        best_industry = max(industry_scores.items(), key=lambda x: x[1])
        return best_industry[0] if best_industry[1] > 3 else 'general'

    def _advanced_keyword_analysis(self, resume_skills: List[str], jd_skills: List[str], 
                                 resume_text: str, job_description: str, industry: str) -> float:
        """Fixed keyword analysis that actually works"""
        
        if not jd_skills or not resume_skills:
            return 0.0
        
        # 1. Direct skill matches (most important)
        resume_set = set(skill.lower() for skill in resume_skills)
        jd_set = set(skill.lower() for skill in jd_skills)
        
        direct_matches = len(resume_set & jd_set)
        direct_score = direct_matches / len(jd_skills) if jd_skills else 0
        
        # 2. Partial matches (e.g., "python programming" matches "python")
        partial_matches = 0
        for jd_skill in jd_skills:
            jd_skill_lower = jd_skill.lower()
            for resume_skill in resume_skills:
                resume_skill_lower = resume_skill.lower()
                if (jd_skill_lower in resume_skill_lower or 
                    resume_skill_lower in jd_skill_lower or
                    self._skills_are_similar(jd_skill_lower, resume_skill_lower)):
                    partial_matches += 1
                    break
        
        partial_score = partial_matches / len(jd_skills) if jd_skills else 0
        
        # 3. Industry relevance bonus
        industry_bonus = 0
        if industry in self.industry_skills:
            relevant_skills = self.industry_skills[industry]['tools']
            industry_matches = sum(1 for skill in relevant_skills if skill in resume_text.lower())
            industry_bonus = min(0.2, industry_matches / len(relevant_skills) * 0.2)
        
        # Combine scores with weights
        final_score = (
            direct_score * 0.7 +          # 70% weight for exact matches
            (partial_score - direct_score) * 0.2 +  # 20% for additional partial matches
            industry_bonus                  # 10% industry bonus
        )
        
        return min(1.0, final_score)

    def _skills_are_similar(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are conceptually similar"""
        similar_groups = [
            {'python', 'py', 'python programming'},
            {'javascript', 'js', 'node.js', 'nodejs'},
            {'sql', 'mysql', 'postgresql', 'database'},
            {'machine learning', 'ml', 'artificial intelligence', 'ai'},
            {'data analysis', 'analytics', 'data science'},
        ]
        
        for group in similar_groups:
            if skill1 in group and skill2 in group:
                return True
        return False

    def _experience_analysis(self, experience_text: str, job_requirements: str, 
                           seniority_level: str, industry: str) -> float:
        """Enhanced experience analysis with industry context"""
        
        if not experience_text.strip():
            return 0.0
        
        # Extract years of experience
        years_pattern = r'(\d+)[\s\+]*(?:years?|yrs?)'
        years_matches = re.findall(years_pattern, experience_text.lower())
        total_years = max([int(year) for year in years_matches]) if years_matches else 0
        
        # Expected years by seniority
        expected_years = {
            'entry': (0, 3),
            'mid': (2, 6),
            'senior': (5, 12),
            'lead': (7, 15),
            'executive': (10, 25)
        }
        
        min_exp, max_exp = expected_years.get(seniority_level, (0, 5))
        
        # Score based on experience alignment
        if min_exp <= total_years <= max_exp + 2:
            experience_score = 0.9
        elif total_years >= min_exp:
            experience_score = 0.7
        else:
            experience_score = 0.4
        
        # Content relevance bonus
        if job_requirements and industry != 'general':
            content_similarity = self._calculate_text_similarity(experience_text, job_requirements)
            experience_score = min(1.0, experience_score + content_similarity * 0.2)
        
        return experience_score

    # [Rest of the methods remain largely the same, with minor improvements]
    def _parse_resume_sections(self, resume_text: str) -> Dict[str, str]:
        """Parse resume into standard ATS sections"""
        sections = {
            'contact': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'certifications': '',
            'projects': ''
        }
        
        section_patterns = {
            'experience': r'(experience|work history|employment|professional experience)',
            'education': r'(education|academic|qualifications|degrees?)',
            'skills': r'(skills|technical skills|competencies|technologies)',
            'projects': r'(projects|portfolio|key projects)',
            'certifications': r'(certifications?|certificates?|credentials)'
        }
        
        lines = resume_text.split('\n')
        current_section = 'summary'
        
        for line in lines:
            line_lower = line.lower().strip()
            
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line.strip()) < 50:
                    current_section = section
                    break
            else:
                sections[current_section] += line + '\n'
        
        return sections

    def _parse_job_sections(self, job_description: str) -> Dict[str, str]:
        """Parse job description into standard sections"""
        sections = {
            'title': '',
            'summary': '',
            'requirements': '',
            'responsibilities': '',
            'qualifications': '',
            'skills': '',
            'benefits': ''
        }
        
        section_patterns = {
            'requirements': r'(requirements?|qualifications?|must have|required skills)',
            'responsibilities': r'(responsibilities?|duties|role|what you.ll do)',
            'qualifications': r'(qualifications?|education|degree|certification)',
            'skills': r'(skills?|technical requirements|competencies)',
            'benefits': r'(benefits?|compensation|what we offer|perks)'
        }
        
        lines = job_description.split('\n')
        current_section = 'summary'
        
        for line in lines:
            line_lower = line.lower().strip()
            
            section_found = False
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line.strip()) < 100:
                    current_section = section
                    section_found = True
                    break
            
            if not section_found:
                sections[current_section] += line + '\n'
        
        if not any(sections[key].strip() for key in ['requirements', 'responsibilities', 'qualifications']):
            sections['requirements'] = job_description
        
        return sections

    def _format_compliance_check(self, resume_text: str, sections: Dict[str, str]) -> float:
        """Check ATS format compliance"""
        compliance_score = 0
        total_checks = 8
        
        if self._has_contact_info(resume_text):
            compliance_score += 1
        if self._has_proper_sections(sections):
            compliance_score += 1
        if not self._has_problematic_formatting(resume_text):
            compliance_score += 1
        if self._appropriate_length(resume_text):
            compliance_score += 1
        if self._consistent_formatting(resume_text):
            compliance_score += 1
        
        compliance_score += 3  # Additional checks placeholder
        
        return min(1.0, compliance_score / total_checks)

    def _education_analysis(self, education_text: str, job_requirements: str) -> float:
        """Analyze education requirements match"""
        if not education_text.strip():
            return 0.5
        
        degree_patterns = {
            'bachelor': r"(bachelor|b\.?s\.?|b\.?a\.?)",
            'master': r"(master|m\.?s\.?|m\.?a\.?|mba)",
            'phd': r"(ph\.?d\.?|doctorate)",
            'associate': r"(associate|a\.?s\.?)"
        }
        
        education_lower = education_text.lower()
        found_degrees = []
        
        for degree, pattern in degree_patterns.items():
            if re.search(pattern, education_lower):
                found_degrees.append(degree)
        
        base_score = 0.7 if found_degrees else 0.3
        
        requirements_lower = job_requirements.lower() if job_requirements else ""
        if "bachelor" in requirements_lower or "degree" in requirements_lower:
            if any(d in ['bachelor', 'master', 'phd'] for d in found_degrees):
                return min(1.0, base_score + 0.2)
        
        return base_score

    def _soft_skills_analysis(self, resume_text: str, job_description: str) -> float:
        """Analyze soft skills alignment"""
        soft_skills_keywords = list(self.valid_soft_skills)
        
        resume_lower = resume_text.lower()
        jd_lower = job_description.lower()
        
        resume_soft_skills = [skill for skill in soft_skills_keywords if skill in resume_lower]
        jd_soft_skills = [skill for skill in soft_skills_keywords if skill in jd_lower]
        
        if not jd_soft_skills:
            return 0.7
        
        matches = len(set(resume_soft_skills) & set(jd_soft_skills))
        return min(1.0, matches / len(jd_soft_skills))

    # Helper methods
    def _has_contact_info(self, resume_text: str) -> bool:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d[\d\s\-\(\)]{9,}\d)'
        return bool(re.search(email_pattern, resume_text)) or bool(re.search(phone_pattern, resume_text))
    
    def _has_proper_sections(self, sections: Dict[str, str]) -> bool:
        required_sections = ['experience', 'skills']
        return all(sections.get(section, '').strip() for section in required_sections)
    
    def _has_problematic_formatting(self, resume_text: str) -> bool:
        problematic_patterns = [r'<img\s', r'<table\s', r'│', r'┌', r'└']
        return any(re.search(pattern, resume_text) for pattern in problematic_patterns)
    
    def _appropriate_length(self, resume_text: str) -> bool:
        word_count = len(resume_text.split())
        return 300 <= word_count <= 1200
    
    def _consistent_formatting(self, resume_text: str) -> bool:
        lines = resume_text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        return len(non_empty_lines) > 10

    def _detect_seniority(self, job_description: str, resume_text: str) -> str:
        combined_text = (job_description + " " + resume_text).lower()
        
        seniority_indicators = {
            'entry': ['entry level', 'junior', 'associate', 'new grad', '0-2 years'],
            'mid': ['mid level', 'experienced', '3-5 years', 'intermediate'],
            'senior': ['senior', 'lead', '5+ years', 'expert'],
            'executive': ['director', 'vp', 'chief', 'head of', 'manager']
        }
        
        for level, indicators in seniority_indicators.items():
            if any(indicator in combined_text for indicator in indicators):
                return level
        
        return 'mid'

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            vectors = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return max(0, similarity)
        except:
            return 0.0

    def _calculate_confidence(self, sections: Dict[str, str], scores: Dict, relevance: float) -> float:
        """Calculate confidence with relevance factor"""
        confidence_factors = []
        
        complete_sections = sum(1 for section in sections.values() if section.strip())
        completeness = complete_sections / len(sections)
        confidence_factors.append(completeness)
        
        score_values = list(scores.values())
        if score_values:
            score_variance = np.var(score_values)
            consistency = max(0, 1 - score_variance)
            confidence_factors.append(consistency)
        
        total_text_length = sum(len(text) for text in sections.values())
        text_quality = min(1.0, total_text_length / 2000)
        confidence_factors.append(text_quality)
        
        # Add relevance as confidence factor
        confidence_factors.append(relevance)
        
        return np.mean(confidence_factors)

    def _generate_breakdown(self, scores: Dict, resume_skills: List[str], 
                          jd_skills: List[str], industry: str, relevance: float) -> Dict:
        """Generate detailed scoring breakdown"""
        
        matched_skills = list(set([s.lower() for s in resume_skills]) & 
                            set([s.lower() for s in jd_skills]))
        missing_skills = [s for s in jd_skills if s.lower() not in 
                         [r.lower() for r in resume_skills]]
        
        return {
            'keyword_analysis': {
                'matched_skills': matched_skills,
                'missing_skills': missing_skills[:10],
                'match_rate': len(matched_skills) / len(jd_skills) if jd_skills else 0
            },
            'format_analysis': {
                'ats_friendly': scores['format'] > 0.8,
                'compliance_score': scores['format']
            },
            'experience_analysis': {
                'relevance_score': scores['experience'],
                'level_appropriate': scores['experience'] > 0.7
            },
            'education_analysis': {
                'requirements_met': scores['education'] > 0.7,
                'education_score': scores['education']
            },
            'industry_focus': industry,
            'job_relevance': relevance
        }

    def _generate_recommendations(self, scores: Dict, breakdown: Dict, 
                                seniority_level: str, relevance: float) -> List[str]:
        """Generate recommendations with relevance consideration"""
        
        recommendations = []
        
        # Job relevance warnings
        if relevance < 0.3:
            recommendations.append("⚠️ MAJOR ISSUE: Your resume appears to be for a different field entirely")
            recommendations.append("🔄 Consider applying to jobs that better match your background")
            recommendations.append("📝 If this is intentional, heavily customize your resume for this role")
        elif relevance < 0.6:
            recommendations.append("⚠️ NOTICE: Limited relevance between your background and this job")
            recommendations.append("🎯 Focus on transferable skills and relevant experience")
        
        # Keyword recommendations (only if job is somewhat relevant)
        if relevance > 0.3 and scores['keywords'] < 0.6:
            missing_skills = breakdown['keyword_analysis']['missing_skills']
            if missing_skills:
                recommendations.append(
                    f"🎯 PRIORITY: Add these missing keywords: {', '.join(missing_skills[:5])}"
                )
                recommendations.append(
                    f"📝 Integrate naturally: {', '.join(missing_skills[:3])}"
                )
        
        # Format recommendations
        if scores['format'] < 0.8:
            recommendations.append("📄 Fix resume formatting - use standard sections and avoid graphics")
            recommendations.append("✅ Ensure contact information is clearly visible")
        
        # Experience recommendations
        if scores['experience'] < 0.7:
            recommendations.append(f"💼 Highlight experience more relevant to {seniority_level}-level positions")
            recommendations.append("📈 Add quantifiable achievements and impact metrics")
        
        # Education recommendations
        if scores['education'] < 0.6:
            recommendations.append("🎓 Clearly state your educational qualifications")
            recommendations.append("📚 Add relevant certifications or training")
        
        # Positive feedback for good matches
        if scores['keywords'] > 0.7 and scores['format'] > 0.8 and relevance > 0.7:
            recommendations.append("🚀 Excellent match! Your resume is well-optimized for this role")
        elif relevance > 0.8:
            recommendations.append("✅ Strong background match - focus on keyword optimization")
        
        # Default if no recommendations
        if not recommendations:
            recommendations.append("📋 Review job requirements and tailor resume accordingly")
        
        return recommendations


# Usage function
def run_professional_ats_analysis(resume_text: str, job_description: str,
                                resume_skills: List[str], jd_skills: List[str]) -> ATSScore:
    """
    Run complete ATS analysis with proper validation and relevance checking
    """
    ats_engine = ProfessionalATSEngine()
    return ats_engine.analyze_resume(resume_text, job_description, resume_skills, jd_skills)