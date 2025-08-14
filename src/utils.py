import re
import spacy
from spacy.matcher import Matcher,PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor



# Load spaCy model
nlp = spacy.load("en_core_web_lg")

skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

skill_extractor = None

def _get_skill_extractor():
    """Initialize skill extractor only when needed"""
    global skill_extractor
    if skill_extractor is None:
        skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return skill_extractor


spacy_matcher = Matcher(nlp.vocab)



def remove_Stopwords(resume_txt, job_txt):
    
    resume_doc =  nlp(resume_txt)
    job_doc = nlp(job_txt)

    filtered_tokens_Resume = [token.text for token in resume_doc if not token.is_stop and not token.is_punct]

    filtered_tokens_job = [token.text for token in job_doc  if not token.is_stop and not token.is_punct]

   ## return(filtered_tokens_Resume, filtered_tokens_Resume)
    return " ".join(filtered_tokens_Resume), " ".join(filtered_tokens_job)
    
def lem_txt(resume_txt, job_txt):

    resume_doc = nlp(resume_txt)
    job_doc = nlp(job_txt)

    lemmatized_resume = " ".join([token.lemma_ for token in resume_doc])
    lemmatized_job = " ".join([token.lemma_ for token in job_doc])
    
    return " ".join(lemmatized_resume), " ".join(lemmatized_job)

def get_candidate_name(resume_txt):

    resume_doc =  nlp(resume_txt)

    # to get name out of the resume content
    ne_pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    spacy_matcher.add('NAME', [ne_pattern])
    name_found = spacy_matcher(resume_doc, as_spans=True)

    return name_found[0]

def get_candidate_mobile_number(resume_txt):
    # Matches numbers with optional +, spaces, dashes, parentheses
    pattern = r'(\+?\d[\d\s\-\(\)]{9,}\d)'
    matches = re.findall(pattern, resume_txt)
    # Filter out numbers with less than 10 digits
    for number in matches:
        digits = re.sub(r'\D', '', number)
        if len(digits) >= 10:
            return number.strip()
    return None

def extract_email(resume_txt):
    candidate_em = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", resume_txt)
    if candidate_em:
        try:
            return candidate_em[0].split()[0].strip(';')
        except IndexError:
            return ""

def get_candidate_skills(text):
    if not text:
        return []
    
    try:
        # Get the skill extractor instance
        extractor = _get_skill_extractor()
        
        # Extract skills using SkillNER
        annotations = extractor.annotate(text)
        
        # Extract skills from both full_matches and ngram_scored
        skills = []
        
        # From full_matches (exact matches)
        if 'results' in annotations and 'full_matches' in annotations['results']:
            for match in annotations['results']['full_matches']:
                skill_name = match['doc_node_value'].lower().strip()
                if len(skill_name) > 1 and skill_name not in skills:
                    skills.append(skill_name)
        
        # From ngram_scored (partial matches with good scores)
        if 'results' in annotations and 'ngram_scored' in annotations['results']:
            for match in annotations['results']['ngram_scored']:
                skill_name = match['doc_node_value'].lower().strip()
                # Only include high-confidence matches
                if match.get('score', 0) >= 0.8 and len(skill_name) > 1 and skill_name not in skills:
                    skills.append(skill_name)
        
        return skills
    
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []

def extract_all_skills(text):
    """Extract all skills from text"""
    return get_candidate_skills(text)
