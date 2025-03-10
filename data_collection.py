import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

def collect_ai_ethics_papers(sources):
    """
    Collect AI ethics papers and resources from academic sources
    """
    collected_data = []
    
    for source in sources:
        print(f"Collecting data from {source}")
        try:
            response = requests.get(source)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This is a simplified example - you'll need to adapt this to each source
            if "arxiv.org" in source:
                papers = extract_arxiv_papers(soup)
            elif "deon.drivendata.org" in source:
                papers = extract_deon_principles(soup)
            elif "partnershiponai.org" in source:
                papers = extract_pai_resources(soup)
            else:
                papers = []
                
            collected_data.extend(papers)
        except Exception as e:
            print(f"Error collecting from {source}: {e}")
    
    # If no data was collected, use some sample data for testing
    if not collected_data:
        collected_data = generate_sample_data()
        
    return pd.DataFrame(collected_data)

def extract_arxiv_papers(soup):
    """Extract papers from arXiv"""
    papers = []
    # Find paper titles and abstracts
    entries = soup.find_all('div', class_='list-title')
    
    for entry in entries:
        title_elem = entry.find('a')
        if title_elem and 'ethics' in title_elem.text.lower():
            abstract_elem = entry.find_next('p', class_='mathjax')
            if abstract_elem:
                papers.append({
                    'question': f"Can you explain the key points of '{title_elem.text.strip()}'?",
                    'explanation': abstract_elem.text.strip()
                })
    
    return papers

def extract_deon_principles(soup):
    """Extract ethical principles from Deon"""
    principles = []
    items = soup.find_all('div', class_='checklist-item')
    
    for item in items:
        title = item.find('h3')
        description = item.find('p')
        if title and description:
            principles.append({
                'question': f"What is the ethical principle of {title.text.strip()}?",
                'explanation': description.text.strip()
            })
    
    return principles

def extract_pai_resources(soup):
    """Extract resources from Partnership on AI"""
    resources = []
    items = soup.find_all('div', class_='resource-item')
    
    for item in items:
        title = item.find('h3')
        description = item.find('div', class_='description')
        if title and description:
            resources.append({
                'question': f"What does the Partnership on AI say about {title.text.strip()}?",
                'explanation': description.text.strip()
            })
    
    return resources

def generate_sample_data():
    """Generate sample data for testing when scraping fails"""
    return [
        {
            'question': 'What is algorithmic bias?',
            'explanation': 'Algorithmic bias occurs when an algorithm produces results that are systematically prejudiced due to erroneous assumptions in the machine learning process. These biases can reflect existing social inequalities or introduce new ones through the data collection, algorithm design, or implementation phases.'
        },
        {
            'question': 'What are the key principles of ethical AI?',
            'explanation': 'The key principles of ethical AI include fairness, transparency, privacy, human autonomy, and accountability. Fairness ensures AI systems treat all people equitably. Transparency means AI systems are explainable and understandable. Privacy protects personal data. Human autonomy ensures humans maintain control over AI systems. Accountability establishes clear responsibility for AI outcomes.'
        },
        {
            'question': 'How can we ensure AI systems are fair?',
            'explanation': 'Ensuring AI fairness involves diverse training data, regular bias audits, clear documentation of design choices, inclusive development teams, and ongoing monitoring after deployment. Different fairness metrics (demographic parity, equal opportunity, etc.) should be considered based on the specific context and application.'
        }
    ]

def prepare_training_data(data, output_file="ai_ethics_dataset.jsonl"):
    """
    Convert collected data into training format
    """
    with open(output_file, 'w') as f:
        for _, row in data.iterrows():
            # Format as instruction-response pairs
            entry = {
                "instruction": row['question'],
                "response": row['explanation']
            }
            f.write(json.dumps(entry) + '\n')

# Example sources
sources = [
    "https://arxiv.org/list/cs.AI/recent",
    "https://deon.drivendata.org/",
    "https://www.partnershiponai.org/resources/"
]

data = collect_ai_ethics_papers(sources)
prepare_training_data(data) 