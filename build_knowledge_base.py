import os
import sys
import re
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
from html.entities import name2codepoint

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: BeautifulSoup not available, will use basic HTML parsing")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available, Gemini API calls disabled")


TOP_N = 5
NUM_COUPLING_TOPOLOGIES = 3
ENERGY_REDUCTION_THRESHOLD = 1e-4
MAX_CONTENT_LENGTH = None
AUTO_DISCOVER_DEPTH = 1
MAX_DISCOVERED_URLS = 20

API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL = "YOUR_MODEL_NAME"

KNOWLEDGE_BASE_DIR = "knowledge_base3"
KNOWLEDGE_BASE_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "pennylane_vqe_knowledge_base.txt")
KEYWORDS_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "keywords_list.txt")
METADATA_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "scraping_metadata.json")
CIRCUIT_GUIDANCE_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "circuit_design_guidance.txt")

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 2
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

DEFAULT_KEYWORDS = [
    "VQE", "Variational Quantum Eigensolver", "variational eigensolver",
    "Heisenberg model", "Heisenberg Hamiltonian", "Heisenberg",
    "quantum Hamiltonian", "Hamiltonian simulation",

    "quantum circuit", "parametrized circuit", "ansatz",
    "variational form", "trial state", "trial wavefunction",

    "quantum gate", "CNOT", "CZ", "RX", "RY", "RZ", "Hadamard",
    "entangling gate", "two-qubit gate", "single-qubit gate",

    "classical optimizer", "gradient descent", "Adam optimizer",
    "parameter optimization", "energy minimization", "cost function",

    "molecular Hamiltonian", "electronic structure",
    "ground state energy", "Hartree-Fock", "FCI",
    "Jordan-Wigner", "fermionic Hamiltonian",

    "expectation value", "measurement", "quantum gradient",
    " PennyLane", "quantum machine learning", "VQE applications"
]

SEED_URLS = [
    "https://pennylane.ai/qml/demos/tutorial_vqe.html",
    "https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html",
    "https://pennylane.ai/qml/demos/heisenberg.html",

    "https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.molecular_hamiltonian.html",

    "https://pennylane.ai/topics/hamiltonian-simulation",
    "https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series",

    "https://pennylane.ai/qml/demos/tutorial_barren_plateaus",
    "https://pennylane.ai/qml/demos/tutorial_quantum_circuit_cutting",

]

EXCLUDED_URLS = [
    "https://pennylane.ai/qml/demos/vqe_between_classical_methods.html",
    "https://docs.pennylane.ai/en/stable/introduction/quantum_circuits.html",
    "https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.html",
    "https://pennylane.ai/qml/topics/quantum-optimization-tutorial.html",
    "https://pennylane.ai/qml/topics/varq.html",
]


class HTMLContentExtractor(HTMLParser):

    def __init__(self):
        super().__init__()
        self.content = []
        self.in_script = False
        self.in_style = False
        self.in_nav = False
        self.skip_tags = {'script', 'style', 'nav', 'footer', 'header'}
        self.current_skip = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in self.skip_tags:
            self.current_skip += 1
        if tag == 'code' or tag == 'pre':
            self.content.append('\n```\n')

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.current_skip = max(0, self.current_skip - 1)
        if tag == 'code' or tag == 'pre':
            self.content.append('\n```\n')

    def handle_data(self, data):
        if self.current_skip == 0 and data.strip():
            self.content.append(data)

    def get_text(self):
        text = ' '.join(self.content)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()


class KnowledgeBaseBuilder:

    def __init__(self, output_dir: str = KNOWLEDGE_BASE_DIR):
        self.output_dir = output_dir
        self.keywords = DEFAULT_KEYWORDS.copy()
        self.scraped_urls = set()
        self.content_cache = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'total_urls_scraped': 0,
            'keywords': self.keywords,
            'hyperparameters': {
                'TOP_N': TOP_N,
                'NUM_COUPLING_TOPOLOGIES': NUM_COUPLING_TOPOLOGIES,
                'ENERGY_REDUCTION_THRESHOLD': ENERGY_REDUCTION_THRESHOLD,
                'MAX_CONTENT_LENGTH': MAX_CONTENT_LENGTH,
                'AUTO_DISCOVER_DEPTH': AUTO_DISCOVER_DEPTH,
                'MAX_DISCOVERED_URLS': MAX_DISCOVERED_URLS
            }
        }

        os.makedirs(output_dir, exist_ok=True)

        self._load_keywords()

    def _load_keywords(self):
        if os.path.exists(KEYWORDS_FILE):
            with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
                self.keywords = [line.strip() for line in f if line.strip()]

    def save_keywords(self):
        with open(KEYWORDS_FILE, 'w', encoding='utf-8') as f:
            f.write("# Knowledge base scraping keywords\n")
            f.write("# One keyword per line; add or remove them as needed\n")
            f.write("# Use English for new keywords because the source websites are in English\n\n")
            for keyword in self.keywords:
                f.write(f"{keyword}\n")

    def add_keyword(self, keyword: str):
        if keyword not in self.keywords:
            self.keywords.append(keyword)
            self.save_keywords()
            print(f"Added keyword: {keyword}")

    def remove_keyword(self, keyword: str):
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            self.save_keywords()
            print(f"Removed keyword: {keyword}")

    def _make_request(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        for excluded_url in EXCLUDED_URLS:
            if excluded_url in url:
                print(f"URL is in excluded list, skipping: {url}")
                return None

        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(REQUEST_DELAY)

        return None

    def _extract_content(self, html: str, url: str) -> str:
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, 'html.parser')

            for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))

            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
        else:
            extractor = HTMLContentExtractor()
            try:
                extractor.feed(html)
                text = extractor.get_text()
            except Exception as e:
                print(f"HTML parsing error: {e}")
                text = ""

        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _filter_by_keywords(self, content: str) -> bool:
        content_lower = content.lower()
        return any(keyword.lower() in content_lower for keyword in self.keywords)

    def _score_content(self, content: str) -> float:
        content_lower = content.lower()
        score = 0.0

        core_keywords = ['vqe', 'heisenberg', 'hamiltonian', 'variational', 'quantum circuit', 'ansatz']
        secondary_keywords = ['optimization', 'energy', 'ground state', 'expectation', 'measurement']

        for keyword in core_keywords:
            if keyword.lower() in content_lower:
                score += 2.0

        for keyword in secondary_keywords:
            if keyword.lower() in content_lower:
                score += 1.0

        for keyword in self.keywords:
            if keyword.lower() in content_lower and keyword.lower() not in core_keywords + secondary_keywords:
                score += 0.5

        return score

    def scrape_url(self, url: str) -> Optional[Dict]:
        if url in self.scraped_urls:
            print(f"Already scraped: {url}")
            return None

        print(f"Scraping: {url}")

        html = self._make_request(url)
        if not html:
            print(f"Failed to fetch: {url}")
            return None

        content = self._extract_content(html, url)

        if not self._filter_by_keywords(content):
            print(f"No relevant keywords found in: {url}")
            return None

        score = self._score_content(content)

        title = ""
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
        else:
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()

        result = {
            'url': url,
            'title': title,
            'content': content,
            'score': score,
            'scraped_at': datetime.now().isoformat()
        }

        self.scraped_urls.add(url)
        self.content_cache[url] = result

        return result

    def scrape_multiple_urls(self, urls: List[str], delay: float = REQUEST_DELAY) -> List[Dict]:
        results = []

        for i, url in enumerate(urls):
            print(f"\n[{i+1}/{len(urls)}] Processing: {url}")

            result = self.scrape_url(url)
            if result:
                results.append(result)
                print(f"  Score: {result['score']:.2f}, Content length: {len(result['content'])} chars")

            if i < len(urls) - 1:
                time.sleep(delay)

        return results

    def scrape_related_urls(self, base_url: str, max_depth: int = AUTO_DISCOVER_DEPTH) -> List[str]:
        print(f"\nDiscovering related URLs from: {base_url}")

        html = self._make_request(base_url)
        if not html:
            return []

        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, 'html.parser')
            links = set()

            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(base_url, href)

                parsed = urlparse(full_url)
                if parsed.netloc == urlparse(base_url).netloc:
                    if '/doc' in parsed.path or '/qml' in parsed.path or '/tutorial' in parsed.path:
                        links.add(full_url)

            return list(links)[:MAX_DISCOVERED_URLS]
        else:
            href_pattern = re.compile(r'href=["\'](.*?)["\']')
            links = set()

            for match in href_pattern.finditer(html):
                href = match.group(1)
                if href.startswith('/') or href.startswith('http'):
                    full_url = urljoin(base_url, href)
                    if 'pennylane' in full_url.lower():
                        links.add(full_url)

            return list(links)[:MAX_DISCOVERED_URLS]

    def build_knowledge_base(self, urls: List[str] = None, auto_discover: bool = True) -> str:
        if urls is None:
            urls = SEED_URLS

        all_results = []

        print("=" * 60)
        print("Starting to scrape PennyLane knowledge content")
        print(f"Hyperparameter configuration: TOP_N={TOP_N}, NUM_COUPLING_TOPOLOGIES={NUM_COUPLING_TOPOLOGIES}")
        print(f"ENERGY_REDUCTION_THRESHOLD={ENERGY_REDUCTION_THRESHOLD}")
        print("=" * 60)

        results = self.scrape_multiple_urls(urls)
        all_results.extend(results)

        if auto_discover:
            print("\n" + "=" * 60)
            print("Discovering related links automatically")
            print("=" * 60)

            for seed_url in urls[:3]:
                related = self.scrape_related_urls(seed_url)
                if related:
                    related_results = self.scrape_multiple_urls(related)
                    all_results.extend(related_results)
                    time.sleep(REQUEST_DELAY)

        all_results.sort(key=lambda x: x['score'], reverse=True)

        if TOP_N > 0 and len(all_results) > TOP_N:
            print(f"\nKeeping the {TOP_N} highest-scoring results according to TOP_N={TOP_N}")
            all_results = all_results[:TOP_N]

        knowledge_text = self._format_knowledge_base(all_results)

        with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            f.write(knowledge_text)

        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['total_urls_scraped'] = len(self.scraped_urls)
        self.metadata['top_n_results'] = len(all_results)

        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("Knowledge base construction complete")
        print("=" * 60)
        print(f"Total pages scraped: {len(self.scraped_urls)}")
        print(f"Relevant pages: {len(all_results)} (TOP_N={TOP_N})")
        print(f"Knowledge base file: {KNOWLEDGE_BASE_FILE}")

        return knowledge_text

    def _format_knowledge_base(self, results: List[Dict]) -> str:
        lines = []

        lines.append("=" * 80)
        lines.append("PennyLane VQE and Heisenberg Model Knowledge Base")
        lines.append("=" * 80)
        lines.append(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Source: pennylane.ai")
        lines.append(f"Number of keywords: {len(self.keywords)}")
        lines.append(f"Hyperparameter configuration: TOP_N={TOP_N}, NUM_COUPLING_TOPOLOGIES={NUM_COUPLING_TOPOLOGIES}")
        lines.append(f"ENERGY_REDUCTION_THRESHOLD={ENERGY_REDUCTION_THRESHOLD}")
        lines.append(f"MAX_CONTENT_LENGTH={MAX_CONTENT_LENGTH} (None means no truncation)")
        lines.append("=" * 80)
        lines.append("")

        lines.append("## Keyword List")
        lines.append("-" * 40)
        lines.append(", ".join(self.keywords[:20]))
        if len(self.keywords) > 20:
            lines.append(f"... and {len(self.keywords) - 20} more keywords")
        lines.append("")

        lines.append("## Knowledge Content")
        lines.append("=" * 80)
        lines.append("")

        for i, result in enumerate(results):
            lines.append(f"### [{i+1}] {result['title']}")
            lines.append(f"Source: {result['url']}")
            lines.append(f"Relevance score: {result['score']:.2f}")
            lines.append("")
            lines.append("---")
            lines.append("")

            content = result['content']
            if MAX_CONTENT_LENGTH is not None and len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated...]"
                lines.append(f"Note: Content truncated to the first {MAX_CONTENT_LENGTH} characters")
                lines.append("")

            lines.append(content)
            lines.append("")
            lines.append("=" * 80)
            lines.append("")

        lines.append("")
        lines.append("=" * 80)
        lines.append("End of Knowledge Base")
        lines.append("=" * 80)

        return '\n'.join(lines)

    def append_to_knowledge_base(self, new_content: Dict) -> None:
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        else:
            existing_content = ""

        new_text = self._format_single_content(new_content)

        updated_content = existing_content + "\n\n" + new_text

        with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        print(f"Added new content to knowledge base: {new_content['title']}")

    def _format_single_content(self, result: Dict) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(f"### {result['title']}")
        lines.append(f"Source: {result['url']}")
        lines.append(f"Added at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(result['content'])
        lines.append("")
        lines.append("=" * 80)
        return '\n'.join(lines)

    def get_knowledge_base_summary(self) -> Dict:
        if not os.path.exists(KNOWLEDGE_BASE_FILE):
            return {'exists': False}

        with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'exists': True,
            'file_path': KNOWLEDGE_BASE_FILE,
            'size_bytes': len(content),
            'size_chars': len(content),
            'scraped_urls': len(self.scraped_urls),
            'keywords_count': len(self.keywords),
            'hyperparameters': {
                'TOP_N': TOP_N,
                'NUM_COUPLING_TOPOLOGIES': NUM_COUPLING_TOPOLOGIES,
                'ENERGY_REDUCTION_THRESHOLD': ENERGY_REDUCTION_THRESHOLD,
                'MAX_CONTENT_LENGTH': MAX_CONTENT_LENGTH
            }
        }

class GeminiKnowledgeEnhancer:

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        print(f"API Base URL: {self.client.base_url}")

    def generate_circuit_guidance(self, knowledge_base: str, task: str = "8-qubit Heisenberg model VQE") -> str:

        system_prompt = """You are a quantum computing expert specializing in VQE (Variational Quantum Eigensolver) and quantum circuit design.
Based on the provided knowledge base, generate quantum circuit design guidance for the specified task."""

        user_prompt = f"""## Task
Based on the following PennyLane knowledge base, generate quantum circuit design guidance for {task}.

Current hyperparameter configuration:
- TOP_N: {TOP_N} (number of highest-scoring results to retain)
- NUM_COUPLING_TOPOLOGIES: {NUM_COUPLING_TOPOLOGIES} (number of coupling topologies)
- ENERGY_REDUCTION_THRESHOLD: {ENERGY_REDUCTION_THRESHOLD} (energy reduction threshold)

## Knowledge Base Content
{knowledge_base[:30000]}

## Requirements
1. Extract the technical points most relevant to VQE and the Heisenberg model
2. Recommend key quantum circuit design parameters, such as circuit depth, gate types, and parameter count
3. Consider {NUM_COUPLING_TOPOLOGIES} different coupling topologies
4. Energy optimization target: reduce energy by {ENERGY_REDUCTION_THRESHOLD}
5. Provide best practice recommendations
6. List common mistakes to avoid

Present the guidance in a structured format."""

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""

    def generate_circuit_for_topology(self, knowledge_base: str, topology_type: str = "grid") -> str:

        prompt = f"""Based on the following knowledge base, generate a quantum circuit design for the {topology_type} coupling topology of the 8-qubit Heisenberg model.
Consider {NUM_COUPLING_TOPOLOGIES} different coupling topologies; the current topology is {topology_type}.
The energy reduction threshold target is {ENERGY_REDUCTION_THRESHOLD}.

Knowledge base content:
{knowledge_base[:20000]}

Please provide:
1. A specific quantum circuit design (gate sequence and parameterization)
2. The expected number of parameters
3. Optimization strategy recommendations
4. Special considerations for this topology"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in quantum circuit design."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating circuit for topology {topology_type}: {e}")
            return ""


def main():
    print("=" * 80)
    print("Quantum Circuit Knowledge Base Builder - quantum_principle_learner36.py")
    print("=" * 80)
    print(f"Hyperparameter configuration:")
    print(f"  TOP_N: {TOP_N}")
    print(f"  NUM_COUPLING_TOPOLOGIES: {NUM_COUPLING_TOPOLOGIES}")
    print(f"  ENERGY_REDUCTION_THRESHOLD: {ENERGY_REDUCTION_THRESHOLD}")
    print(f"  MAX_CONTENT_LENGTH: {MAX_CONTENT_LENGTH}")
    print("=" * 80)

    builder = KnowledgeBaseBuilder()

    builder.save_keywords()
    print(f"\nKeyword list saved to: {KEYWORDS_FILE}")
    print(f"Current number of keywords: {len(builder.keywords)}")

    print(f"\nExcluded URLs ({len(EXCLUDED_URLS)}):")
    for url in EXCLUDED_URLS:
        print(f"  - {url}")

    print(f"\nSeed URLs ({len(SEED_URLS)}):")
    for url in SEED_URLS:
        print(f"  - {url}")

    print("\nCurrent keywords (first 15):")
    for i, keyword in enumerate(builder.keywords[:15]):
        print(f"  {i+1}. {keyword}")
    if len(builder.keywords) > 15:
        print(f"  ... and {len(builder.keywords) - 15} more keywords")

    print("\n" + "=" * 80)
    print("Starting knowledge base construction")
    print("=" * 80)

    knowledge_text = builder.build_knowledge_base(
        urls=SEED_URLS,
        auto_discover=True
    )

    summary = builder.get_knowledge_base_summary()
    print("\nKnowledge base summary:")
    print(f"  File path: {summary['file_path']}")
    print(f"  File size: {summary['size_bytes']} bytes")
    print(f"  Character count: {summary['size_chars']} chars")
    print(f"  Pages scraped: {summary['scraped_urls']}")
    print(f"  Pages retained (TOP_N): {summary.get('top_n_results', 'N/A')}")

    if OPENAI_AVAILABLE:
        print("\n" + "=" * 80)
        print("Gemini API configuration check")
        print("=" * 80)
        print(f"API_KEY: {API_KEY[:10]}...")
        print(f"BASE_URL: {BASE_URL}")
        print(f"MODEL: {MODEL}")

        use_gemini = input("\nUse the Gemini API to enhance the knowledge base? (y/n): ").strip().lower()
        if use_gemini == 'y':
            print("\nEnhancing content with the Gemini API...")

            try:
                enhancer = GeminiKnowledgeEnhancer()

                print("Generating general quantum circuit design guidance...")
                guidance = enhancer.generate_circuit_guidance(knowledge_text)

                if guidance:
                    with open(CIRCUIT_GUIDANCE_FILE, 'w', encoding='utf-8') as f:
                        f.write(guidance)
                    print(f"General design guidance saved to: {CIRCUIT_GUIDANCE_FILE}")

                print(f"\nGenerating circuit designs for {NUM_COUPLING_TOPOLOGIES} coupling topologies...")
                topologies = ["grid", "linear", "star", "ring", "fully_connected"]
                for i in range(min(NUM_COUPLING_TOPOLOGIES, len(topologies))):
                    topology = topologies[i]
                    print(f"Generating circuit design for the {topology} topology...")
                    circuit_design = enhancer.generate_circuit_for_topology(knowledge_text, topology)

                    if circuit_design:
                        topology_file = os.path.join(KNOWLEDGE_BASE_DIR, f"circuit_design_{topology}.txt")
                        with open(topology_file, 'w', encoding='utf-8') as f:
                            f.write(circuit_design)
                        print(f"  Circuit design for the {topology} topology saved to: {topology_file}")

            except Exception as e:
                print(f"Gemini API call failed: {e}")
                print("Check the API key and network connection")
    else:
        print("\nOpenAI library not found; skipping Gemini API enhancement")

    print("\n" + "=" * 80)
    print("Knowledge base construction complete")
    print("=" * 80)
    print(f"Hyperparameter summary:")
    print(f"  TOP_N: {TOP_N} (adjust to 10, 50, etc. for experiments at different scales)")
    print(f"  NUM_COUPLING_TOPOLOGIES: {NUM_COUPLING_TOPOLOGIES}")
    print(f"  ENERGY_REDUCTION_THRESHOLD: {ENERGY_REDUCTION_THRESHOLD}")
    print(f"  MAX_CONTENT_LENGTH: {MAX_CONTENT_LENGTH} (None means no truncation)")
    print("\nOutput files:")
    print(f"  Knowledge base: {KNOWLEDGE_BASE_FILE}")
    print(f"  Keyword list: {KEYWORDS_FILE}")
    print(f"  Metadata: {METADATA_FILE}")
    if os.path.exists(CIRCUIT_GUIDANCE_FILE):
        print(f"  Circuit design guidance: {CIRCUIT_GUIDANCE_FILE}")

    return builder


if __name__ == "__main__":
    builder = main()
