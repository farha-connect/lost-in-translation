# sentences.py
# Loads sentences from real datasets for research credibility

from datasets import load_dataset
import wikipediaapi
import random

def get_sarcasm_sentences(n=50):
    """
    Loads sarcastic headlines from the Sarcasm News Headlines dataset
    (Misra & Arora, 2023) — 28,619 headlines total
    We filter for sarcastic ones only (is_sarcastic == 1)
    """
    print("Loading sarcasm dataset...")
    
    dataset = load_dataset("raquiba/Sarcasm_News_Headline")
    
    # Filter only sarcastic headlines
    # is_sarcastic == 1 means it IS sarcastic
    sarcastic_only = [
        row['headline'] 
        for row in dataset['train'] 
        if row['is_sarcastic'] == 1
    ]
    
    # Shuffle so we get variety each time
    random.seed(42)  # seed=42 means same shuffle every run (reproducible)
    random.shuffle(sarcastic_only)
    
    # Take first n sentences
    selected = sarcastic_only[:n]
    
    print(f"✅ Loaded {len(selected)} sarcastic headlines")
    return selected


def get_emotional_sentences(n=50):
    """
    Loads emotional sentences from GoEmotions dataset (Google Research)
    43,410 Reddit comments labeled with 28 emotions
    We filter for high-emotion sentences only
    """
    print("Loading emotional sentences dataset...")
    
    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified"
    )
    
    # Emotion labels we want — clearly emotional, not neutral
    # In GoEmotions simplified: 0-27 are emotion indices
    # 27 = neutral — we EXCLUDE this
    # We want sentences with strong clear emotions
    emotional_labels = [
        0,   # admiration
        2,   # anger  
        3,   # annoyance
        4,   # approval
        6,   # disappointment
        7,   # disapproval
        8,   # disgust
        9,   # embarrassment
        10,  # excitement
        11,  # fear
        12,  # gratitude
        13,  # grief
        14,  # joy
        15,  # love
        16,  # nervousness
        17,  # optimism
        18,  # pride
        19,  # realization
        20,  # relief
        21,  # remorse
        22,  # sadness
        23,  # surprise
    ]
    
    # Filter sentences that:
    # 1. Have at least one strong emotion label
    # 2. Are between 10 and 100 characters (not too short or long)
    # 3. Don't contain links or special characters
    emotional_sentences = []
    
    for row in dataset['train']:
        text = row['text']
        labels = row['labels']
        
        # Check if any label is a strong emotion
        has_emotion = any(label in emotional_labels for label in labels)
        
        # Check length is reasonable
        good_length = 20 < len(text) < 120
        
        # Check no URLs
        no_urls = 'http' not in text and 'www' not in text
        
        # Check no Reddit-specific formatting
        no_formatting = '&gt;' not in text and '/r/' not in text
        
        if has_emotion and good_length and no_urls and no_formatting:
            emotional_sentences.append(text)
    
    # Shuffle and select
    random.seed(42)
    random.shuffle(emotional_sentences)
    selected = emotional_sentences[:n]
    
    print(f"✅ Loaded {len(selected)} emotional sentences")
    return selected


def get_technical_sentences(n=50):
    """
    Extracts technical sentences from Wikipedia CS articles
    Using Wikipedia-API library
    """
    print("Loading technical sentences from Wikipedia...")
    
    # Initialize Wikipedia API
    # The user agent string identifies our app to Wikipedia
    wiki = wikipediaapi.Wikipedia(
        'LostInTranslation/1.0 (research.project@example.com)',
        'en'
    )
    
    # CS topics to pull sentences from
    # More topics = more variety
    topics = [
        'Machine learning',
        'Artificial intelligence',
        'Computer network',
        'Database',
        'Algorithm',
        'Operating system',
        'Computer programming',
        'Data structure',
        'Cryptography',
        'Natural language processing'
    ]
    
    all_sentences = []
    
    for topic in topics:
        page = wiki.page(topic)
        
        if not page.exists():
            continue
        
        # Split page text into sentences
        # Simple split by period
        raw_sentences = page.text.split('.')
        
        for sentence in raw_sentences:
            sentence = sentence.strip()
            
            # Filter for good technical sentences
            good_length = 40 < len(sentence) < 150
            no_references = '[' not in sentence
            no_newlines = '\n' not in sentence
            has_content = len(sentence.split()) > 8
            
            if good_length and no_references and no_newlines and has_content:
                all_sentences.append(sentence + '.')
    
    # Remove duplicates
    all_sentences = list(set(all_sentences))
    
    # Shuffle and select
    random.seed(42)
    random.shuffle(all_sentences)
    selected = all_sentences[:n]
    
    print(f"✅ Loaded {len(selected)} technical sentences")
    return selected


def get_idiomatic_sentences(n=50):
    """
    Manually curated idiomatic expressions
    No suitable free dataset exists for English idioms
    These are common English idioms that translate poorly
    to Japanese and Hindi — chosen specifically for high drift
    """
    print("Loading idiomatic sentences...")
    
    idioms = [
        "Break a leg at your performance tonight.",
        "It's raining cats and dogs outside right now.",
        "She let the cat out of the bag accidentally.",
        "He's been burning the midnight oil all week.",
        "Don't beat around the bush, just tell me the truth.",
        "That idea is a real piece of cake to implement.",
        "We need to think outside the box on this one.",
        "She hit the nail right on the head with that comment.",
        "He bit off more than he could chew with that project.",
        "Don't judge a book by its cover.",
        "We're back to square one after that failure.",
        "She spilled the beans before the announcement.",
        "He's barking up the wrong tree with that theory.",
        "The ball is in your court now, make a decision.",
        "That meeting was a blessing in disguise.",
        "She's been feeling under the weather all week.",
        "He always passes the buck when things go wrong.",
        "Don't count your chickens before they hatch.",
        "She cut corners to finish the project faster.",
        "He's been dragging his feet on this decision.",
        "That's just the tip of the iceberg.",
        "She has too many irons in the fire right now.",
        "He jumped on the bandwagon when it became popular.",
        "She kept him in the loop throughout the project.",
        "He's been killing two birds with one stone.",
        "That argument doesn't hold water at all.",
        "She let sleeping dogs lie instead of confronting him.",
        "He missed the boat on that golden opportunity.",
        "Don't add fuel to the fire right now.",
        "She's been on the fence about the big decision.",
        "He opened a can of worms with that question.",
        "That's a whole other kettle of fish entirely.",
        "She pulled the plug on the failing project.",
        "He put all his eggs in one basket.",
        "Don't rock the boat when things are going well.",
        "She saved the day at the very last minute.",
        "He's been spinning his wheels without any progress.",
        "That's just the straw that broke the camel's back.",
        "She took the bull by the horns and fixed everything.",
        "He threw in the towel after months of struggling.",
        "That's water under the bridge now, let it go.",
        "She went back to the drawing board completely.",
        "He wrapped his head around the complex problem.",
        "You can't have your cake and eat it too.",
        "She was caught between a rock and a hard place.",
        "He was given the cold shoulder at the meeting.",
        "That project is on the back burner for now.",
        "She read between the lines of his ambiguous message.",
        "He really stepped up to the plate when it mattered.",
        "Once in a blue moon she actually admits she's wrong."
    ]
    
    print(f"✅ Loaded {len(idioms)} idiomatic sentences")
    return idioms[:n]


def build_all_sentences():
    """
    Loads all 4 categories and returns them as a dictionary
    This is the main function you call from other files
    """
    
    print("\n" + "="*50)
    print("LOADING ALL SENTENCE DATASETS")
    print("="*50 + "\n")
    
    sentences = {
        'sarcasm': get_sarcasm_sentences(50),
        'emotional': get_emotional_sentences(50),
        'technical': get_technical_sentences(50),
        'idiomatic': get_idiomatic_sentences(50)
    }
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    for category, sents in sentences.items():
        print(f"{category}: {len(sents)} sentences")
    print(f"Total: {sum(len(s) for s in sentences.values())} sentences")
    print("="*50 + "\n")
    
    return sentences


# Test when run directly
if __name__ == "__main__":
    sentences = build_all_sentences()
    
    # Show one example from each category
    print("\nSAMPLE SENTENCES:")
    for category, sents in sentences.items():
        print(f"\n{category.upper()}:")
        print(f"  → {sents[0]}")
        print(f"  → {sents[1]}")
        print(f"  → {sents[2]}")