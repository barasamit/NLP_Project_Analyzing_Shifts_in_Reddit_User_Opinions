import json
import transformers
import torch
from typing import Dict, Any, List
from tqdm import tqdm
from collections import defaultdict

class RedditCommentTree:
    def __init__(self):
        self.comments_by_id = {}  # Store all comments for easy lookup
        self.children = defaultdict(list)  # Store parent->child relationships

    def add_comment(self, comment: Dict[str, Any]):
        """Add a comment to the tree and update relationships."""
        comment_id = self._extract_id_from_fullname(comment.get('parent_id', ''))
        self.comments_by_id[comment_id] = comment
        self.children[comment_id].append(comment)

    def _extract_id_from_fullname(self, fullname: str) -> str:
        """Extract the ID part from Reddit's fullname format (t1_xxx, t3_xxx, etc)."""
        if not fullname:
            return ''
        parts = fullname.split('_')
        return parts[1] if len(parts) > 1 else fullname

    def count_total_comments(self) -> int:
        """Count total number of comments in the tree."""
        # Count unique comments by their content
        unique_comments = set()
        for comments_list in self.children.values():
            for comment in comments_list:
                # Create a tuple of identifying fields
                comment_key = (
                    comment.get('author', ''),
                    comment.get('body', ''),
                    comment.get('created_utc', ''),
                    comment.get('author_fullname', '')
                )
                unique_comments.add(comment_key)
        return len(unique_comments)

def create_prompt(text: str) -> List[Dict[str, str]]:
    """Create a prompt for the LLaMA model to analyze stance with few-shot examples."""
    few_shot_examples = [
        # Pro-Israel examples
        ("About wars that were fought without innocents getting caught up? There's no such thing.", "pro-israel"),
        ("What are you referring to? Could you please provide examples of Israeli laws that treat the Arab citizens different from the Jewish ones?", "pro-israel"),
        ("You base your claims on reporters while sitting hundreds of miles away. I'm basing my knowledge on personal friends operating in Gaza. Palestinian combatants are almost nonexistent; casualties to our forces are almost always side charges hidden in rubble and occasional RPG shots from miles away.", "pro-israel"),
        
        # Pro-Palestine examples
        ("Hamas has no plans to eliminate Israel.", "pro-palestine"),
        ("When someone commits a genocide on anyone else, they will obviously get public mercy. IMO, Israel is a terrorist organization right now.", "pro-palestine"),
        ("You're talking about the Arab citizens of 'Israel.' A state that the people who used to live on that land are largely not a part of now. They live in a large concentration camp and can't freely move without the permission of the Israeli state.", "pro-palestine"),
        
        # Neutral examples
        ("Yes, that is a more polite version. But since you are essentially denying the legitimacy of a Jewish state, it's just as hostile. I say that because what you suggest is impossible.", "neutral"),
        ("It's really upsetting how people live under Israeli occupation, but the fact is that a fully independent Palestine would immediately erupt into a civil war with factions funded by Iran and other groups.", "neutral"),
        ("The antisemitism and Islamophobia mainly come from the wars between Israel and Palestine. If Israel accepts the two-state solution and gets out of Palestine, both countries would live in peace.", "neutral")
    ]

    # Create the prompt with examples
    prompt_examples = "\n\n".join([
        f"Text: {example[0]}\nStance: {example[1]}"
        for example in few_shot_examples
    ])

    return [
        {
            "role": "system",
            "content": "You are an expert at analyzing text about the Israel-Palestine conflict. "
                      "Classify the stance as either 'pro-israel', 'pro-palestine', or 'neutral'. "
                      "Only respond with one of these three classifications.\n\n"
                      "Here are some examples:\n\n" + prompt_examples
        },
        {
            "role": "user",
            "content": f"Analyze this text and determine if it's pro-israel, pro-palestine, or neutral: {text}"
        }
    ]

def setup_pipeline():
    """Initialize the LLaMA pipeline."""
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

def predict_stance(pipeline, text: str) -> str:
    """Predict the stance of a given text."""
    messages = create_prompt(text)
    output = pipeline(messages, max_new_tokens=32)
    
    # Extract the assistant's response from the conversation
    try:
        # Get the last message which should be the assistant's response
        assistant_response = output[0]['generated_text'][-1]['content'].lower()
        # print(f"Text: {text}")
        # print(f"Assistant response: {assistant_response}")
        
        # Extract the stance
        if "pro-israel" in assistant_response:
            return "pro-israel"
        elif "pro-palestine" in assistant_response:
            return "pro-palestine"
        else:
            return "neutral"
    except (KeyError, IndexError):
        # If there's any error in extracting the stance, return neutral as default
        print(f"Warning: Could not properly extract stance from model output: {output}")
        return "neutral"
    
def count_all_comments(comments_obj: Dict) -> int:
    """Count all unique comments in a submission."""
    if not comments_obj:
        return 0
    
    tree = RedditCommentTree()
    for comment_list in comments_obj.values():
        for comment in comment_list:
            tree.add_comment(comment)
    return tree.count_total_comments()


def process_reddit_data(file_path: str):
    """Process Reddit data and add stance predictions."""
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Setup the pipeline
    pipeline = setup_pipeline()
    
    # Count total raw comments for progress bar
    total_comments = 0
    submission_comment_counts = {}  # Store comment counts per submission
    for submission_id, submission_data in data.items():
        submission_count = 0
        for comments_list in submission_data['comments'].values():
            submission_count += len(comments_list)
        submission_comment_counts[submission_id] = submission_count
        total_comments += submission_count
    
    print(f"\nProcessing {len(data)} submissions with {total_comments} total comments")
    
    # Initialize counter for overall progress
    processed_comments = 0
    
    # Create main progress bar for total comments
    with tqdm(total=total_comments, desc="Processing all comments") as main_pbar:
        # Process submissions
        for submission_id, submission_data in data.items():
            # Add stance to submission
            submission_text = submission_data['submission']['title'] + " " + submission_data['submission']['selftext']
            submission_data['submission']['stance'] = predict_stance(pipeline, submission_text)
            
            # Process comments
            for comment_id, comments in submission_data['comments'].items():
                for comment in comments:
                    comment['stance'] = predict_stance(pipeline, comment['body'])
                    processed_comments += 1
                    main_pbar.update(1)
    
    # Generate output filename and save
    input_filename = file_path.split('/')[-1]
    output_filename = input_filename.replace('filtered_', 'labeled_')
    output_path = '/home/meirio/nlp/cleaned_and_labeled_reddit_files/' + output_filename
    
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=2)
    
    print(f"\nProcessed {len(data)} submissions and {processed_comments} comments")
    print(f"Results saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    input_file = "/home/meirio/nlp/Filtered_cleaned_reddit_files/filtered_cleaned_IsraelPalestine_2024-10_full_data.json"  # Replace with your input file path
    print(f"Processing Reddit data from {input_file}")
    output_file = process_reddit_data(input_file)
    print(f"Processing complete. Results saved to {output_file}")