"""
Utility functions for text processing.
"""

import re
import string


def clean_text(text: str, remove_non_printable: bool = True) -> str:
    """
    Clean a single text string.
    
    Args:
        text: Input text to clean
        remove_non_printable: Whether to remove non-printable characters
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    cleaned = text.strip()
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'(</s>|<eos>)', '', cleaned)
    
    if remove_non_printable:
        printable_chars = set(string.printable)
        cleaned = ''.join(filter(lambda x: x in printable_chars, cleaned))
    
    return cleaned


def create_context_window(text: str, target_string: str, window_size: int = 200) -> str:
    """
    Create a context window around the target string.
    
    Args:
        text: Full text to search in
        target_string: String to find and create context around
        window_size: Number of characters before and after target
    
    Returns:
        Context window text, or full text if target not found
    """
    target_index = text.find(target_string)
    
    if target_index != -1:
        start_index = max(0, target_index - window_size)
        end_index = min(len(text), target_index + len(target_string) + window_size)
        context_window = text[start_index:end_index]
        return context_window
    
    return text  # Return full text if target not found
