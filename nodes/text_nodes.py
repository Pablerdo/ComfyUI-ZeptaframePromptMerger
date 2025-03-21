import json

class MergePrompts():
  @classmethod
  def INPUT_TYPES(self):
    return {
      "required": {
        "generalSa2VAPrompt": ("STRING", ),
        "generalTextPrompt": ("STRING", ),
        "subjectTextPrompts": ("STRING", ),
      },
    }
  
  RETURN_TYPES = ("STRING", )
  RETURN_NAMES = ("prompt", )
  FUNCTION = "run"

  CATEGORY = "ZeptaFramePromptMerger/text"

  def run(self, generalSa2VAPrompt, generalTextPrompt, subjectTextPrompts):
    # Deserialize the prompts
    generalSa2VAPrompt = json.loads(generalSa2VAPrompt)
    generalTextPrompt = json.loads(generalTextPrompt)
    subjectTextPrompts = json.loads(subjectTextPrompts)

    # Merge the prompts
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")
    
    # Initialize LLama model (adjust path to your model)
    try:
        model_path = "zepta/llama-2-7b-chat.Q8_0.gguf"  # Path to your LLama model
        llm = Llama(model_path=model_path, n_ctx=4096)
    except Exception as e:
        return f"Error loading LLama model: {str(e)}"
    
    # Format subjects and their descriptions
    subject_descriptions = ""
    for subject, description in subjectTextPrompts.items():
        subject_descriptions += f"- {subject}: {description}\n"
    
    # Create prompt with importance weightings reflected in the instruction
    prompt = f"""
You are creating a detailed prompt for a text-to-video generation system.

MOST IMPORTANT (8/10): Subject descriptions that detail how specific elements should move or appear:
{subject_descriptions}

IMPORTANT (7/10): General description of the video:
{generalTextPrompt}

BACKGROUND CONTEXT (2/10): System-generated video caption:
{generalSa2VAPrompt}

Based on the above information, merge the inputs into a cohesive and detailed prompt for a text-to-video system. Emphasize the subject movements and appearance as the highest priority
"""

    # Generate response from LLama
    try:
        response = llm(prompt, max_tokens=512, temperature=0.7, top_p=0.95)
        merged_prompt = response["choices"][0]["text"].strip()
    except Exception as e:
        return (f"Error generating with LLama: {str(e)}",)

    return (merged_prompt,) 
  