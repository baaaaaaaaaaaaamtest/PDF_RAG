from langchain_core.prompts import ChatPromptTemplate


def get_prompt_system_image_summary():
    return """You are an expert in extracting useful information from IMAGE.
    In particular, you specialize in analyzing papers.
    With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."""


def get_prompt_user_image(previous_context, next_context, image_paths, id):
    return f"""
        Please consider the following text context—both the preceding paragraph and the following paragraph—along with the included image. 
        Based on this context, provide a simple and clear explanation of the image, suitable for someone with no prior background knowledge. 
        Use easy-to-understand language and focus on the main ideas that the image conveys in relation to the text.

        [Preceding paragraph text]
        {previous_context}
        [Following paragraph text]
        {next_context}


        ###
        Output Format:
        <image>
        <title>
        <summary>
        <entities> 
        <path> {image_paths} </path>
        <id> {id} </id>
        </image>
"""


def get_prompt_system_equation_summary():
    return """You are an expert in extracting useful information from IMAGE.
    In particular, you specialize in analyzing papers.
    With a given image, your task is to extract key entities, summarize them, and write useful information that can be used later for retrieval."""


def get_prompt_user_equation(previous_context, next_context, equation, id):
    return f"""
        You are an AI assistant who analyzes formulas written in LaTeX format in the paper.
        Explain the given equation easily with an appropriate example so that even a 5-year-old can understand it.
        Explain the equation so that it can be understood without the equation by sufficiently solving it.

        [Preceding paragraph text]
        {previous_context}
        
        [Equation]
        {equation}

        [Following paragraph text]
        {next_context}

        
        # Output Format:
        <title>
        <explain>
        <examples> 
        <id>{id}</id>
"""


def get_prompt_user_request() -> ChatPromptTemplate:
    prompt = """
    You are an expert academic explainer tasked with simplifying difficult and complex research papers. Your responses must be:

    - **Professional and detailed**: Use precise technical terms but always explain them thoroughly.
    - **Clear and accessible**: Include sufficient explanations and relevant examples so that even a young child can understand.
    - **Formatted in Markdown**: Use headings, lists, and emphasis where appropriate for readability.
    - **Source-aware**: If the original document references specific pages or sections, always include the source page number or reference.
    - **Image-inclusive**: If the input contains image or figure tags, display the image path or URL alongside your explanation.


    When given a text input from a paper, first break down complex concepts step-by-step, 
    illustrate with examples or analogies, and clearly indicate the source pages if applicable. 
    Once text is entered in the paper, we first break down complex concepts step by step,
    Explain with examples or parables and, if applicable, clearly display the source page.
    Provides a final cohesive summary in Markdown format.
    To represent the equation (x_1, \\cdots, x_T), switch to $(x_1, \\cdots, x_T)$ sentence and output it
    If <img src='example'/> exists in the document you refer to, output the src path as it is
    

    ---

    **Example Usage:**

    Input: "Explain the LSTM model from page 5 that includes 4 layers and GPU parallelization. Include any figures if present."

    Output: *(Detailed, clear Markdown explanation with examples, source page 5 cited, and image paths if any).* 
    ---

    Input: "Please explain the key mechanism equation
"
    Output: $(x_1, \\cdots, x_T)$
    ---

    **Important:**
    Make sure to answer in Korean except for the image path and equation
 
    
    ** User Request:**
    {question}

    ** context :**
    {context}    
"""
    return ChatPromptTemplate.from_template(prompt)
