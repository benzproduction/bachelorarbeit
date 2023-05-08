"""Helper functions for the project."""
import re

def get_env(case=None):
    """Get the environment vars for the project.
    Args:
        case (str): The case to setup the environment for.
                    Possible values are:
                    - "azure-openai"
    Returns:
        case: "azure-openai" -> (OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)

    """
    import os
    if case == "azure-openai":
        with open("/Users/shuepers001/dev/bachelorarbeit/.env", "r") as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                key, value = line.split("=")
                if key == "OPENAI_API_KEY":
                    os.environ["OPENAI_API_KEY"] = value.strip()
                elif key == "AZURE_OPENAI_ENDPOINT":
                    os.environ["AZURE_OPENAI_ENDPOINT"] = value.strip()
        return os.environ["OPENAI_API_KEY"], os.environ["AZURE_OPENAI_ENDPOINT"]
    

def get_segments_from_output(output):
    """The start of a segment is marked by <SEG> and the end by </SEG>."""
    segments = []
    start = "<SEG>"
    end = "</SEG>"
    while output.find(start) != -1 and output.find(end) != -1:
        segment_start = output.find(start) + len(start)
        segment_end = output.find(end)
        segment = output[segment_start:segment_end]
        segments.append(segment)
        output = output[segment_end+len(end):]
    return segments

class SimpleSentenceTokenizer:

    def __init__(self, breaking_chars='.!?'):
        assert len(breaking_chars) > 0
        self.breaking_chars = breaking_chars
        self.prog = re.compile(r".+?[{}]\W+".format(breaking_chars), re.DOTALL)

    def __call__(self, text):
        return self.prog.findall(text)