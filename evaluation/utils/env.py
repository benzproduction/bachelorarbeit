"""Helper functions to work with environment variables."""

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