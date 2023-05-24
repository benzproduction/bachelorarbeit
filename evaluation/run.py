import os
import yaml
import jsonlines
import inquirer
from pathlib import Path
from base import RunSpec
from registry import Registry
from eval import EvalRun
import logging
import base64
import random
from typing import Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

cwd = os.getcwd()

def _purple(str):
    return f"\033[1;35m{str}\033[0m"
def _green(str):
    return f"\033[1;32m{str}\033[0m"

SHUFFLE_SEED = 123


def _index_samples(samples: List[Any], max_samples: Optional[int] = None) -> List[Tuple[Any, int]]:
    """Shuffle `samples` and pair each sample with its index."""
    indices = list(range(len(samples)))
    random.Random(SHUFFLE_SEED).shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]
    logger.info(f"Evaluating {len(indices)} samples")
    work_items = [(samples[i], i) for i in indices]
    return work_items

def evaluate():
    """Evaluate a model based on a selected configuration and data file."""

    # Prompt user to choose the type of evaluation
    evaluation_types = ['information_retriever', 'answer_generator', 'end_to_end']
    evaluation_type_question = [
        inquirer.List('evaluation_type',
                      message="Select the type of evaluation:",
                      choices=evaluation_types,
                      carousel=True)
    ]
    evaluation_type = inquirer.prompt(evaluation_type_question)['evaluation_type']

    if evaluation_type == 'information_retriever':
        # List retriever configurations and prompt the user to select one
        config_dir = Path(cwd + '/evaluation/registry/retriever')
        config_choices = [f"{f.stem}" for f in config_dir.glob('*.yaml')]
        config_question = [
            inquirer.List('retriever',
                            message=f"Select a retriever to evaluate",
                            choices=config_choices,
                            carousel=True)
        ]
        retriever_choice = inquirer.prompt(config_question)['retriever']
        registry = Registry()
        embedder = registry.make_embedding('text_embedding_ada_002') # TODO: Make this configurable
        retriever = registry.make_retriever(retriever_choice)
        run_config = {
            'embedder': embedder,
            'retriever': retriever
        }
        data_dir = Path(cwd +f'/evaluation/data/{evaluation_type}') 

    elif evaluation_type == 'end_to_end':
        # List model configurations and prompt the user to select one 
        config_dir = Path(cwd + '/evaluation/registry/model_config')
        # Load the YAML files into dictionaries
        config_dicts = {}
        for config_file in config_dir.glob('*.yaml'):
            with open(config_file, 'r') as file:
                config_data = yaml.safe_load(file)
                config_name = str(config_file.relative_to(config_dir).with_suffix(''))
                config_dicts[config_name] = config_data

        # Create a custom message for each option
        config_choices = [f"{name}: {config['metadata']['description']} ({config['metadata']['version']})"
                        for name, config in config_dicts.items()]

        config_question = [
            inquirer.List('config',
                        message=f"Select a {evaluation_type} configuration",
                        choices=config_choices,
                        carousel=True)
        ]
        config_choice = inquirer.prompt(config_question)['config']

        # Extract the selected config name from the custom message
        selected_config_name = config_choice.split(':', 1)[0].strip()
        print(_purple(f"Creating {evaluation_type} with {selected_config_name}..."))
        model_config = config_dicts[selected_config_name][selected_config_name]
        registry = Registry()
        embedder = registry.make_embedding(model_config['embedding'])
        retriever = registry.make_retriever(model_config['retriever'], model_config.get('retriever_args', {}))
        llm = registry.make_llm(model_config['language_model'])
        prompt = registry.get_prompt(model_config['prompt'])
        run_config = {
            "embedder": embedder,
            "retriever": retriever,
            "llm": llm,
            "prompt_template": prompt,
            **model_config.get('run_args', {})
        }
        print(_green("Successfully created run config!"))

        data_dir = Path(cwd +f'/evaluation/data/{evaluation_type}')
    elif evaluation_type == 'answer_generator':
        exit("Answer generator evaluation not yet implemented")

    data_files = [str(f.relative_to(data_dir).with_suffix('')) for f in data_dir.glob('*.jsonl')]
    data_question = [
        inquirer.List('data',
                        message=f"Select a dataset file",
                        choices=data_files,
                        carousel=True)
    ]
    data_file = inquirer.prompt(data_question)['data']

    # Load data
    data_path = data_dir / f'{data_file}.jsonl'
    with jsonlines.open(data_path) as reader:
        data_list = [item for item in reader]

    # Prompt user to choose the number of samples to evaluate
    sample_question = [
        inquirer.Text('sample_size',
                    message=f"Enter the number of samples to evaluate (max {len(data_list)})",
                    validate=lambda _, x: x.isdigit() and int(x) > 0 and int(x) <= len(data_list))
    ]
    sample_size = int(inquirer.prompt(sample_question)['sample_size'])
    rand_suffix = base64.b32encode(os.urandom(5)).decode("ascii")
    try:
        config_name = selected_config_name
    except NameError:
        config_name = retriever_choice
    eval_name = f'{evaluation_type}_{config_name}_{data_file}_{rand_suffix}'
    run_spec = RunSpec(
            eval_name=eval_name,
            eval_type=evaluation_type,
            run_config=run_config,
            data=_index_samples(data_list, sample_size),
    )
    log_dir = Path(cwd + '/evaluation/runs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / f'{eval_name}.jsonl'
    eval_run = EvalRun(run_spec, log_path.as_posix())
    eval_run.run()
    final_report = eval_run.generate_report()
    eval_run.record_final_report(final_report.to_dict())
    print("\n\n\n"+str(final_report))


if __name__ == '__main__':
    evaluate()
