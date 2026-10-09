import pathlib

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_output_dir_name(args) -> str:
    """Directory name under ./lcb_outputs/ for generation/eval artifacts."""
    custom_name = getattr(args, "custom_output_save_name", None)
    if custom_name:
        return custom_name
    local_model_path = getattr(args, "local_model_path", None)
    if local_model_path:
        return local_model_path.rstrip("/").split("/")[-1]
    return "unknown_model"


def get_lcb_output_root(args) -> str:
    """Root for LCB generation, eval, and prompt cache files."""
    custom_root = getattr(args, "lcb_output_root", None)
    if custom_root:
        return custom_root.rstrip("/")
    model_name = get_output_dir_name(args)
    return f"./lcb_outputs/{model_name}"


def get_cache_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    root = get_lcb_output_root(args)
    path = f"{root}/cache/{scenario}_{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    root = get_lcb_output_root(args)
    path = f"{root}/{scenario}_{n}_{temperature}{cot_suffix}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    root = get_lcb_output_root(args)
    path = f"{root}/{scenario}_{n}_{temperature}{cot_suffix}_eval_all.json"
    return path
