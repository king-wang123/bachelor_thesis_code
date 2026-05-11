from reward_utils import evaluate_response


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    item = extra_info or {}
    result = evaluate_response(solution_str, item)
    return result
