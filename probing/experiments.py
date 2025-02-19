import os
import pickle

import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
import tqdm
import helpers
from notebooks.helpers import get_dif, load_ratings, get_top_attributes, attribute2class


def get_stereotype_strength(ratio_df):
    attribute_name = "katz"
    attributes = helpers.load_attributes(attribute_name)
    k = 5
    attribute2score = load_ratings("katz")
    # load top 5 attributes
    stereo_attributes = get_top_attributes(
        attributes,
        attribute2score,
        k
    )
    ratio_df["attribute_class"] = ratio_df.attribute.apply(
        lambda x: attribute2class(x, stereo_attributes)
    )
    dif = get_dif(
        ratio_df[
            (ratio_df.attribute_class == "general")
            ],
        ratio_df[
            (ratio_df.attribute_class == "stereo")
            ]
    )
    return dif


def probe(model_name, model, tok, device, variable_pairs, prompts, attributes):
    # Prepare list to store results
    ratio_list = []

    # Evaluation loop
    model.eval()
    with torch.no_grad():

        # Loop over prompts
        for prompt in prompts:
            print(f"Processing prompt: {prompt}")

            # Compute prompt-specific results
            results = []
            for variable_pair in tqdm.tqdm(variable_pairs):
                variable_aae, variable_sae = variable_pair.strip().split("\t")

                # Compute probabilities for attributes after AAE text
                probs_attribute_aae = helpers.get_attribute_probs(
                    prompt.format(variable_aae),
                    attributes,
                    model,
                    model_name,
                    tok,
                    device,
                    labels=None
                )

                # Compute probabilities for attributes after SAE text
                probs_attribute_sae = helpers.get_attribute_probs(
                    prompt.format(variable_sae),
                    attributes,
                    model,
                    model_name,
                    tok,
                    device,
                    labels=None
                )

                # Loop over attributes
                for a_idx in range(len(attributes)):
                    # Compute log probability ratio
                    log_prob_ratio = np.log10(
                        probs_attribute_aae[a_idx] /
                        probs_attribute_sae[a_idx]
                    )

                    # Store result
                    ratio_list.append((
                        log_prob_ratio,
                        variable_sae,
                        attributes[a_idx][1:],
                        prompt
                    ))

    ratio_df = pd.DataFrame(
        ratio_list,
        columns=["ratio", "variable", "attribute", "prompt"]
    )
    return ratio_df


def replace_sentences(variable_pairs):
    new_variable_pairs = []
    for variable_pair in variable_pairs:
        variable_0, variable_1 = variable_pair.split("\t")
        variable_0 = variable_1.upper()
        new_variable_pairs.append(f"{variable_0}\t{variable_1}")
    return new_variable_pairs


def main():
    model_name = "gpt2-large"
    variable = "groenwold"  # groenwold
    attribute_name = "katz"
    model = helpers.load_model(model_name)
    tok = helpers.load_tokenizer(model_name)
    # If possible, move model to GPU
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = model.to(device)

    # Load AAE and SAE texts (minimal pairs)
    variable_pairs = helpers.load_pairs(variable)
    # Load attributes
    attributes = helpers.load_attributes(attribute_name, tok)


    # Load prompts
    prompts, _ = helpers.load_prompts(
        model_name,
        attribute_name,
        variable
    )
    if True:
        variable_pairs = variable_pairs[:20]
        variable = variable + 'N20'

    if True:
        variable_pairs = replace_sentences(variable_pairs)
        variable = variable + '_upper'

    ratio_df = probe(model_name, model, tok, device, variable_pairs, prompts, attributes)
    attribute_ratios = ratio_df.groupby([
        "attribute",
    ], as_index=False)["ratio"].mean()
    print(attribute_ratios.sort_values(by="ratio", ascending=False))
    with open(f"{helpers.PROBS_PATH}{os.path.sep}{model_name}_{variable}_{attribute_name}.p", "wb") as f:
        pickle.dump(ratio_df, f)
    dif = get_stereotype_strength(ratio_df)
    print(dif)


if __name__ == '__main__':
    main()