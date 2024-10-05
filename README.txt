=== Supplementary Materials for <Prompt-Guided Retraining-Dataset Generation to Mitigate Biases in Language Models> ===

# To install the project dependencies listed in the requirements.txt file, execute the following command:
    pip install -r requirements.txt

# When running each file, please carefully verify the paths provided as arguments. Ensure that the paths are accurate and point to the intended files or directories.

# This repository contains code for the following processes.

    1. Creating bias-detecting prompts 
        python preprocess/preprocess.py
        python preprocess/get_prompts.py

    2. Identifying Bias
        bash bias-detect/get_triplets.sh

    3. Dataset Generation with Open AI API
        # Please ensure that you have obtained the necessary API keys. 
        python generate/generate_sentence.py
        # Or, you can just use the json file which includes generated sentences
        ./generate/generated_sentence.json

    4. Retraining the model 
        bash train/run_eps_cutoff.sh 

# To load the retrained model's checkpoint, you can use the following code as an example:

    model = getattr(models, args.model)(args.model_name_or_path)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)

# For evaluating the model obtained from the aforementioned process, you can utilize the "bias-bench" implementation. 

    git clone https://github.com/McGill-NLP/bias-bench.git
