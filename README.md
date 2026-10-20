# slm_ai_trainedcobnox
Trained SLM Cobnox

Author: Khadija
Repository: "trained_slm_cobnox"
Technology: Cobnox Programming Language
Project: Small Language Model (SLM)

1. Project Overview

I am Khadija, and I developed this Small Language Model (SLM) using the Cobnox programming language.

This project focuses on training a small AI language model using AI-related question-and-answer data. The dataset contains 70 examples covering Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, Tokenization, Transformers, RAG, and other AI concepts.

The model uses a two-stage training process:

1. Pre-training on an AI knowledge corpus.
2. Supervised Fine-Tuning (SFT) using question-and-answer pairs.

2. Model Configuration

Model Type: Small Language Model (SLM)
Training Examples: 70
Pre-training Epochs: 2
SFT Training Steps: 65
SFT Epochs: 1
Learning Rate: 0.002
Model Format: .cbxm
Tokenizer Format: .json

Model configuration used in Cobnox:

ai = create_slm2_engine(512, 64, 128, 0.002)

3. Training Results

The following results were reported during the training run:

Pre-training Loss:
0.0921

SFT Loss:
0.0891

The SFT loss was lower than the reported pre-training loss. Further evaluation is required to measure the model's accuracy and ability to answer unfamiliar questions.

4. Execution Time

The model's training and execution process takes approximately 2 minutes in the current environment.

Actual execution time may vary depending on hardware, system resources, and training configuration.

5. Sample Model Output

Ask trained model a question:
Who trained you?

i am trained by khadija.

Model Loaded

The model's training data includes the response:

Who trained you?
I am trained by Khadija.

6. Model Files

The project saves the following files:

stage1_base.cbxm
stage1_tok.json
stage2_sft.cbxm
stage2_tok.json

- "stage1_base.cbxm": Saved pre-trained model parameters.
- "stage1_tok.json": Tokenizer saved after pre-training.
- "stage2_sft.cbxm": Saved model parameters after supervised fine-tuning.
- "stage2_tok.json": Tokenizer saved after SFT.

The model and tokenizer must use compatible vocabulary and token-ID mappings.

7. Code Example

The following code demonstrates the main training workflow. The complete 70-example dataset and prompt-response pairs are part of the project source code.

use "sys"
use "engine"

ai = create_slm2_engine(512, 64, 128, 0.002)

pre_loss = engine_pretrain_corpus(ai, arr, 80, 2)

print("Pre-training Loss:")
print(pre_loss)

save_model("stage1_base.cbxm", ai.model.params, "COBNOX_BASE")
tokenizer_save(ai.tokenizer, "stage1_tok.json")

ai2 = create_slm2_engine(512, 64, 128, 0.002)

load_model("stage1_base.cbxm", ai2.model.params)
ai2.tokenizer = tokenizer_load("stage1_tok.json")

slm2_fit_vocab(ai2, prompts)
slm2_fit_vocab(ai2, responses)

sft_loss = engine_train_sft_pairs(ai2, prompts, responses, 65, 1)

print("SFT Loss:")
print(sft_loss)

save_model("stage2_sft.cbxm", ai2.model.params, "COBNOX_SFT")
tokenizer_save(ai2.tokenizer, "stage2_tok.json")

ai4 = create_slm2_engine(512, 64, 128, 0.002)

load_model("stage2_sft.cbxm", ai4.model.params)
ai4.tokenizer = tokenizer_load("stage2_tok.json")

print("Model Loaded")

running = 1

while running == 1 do
    q = input("User: ")

    if q == "stop chat" do
        running = 0
    otherwise do
        if length(q) > 0 do
            reply = engine2_chat(ai4, q)
            print("\nBot: ", reply)
        end
    end
end

Note: This is the main workflow example, not the complete training source. The full dataset contains 70 examples. Confirm the loop syntax and tokenizer-handling functions against your installed Cobnox engine before running.

8. Known Code Issues

The original code contains some potential issues that need verification:

1. Tokenizer compatibility: The tokenizer vocabulary must remain compatible with the model weights after SFT.
2. Incorrect model reference: The original final chat loop loads the fine-tuned model into "ai4" but calls "engine2_chat(ai, q)". The intended model reference should be checked; the cleaned-up example uses "ai4".
3. Loop termination: The original code uses "running == 0", which appears to be a comparison rather than an assignment. The correct assignment syntax depends on Cobnox.
4. Training evaluation: The reported loss values alone do not establish the model's accuracy or generalization.

These are code-review findings; they have not all been confirmed as runtime errors.

9. Future Improvements

- Add more high-quality AI training examples.
- Test answers to questions that are not included in the training dataset.
- Verify tokenizer and model compatibility.
- Improve response formatting and consistency.
- Measure inference speed and model accuracy.

10. Conclusion

This project demonstrates a basic Small Language Model training workflow using Cobnox, including pre-training, supervised fine-tuning, model saving, tokenizer saving, and chat inference.

Developed by Khadija.
