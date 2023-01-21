# Leveraging Training Data for PAL prompting

Our code is adapted from the PAL paper: program-aided language model.
## Usage

1. Obtain the sentence representations

```bash
python3 -m preprocess.sent_embedding --dataset=gsm8k
```

2. Run the prompting

```bash
python3  scripts/eval_basedon_sim.py --dataset=gsm8k --similarity_order=most_similar
```

By default, we use the openAI sentence representation model. 
