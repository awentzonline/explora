import click
from lightning.pytorch import seed_everything
import numpy as np
from peft import LoraConfig, get_peft_model
import ray
from ray.util import ActorPool
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from explora.datasets.hf_datasets import HfDatasetDataModule


class ESState:
    def __init__(self, weights_size, population_size, seed, init_weights=None, sigma=0.002, lr=0.0001):
        self.rng = np.random.default_rng(seed=seed)
        self.pivot_weights = init_weights or np.zeros(weights_size)
        self.population_size = population_size
        self.weights_size = weights_size
        self.sigma = sigma
        self.lr = lr

    def update_perturbations(self):
        self.perturbations = self.rng.normal(0, 1, (self.population_size, self.weights_size))

    def update_with_scores(self, scores):
        z_scores = (scores - np.mean(scores)) / np.std(scores)
        weight_update = np.matmul(self.perturbations.T, z_scores)
        scaled_weight_update = self.lr / (self.population_size * self.sigma) * weight_update
        self.pivot_weights = self.pivot_weights + scaled_weight_update

    def get_perturbed(self, i):
        return self.pivot_weights + self.sigma * self.perturbations[i]


@ray.remote
class ESEvaluator:
    def __init__(
        self, model_name, dataset_name, population_size, seed=42, batch_size=16, max_length=64,
        sigma=0.002, lr=0.0001, dataset_split='train',
    ):
        seed_everything(seed)
        print('loading model...')
        self.model, self.tokenizer = get_model(model_name)
        self.model.eval()
        print('loading dataset...')
        self.dataset_split = dataset_split
        self.dataset = HfDatasetDataModule(
            dataset_name, tokenizer=self.tokenizer, batch_size=batch_size,
            max_seq_len=max_length,
        )
        self.dataset.setup('eval')
        print('loaded.')
        weights = get_model_weights(self.model)
        self.es_state = ESState(len(weights), population_size, seed, sigma=sigma, lr=lr)

    def update_perturbations(self):
        self.es_state.update_perturbations()

    def evaluate(self, worker_id, max_batches=None):
        if worker_id is None:
            model_weights = self.es_state.pivot_weights
        else:
            model_weights = self.es_state.get_perturbed(worker_id)
        self.set_weights(model_weights)

        # evaluate model perplexity on dataset
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            data_loader = getattr(self.dataset, f'{self.dataset_split}_dataloader')()
            for batch_i, batch in enumerate(data_loader):
                if max_batches and max_batches <= batch_i:
                    break
                input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=input_ids
                )
                num_tokens = attention_mask.sum().item()  # Count the number of non-padding words in the batch
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens

        # score = -np.exp(total_loss / total_tokens)
        score = -total_loss / total_tokens
        return score

    def update_state_with_scores(self, scores):
        self.es_state.update_with_scores(scores)

    def set_weights(self, weights):
        set_model_weights(self.model, weights)
        self.num_weights = len(weights)

    def get_weights(self):
        return get_model_weights(self.model)

    def get_pivot_weights(self):
        return self.es_state.pivot_weights


def get_model(model_name, r=2, lora_alpha=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["c_attn", "q_attn"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model, tokenizer


def set_model_weights(model, weights):
    i = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            param_size = param.numel()
            param_weights = weights[i: i + param_size]
            param_weights = param_weights.reshape(*param.shape)
            param.data.copy_(torch.from_numpy(param_weights))
            i += param_size


def get_model_weights(model):
    weights = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            weights.append(param.data.detach().cpu().view(-1).numpy())
    weights = np.concatenate(weights)
    return weights


def evolution_strategies(
    model_name, dataset_name='wikitext/wikitext-2-raw-v1', population_size=5, num_actors=2, max_epochs=5,
    sigma=0.0001, lr=0.0001, max_length=64, batch_size=16, max_batches=None
):
    metric_logger = SummaryWriter()
    eval_actors = [
        ESEvaluator.remote(
            model_name, dataset_name, population_size,
            max_length=max_length, batch_size=batch_size,
            sigma=sigma, lr=lr,
        )
        for _ in range(num_actors)
    ]
    eval_pool = ActorPool(eval_actors)

    max_epochs = max_epochs or np.inf
    epoch_i = 0
    try:
        while epoch_i < max_epochs:
            t0 = time.time()
            # generate new offsets
            for actor in eval_actors:
                actor.update_perturbations.remote()

            # evaluate the current weights on tasks
            scores = np.array(list(eval_pool.map(
                lambda actor, i: actor.evaluate.remote(i, max_batches=max_batches),
                range(population_size)
            )), dtype=np.single)
            # check the score of the pivot weight
            pivot_score = ray.get(eval_actors[0].evaluate.remote(None, max_batches=max_batches))

            # Update state on actors
            scores_ref = ray.put(scores)
            for actor in eval_actors:
                actor.update_state_with_scores.remote(scores_ref)

            score_metrics = dict(
                scores_min=np.min(scores), scores_mean=np.mean(scores), scores_std=np.std(scores),
                scores_max=np.max(scores), pivot_score=pivot_score,
            )
            metric_logger.add_scalars('score', score_metrics, epoch_i)
            epoch_time = time.time() - t0
            print(f'{epoch_i=} {epoch_time=:.2f}s', score_metrics)
            epoch_i += 1
    except KeyboardInterrupt:
        print('Ending search...')
    pivot_weights = eval_actors[0].get_pivot_weights.remote()
    return pivot_weights


def model_repl(model, tokenizer, max_length=None, top_p=None):
    try:
        while True:
            prompt = input('prompt> ')
            if not prompt:
                continue
            tokens = tokenizer(prompt, return_tensors='pt')
            kwargs = {}
            if top_p:
                kwargs['top_p'] = top_p
                kwargs['top_k'] = 0
                kwargs['do_sample'] = True
            with torch.no_grad():
                probs = model(**tokens, labels=tokens['input_ids'])
                print('loss', probs.loss)
                gen = model.generate(
                    **tokens, max_length=max_length, **kwargs
                )
                p_tokens = gen[0]
            decoded = tokenizer.decode(p_tokens, skip_special_tokens=True)
            print(decoded)
    except KeyboardInterrupt:
        pass


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


@click.command()
@click.option('--model', default='gpt2')
@click.option('--dataset-name', default='wikitext:wikitext-2-raw-v1')
@click.option('--max-length', default=128, type=int)
@click.option('--top-p', default=None, type=float)
@click.option('--max-epochs', default=5, type=int)
@click.option('--lr', default=0.00001, type=float)
@click.option('--sigma', default=0.00001, type=float)
@click.option('--batch-size', default=16, type=int)
@click.option('--population-size', default=5, type=int)
@click.option('--num-actors', default=2, type=int)
@click.option('--max-batches', default=None, type=int)
def main(
    model, dataset_name,
    max_length, top_p, max_epochs, lr, sigma, batch_size, population_size, num_actors, max_batches
):
    best_weights = ray.get(evolution_strategies(
        model, dataset_name=dataset_name, max_epochs=max_epochs, lr=lr, batch_size=batch_size,
        population_size=population_size, num_actors=num_actors, max_batches=max_batches,
        sigma=sigma,
    ))

    # check out the best model
    print('Loading model...')
    lora_model, tokenizer = get_model(model)
    set_model_weights(lora_model, best_weights)
    model_repl(lora_model, tokenizer, max_length=max_length, top_p=top_p)


if __name__ == '__main__':
    main()
