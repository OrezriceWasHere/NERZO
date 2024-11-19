import clearml_poc
import llm_interface
from contrastive.args import Arguments
from contrastive.mlp import ContrastiveMLP, Detector
from contrastive.loss import ContrastiveLoss
import torch
from contrastive import fewnerd_processor
import torch.nn.functional as F
from tqdm import trange
from peft import LoraConfig, get_peft_model, PeftType


def main():
    for e in trange(args.epochs):
        train(e)
        evaluate(e)


def avg(l):
    return sum(l) / len(l)


def train(epoch):

    losses = []
    good_similarities = []
    bad_similarities = []
    classifier_accuracies = []

    optimizer.zero_grad()

    for anchor, same_type, different_type in fewnerd_processor.yield_train_dataset(batch_size=args.batch_size, instances_per_type=args.instances_per_type):
        anchor, good_batch, bad_batch = pick_llm_output(anchor, same_type, different_type)

        good_similarity, bad_similarity, similarity_loss = compute_similarity(
            anchor,
            good_batch,
            bad_batch).values()
        classifier_accuracy, classifier_loss = compute_accuracy(anchor, good_batch, bad_batch).values()

        losses.append(classifier_loss + similarity_loss)
        good_similarities.append(good_similarity)
        bad_similarities.append(bad_similarity)
        classifier_accuracies.append(classifier_accuracy)

    optimizer.step()
    log_training_metrics(epoch, avg(losses), avg(good_similarities), avg(bad_similarities),
                         avg(classifier_accuracies),
                         series="train")


def evaluate(epoch):
    with torch.no_grad():
        losses = []
        good_similarities = []
        bad_similarities = []
        classifier_accuracies = []

        for anchor, same_type, different_type in fewnerd_processor.yield_test_dataset(batch_size=args.batch_size, instances_per_type=args.instances_per_type):
            anchor, good_batch, bad_batch = pick_llm_output(anchor, same_type, different_type)

            good_similarity, bad_similarity, similarity_loss = compute_similarity(
                anchor,
                good_batch,
                bad_batch).values()
            classifier_accuracy, classifier_loss = compute_accuracy(anchor, good_batch, bad_batch).values()

            losses.append(classifier_loss + similarity_loss)
            good_similarities.append(good_similarity)
            bad_similarities.append(bad_similarity)
            classifier_accuracies.append(classifier_accuracy)

        log_training_metrics(epoch, avg(losses), avg(good_similarities), avg(bad_similarities),
                             avg(classifier_accuracies),
                             series="eval")

def forward_pass(dataset_document: dict | list[dict]) -> list[torch.Tensor]:
    if isinstance(dataset_document, dict):
        dataset_document = [dataset_document]
    texts = [doc["all_text"] for doc in dataset_document]
    texts_indices = [(doc["index_start"], doc["index_end"]) for doc in dataset_document]
    tokens = llm.tokenize(texts)
    hidden_items = llm.get_llm_at_layer(tokens, layer, clone=False)
    token_indices = [llm.token_indices_given_text_indices(text, text_indices) for text, text_indices in zip(texts, texts_indices)]
    end = [h[token_index[1]] for h, token_index in zip(hidden_items, token_indices)]
    return end


def pick_llm_output(*items):
    tensorify = lambda item: forward_pass(item)
    stack = lambda batch: torch.stack(tensorify(batch))

    return list(map(stack, items))


def compute_accuracy(anchor, good_batch, bad_batch):
    anchor_forward = anchor.to(device).detach().clone()
    good_batch_forward = good_batch.to(device).detach().clone()
    bad_batch_forward = bad_batch.to(device).detach().clone()

    accuracies, losses = [], []
    labels = torch.tensor([1] * len(good_batch_forward) + [0] * len(bad_batch_forward)).to(device)
    true_classification_prediction = classifier_model(anchor_forward, good_batch_forward)
    false_classfication_prediction = classifier_model(anchor_forward, bad_batch_forward)
    classification_predication = torch.cat((true_classification_prediction, false_classfication_prediction), dim=0)
    classifier_loss = classifier_criterion(classification_predication, labels)
    classifier_loss.backward()
    accuracy = torch.sum(torch.argmax(classification_predication, dim=-1) == labels).item() / \
               classification_predication.shape[0]
    accuracies.append(accuracy)
    losses.append(classifier_loss.item())

    return {
        "accuracy": avg(accuracies),
        "loss": avg(losses)
    }

def compute_similarity(
        anchor,
        positive_examples,
        negative_examples):

    good_similarities = []
    bad_similarities = []
    losses = []

    loss = similarity_criterion(anchor, positive_examples, negative_examples)
    loss.backward()

    losses.append(loss.item())
    good_similarities.append(F.cosine_similarity(anchor, positive_examples, dim=-1).mean().item())
    bad_similarities.append(F.cosine_similarity(anchor, negative_examples, dim=-1).mean().item())

    return {
        "good_similarity": avg(good_similarities),
        "bad_similarity": avg(bad_similarities),
        "loss": avg(losses)
    }


def log_training_metrics(index, similarity_loss, good_similarity, bad_similarity, accuracy, series):
    clearml_poc.add_point_to_graph(title="similarity_loss", series=series, x=index, y=similarity_loss)
    clearml_poc.add_point_to_graph(title="good_similarity", series=series, x=index, y=good_similarity)
    clearml_poc.add_point_to_graph(title="bad_similarity", series=series, x=index, y=bad_similarity)
    clearml_poc.add_point_to_graph(title="accuracy", series=series, x=index, y=accuracy)


if __name__ == "__main__":
    clearml_poc.clearml_init()
    assert torch.cuda.is_available(), "no gpu available"
    args: Arguments = Arguments()
    clearml_poc.clearml_connect_hyperparams(args)

    LLM_ID = "meta-llama/Meta-Llama-3.1-8B"
    llm_id, layer = "meta-llama/Meta-Llama-3.1-8B", "base_model.model.model.layers.17.self_attn.v_proj"
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of the LoRA matrices
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,  # Dropout probability
        peft_type=PeftType.LORA,

    )


    llm = llm_interface.LLMInterface(llm_id=llm_id, interested_layers=[layer], lora_config=lora_config)
    model = llm.model

    # Make sure only LoRA parameters are trainable
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient computation for LoRA-specific parameters
    for name, param in model.named_parameters():
        if "lora" in name:  # Adjust to your naming scheme
            param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    similarity_criterion = ContrastiveLoss(loss_fn=args.loss_fn, margin=args.triplet_loss_margin)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_model = Detector().to(device)
    classifier_criterion = torch.nn.CrossEntropyLoss()




    main()
