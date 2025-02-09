from contrastive_ner import *

class ContrastiveNEROneShot(ContrastiveNER):

    async def one_type_epoch_training(self,fine_type):
        good_similarities = []
        bad_similarities = []
        losses = []
        classifier_accuracies = []
        classifier_losses = []
        async for anchor, same_type, different_type in fewnerd_processor.yield_train_dataset(
                anchor_type=fine_type,
                batch_size=self.args.batch_size,
                instances_per_type=self.args.instances_per_type,
                hard_negative_ratio=self.args.hard_negative_ratio,
                llm_layer=self.args.llm_layer):
            self.optimizer.zero_grad()
            anchor, good_batch, bad_batch = self.tensorify(anchor, same_type, different_type)
            good_similarity, bad_similarity, similarity_loss = self.compute_similarity(
                anchor,
                good_batch,
                bad_batch).values()
            classifier_accuracy, classifier_loss = self.compute_accuracy(anchor, good_batch, bad_batch).values()
            self.optimizer.step()

            good_similarities.append(good_similarity)
            bad_similarities.append(bad_similarity)
            losses.append(similarity_loss + classifier_loss)
            classifier_accuracies.append(classifier_accuracy)
            classifier_losses.append(classifier_loss)

        return {
            "good_similarity": good_similarities,
            "bad_similarity": bad_similarities,
            "similarity_loss": losses,
            "classifier_accuracy": classifier_accuracies,
            "classifier_loss": classifier_losses
        }


    async def one_type_epoch_evaluation(self, fine_type):
        good_similarities = []
        bad_similarities = []
        losses = []
        classifier_accuracies = []
        predictions = []
        ground_truth = []
        async for anchor, same_type, different_type in fewnerd_processor.yield_test_dataset(anchor_type=fine_type,
                                                                                            batch_size=self.args.batch_size,
                                                                                            instances_per_type=self.args.instances_per_type,
                                                                                            llm_layer=self.args.llm_layer):
            anchor, good_batch, bad_batch = self.tensorify(anchor, same_type, different_type)

            good_similarity, bad_similarity, similarity_loss = self.compute_similarity(
                anchor,
                good_batch,
                bad_batch).values()
            classifier_accuracy, classifier_loss = self.compute_accuracy(anchor, good_batch, bad_batch).values()

            anchor_mlp = self.forward_similarity_model(anchor, compute_grad=False, detach=False)
            good_batch_mlp = self.forward_similarity_model(good_batch, compute_grad=False, detach=False)
            bad_batch_mlp = self.forward_similarity_model(bad_batch, compute_grad=False, detach=False)

            predictions.extend(torch.cosine_similarity(anchor_mlp, good_batch_mlp, dim=1).cpu().tolist())
            predictions.extend(torch.cosine_similarity(anchor_mlp, bad_batch_mlp, dim=1).cpu().tolist())

            ground_truth.extend([1] * len(good_batch_mlp))
            ground_truth.extend([0] * len(bad_batch_mlp))

            losses.append(classifier_loss + similarity_loss)
            good_similarities.append(good_similarity)
            bad_similarities.append(bad_similarity)
            classifier_accuracies.append(classifier_accuracy)




        return {
            "good_similarity": good_similarities,
            "bad_similarity": bad_similarities,
            "loss": losses,
            "classifier_accuracy": classifier_accuracies,
            "predictions": predictions,
            "ground_truth": ground_truth
        }


if __name__ == "__main__":
    experiment = ContrastiveNEROneShot()
    experiment.main()