
def find_optimal_threshold(y_score, y_true):
    """
    Finds the optimal threshold for maximum accuracy.

    Args:
      y_true: True labels (binary: 0 or 1).
      y_score: Predicted probabilities.

    Returns:
      Optimal threshold.
    """

    n = len(y_true)
    thresholds = sorted(y_score)

    # Initialize with the first threshold
    best_threshold = thresholds[0]
    best_accuracy = 0

    # Calculate initial accuracy
    y_pred = [1 if prob >= best_threshold else 0 for prob in y_score]
    correct_predictions = sum(y_true[i] == y_pred[i] for i in range(n))
    accuracy = correct_predictions / n

    # Iterate through thresholds
    for i in range(1, n):
        threshold = thresholds[i]
        y_pred = [1 if prob >= threshold else 0 for prob in y_score]
        correct_predictions = sum(y_true[i] == y_pred[i] for i in range(n))
        new_accuracy = correct_predictions / n

        if new_accuracy > accuracy:
            best_threshold = threshold
            accuracy = new_accuracy

    return best_threshold, accuracy
