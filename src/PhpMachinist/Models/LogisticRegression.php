<?php

namespace PhpMachinist\Models;


/**
 * Logistic regression is a statistical algorithm used for binary classification,
 * which means it predicts one of two possible outcomes based on input variables.
 * It's like a decision-making tool that separates things into two categories.
 *
 * Formula: sigmoid(x) = 1 / (1 + exp(-x))
 * 
 * x: It represents the input value to the sigmoid function. In the context of
 * logistic regression, 'x' typically refers to the linear combination of input
 * variables and their corresponding weights.
 *
 * exp(-x): It represents the exponential function applied to the negative of 'x'.
 * The exponential function raises Euler's number 'e' (approximately 2.71828)
 * to the power of '-x', resulting in a positive value.
 *
 * 1 + exp(-x): It is the sum of 1 and the exponential of '-x'. This term is the
 * denominator of the sigmoid function and ensures that the sigmoid output is always positive.
 *
 * 1 / (1 + exp(-x)): It represents the sigmoid function itself. It takes the exponential
 * of '-x', adds 1, and then calculates the reciprocal of the sum. The sigmoid function
 * squashes the 'x' value into the range between 0 and 1, yielding a probability value.
 */
class LogisticRegression
{
    private array $weights;
    private float $intercept;

    /**
     * @param $x => an array of age => height values:
     * [
     *      [18, 165],
     *      [25, 170],
     *      [32, 175],
     *      [45, 180],
     *      [60, 185],
     *      [20, 160],
     *      [28, 168],
     *      [35, 172],
     *      [50, 178],
     *      [65, 190]
     * ];
     *
     * @param $y => array of 0 and 1 representing gender female or male
     *  [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]
     *
     * @param float $learningRate => It represents the learning rate or step size, which determines how
     * much the weights and intercept are updated during each iteration of gradient descent. It controls
     * the speed at which the model learns from the training data. A larger learning rate can result in
     * faster convergence but may risk overshooting the optimal solution, while a smaller learning rate
     * may slow down convergence. It's typically a positive value between 0 and 1.
     *
     * @param int $numIterations =>  It specifies the number of iterations or epochs for which the training
     * process runs. Each iteration updates the weights and intercept based on the calculated gradients.
     * Increasing the number of iterations allows the model to refine its parameters and improve its performance.
     * However, setting a very high number of iterations may lead to over-fitting if the model starts to memorize
     * the training data. The appropriate value depends on the complexity of the problem and the convergence
     * behavior of the model.
     */
    public function train($x, $y, float $learningRate = 0.01, int $numIterations = 100): void
    {
        // 10
        $numSamples = count($x);

        // 2
        $numFeatures = count($x[0]);

        /**
         * Initialize weights and intercept's
         *
         * Weights in machine learning are like sliders that control the importance of different features
         * in making predictions. By adjusting these weights during training, the model learns to give
         * appropriate significance to each feature, ultimately improving its prediction accuracy.
         *
         * We start with default weights [0, 0] for each feature [age, height]. During the training
         * process, we increase the weights for each feature until we get the best weights to keep
         * minimize our error in predicting the gender. If the age weight is higher, then the model
         * will put more weight on the age of the person to determine its gender, if the weight for
         * the height is higher, then it will consider height to be more important.
         *
         * $this->weights => [0, 0] (generated an array of 0 for each $numFeatures)
         */
        $this->weights = array_fill(0, $numFeatures, 0);
        $this->intercept = 0;

        /**
         * Gradient descent
         *
         * The gradient descent algorithm calculates the gradient of the loss function
         * (a measure of the model's prediction error) with respect to each parameter.
         * It then updates the parameters by taking steps in the opposite direction of
         * the gradient, gradually moving towards the optimal set of parameter values
         * that minimize the loss.
         */
        for ($iteration = 0; $iteration < $numIterations; $iteration++) {
            /**
             * Get a prediction for each element of [age/height].
             *
             * First iteration will result in an array containing
             * 10 elements of 0.5 values
             */
            $yPred = $this->predictProbability($x);

            /**
             * We calculate the error so that we can evaluate the performance of the logistic
             * regression model by analyzing the magnitude and direction of the discrepancies
             * between the predicted and actual values. (done against the gender array)
             *
             * For first iteration most calculations will be either (0.5 - 0) or (0.5 - 1)
             *
             * Resulting array: [0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5]
             */
            $error = array_map(function ($y, $yPred) {
                return $yPred - $y;
            }, $y, $yPred);

            /**
             * Intercept upgrade step in the gradient descent.
             * formula: new_intercept = old_intercept - learning_rate * (1 / num_samples) * sum(errors)
             *
             * First iteration will result in: 0 -= 0.01 * (0 / 10) => 0
             */
            $this->intercept -= $learningRate * (array_sum($error) / $numSamples);

            /**
             * Adjust weights to lower error
             */
            for ($i = 0; $i < $numFeatures; $i++) {
                /**
                 * Calculations for first iteration will be:
                 *
                 * $gradient = [
                 *      0.5 * 18 => 9,
                 *      0.5 * 25 => 12.5,
                 *      0.5 * 32 => 16,
                 *      -0.5 * 45 => -22.5,
                 *      -0.5 * 60 => -30,
                 *      0.5 * 20 => 10,
                 *      0.5 * 28 => 14,
                 *      -0.5 * 35 => -17.5,
                 *      -0.5 * 50 => -25,
                 * ]
                 */
                $gradient = array_map(function ($error, $x) use ($i) {
                    return $error * $x[$i];
                }, $error, $x);

                /**
                 * Weights upgrade step in the gradient descent.
                 * formula: new_intercept = old_intercept - learning_rate * (1 / num_samples) * sum(errors)
                 *
                 * First iteration will result in
                 * $this->weights = [
                 *      0.01 * (-66 / 10) => 0.066,
                 *      0.01 * (-33.5 / 10) => 0.0335,
                 * ]
                 */
                $this->weights[$i] -= $learningRate * (array_sum($gradient) / $numSamples);
            }
        }
    }

    public function predictProbability($x): array
    {
        $yPred = [];
        foreach ($x as $sample) {

            /**
             * A logit, also known as the log-odds, represents the logarithm of the odd ratio. It is the
             * transformed output of the linear regression equation, which is then passed through the
             * sigmoid function to obtain predicted probabilities.
             *
             * Formula: logit(p) = log(p / (1 - p))
             */
            $logits = $this->intercept;
            for ($i = 0; $i < count($sample); $i++) {
                /**
                 * The logit will be the (current weight * age += current weight * height)
                 */
                $logits += $this->weights[$i] * $sample[$i];
            }

            /**
             * The sigmoid function is commonly used in logistic regression to map the log-odds or
             * logits to a probability value between 0 and 1.
             *
             * Formula: sigmoid(x) = 1 / (1 + exp(-x))
             */
            $sigmoid = 1 / (1 + exp(-$logits));
            $yPred[] = $sigmoid;
        }

        return $yPred;
    }

    public function predict($x, $threshold = 0.5): array
    {
        $yPred = $this->predictProbability($x);

        return array_map(function ($probability) use ($threshold) {
            return $probability >= $threshold ? 1 : 0;
        }, $yPred);
    }

    public function getWeights(): array
    {
        return $this->weights;
    }

    public function getIntercept(): float
    {
        return $this->intercept;
    }
}
