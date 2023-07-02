<?php

namespace PhpMachinist\Models;

/**
 * SVM is a supervised machine learning algorithm used for classification tasks.
 * It is particularly effective for binary classification, where the goal is to
 * assign data points to one of two classes based on their features.
 *
 * In SVM classification, the algorithm constructs a hyperplane in the feature
 * space that separates the data points of different classes with the largest
 * possible margin. The hyperplane is determined by a subset of training samples
 * called support vectors, which are the closest points to the decision boundary.
 *
 * SVM classifiers are known for their ability to handle high-dimensional data,
 * as they can effectively learn complex decision boundaries in feature spaces of
 * large dimensions. They can also handle non-linear classification tasks by using
 * kernel functions to implicitly map the data into higher-dimensional spaces where
 * linear separation is possible.
 *
 * Linear SVM formula: f(x) = sign(w^T * x + b)
 * Polynomial SVM formula:
 *      - decision function: f(x) = sign(sum(alpha_i * y_i * K(x_i, x) + b)
 *      - kernel function: K(x_i, x) = (gamma * (x_i^T * x) + coef0)^degree
 * RBF SVM formula:
 *      - decision function: f(x) = sign(sum(alpha_i * y_i * K(x_i, x) + b)
 *      - kernel function: K(x_i, x) = exp(-gamma * ||x_i - x||^2)
 */
class SVM
{
    private array $supportVectors;
    private array $weights;
    private int $bias;
    private string $kernel;
    private int $degree;
    private float $gamma;

    public function __construct($kernel = 'linear', $degree = 3, $gamma = 0.5)
    {
        $this->supportVectors = array();
        $this->weights = array();
        $this->bias = 0;
        $this->kernel = $kernel;
        $this->degree = $degree;
        $this->gamma = $gamma;
    }

    /**
     * @param array $samples => An array of ages -> height values
     * [
     *      [18, 160],
     *      [25, 165],
     *      [32, 170],
     *      [45, 175],
     *      [60, 180]
     * ]
     * @param array $labels => An array representing height (0 => short, 1 => tall) [0, 0, 1, 1, 1]
     *
     * @param float $regularization => hyper-parameter that determines the penalty for mis-classification
     * of data points during training. It influences the flexibility of the decision boundary of the SVM classifier.
     *
     * @param float $tolerance => The tolerance parameter is used to determine when the optimization process can
     * be considered converged. If the change in Lagrange multipliers between iterations falls below the specified
     * tolerance, the algorithm stops and considers the optimization converged.
     *
     * @throws \Exception
     */
    public function train(array $features, array $labels, float $regularization = 1.0, float $tolerance = 0.001, int $maxIterations = 100): bool
    {
        $numInstances = count($features);

        // Compute the Gram matrix
        $gramMatrix = array();
        for ($i = 0; $i < $numInstances; $i++) {
            $gramMatrix[$i] = array();
            for ($j = 0; $j < $numInstances; $j++) {
                $gramMatrix[$i][$j] = $this->kernelFunction($features[$i], $features[$j]);
            }
        }


        /**
         * This represents the array of Lagrange multipliers or dual coefficients associated
         * with the support vectors. In SVM, these coefficients determine the importance of
         * each support vector in defining the decision boundary.
         * Values will change during the training process to find optimal solution.
         *
         * $this->alpha => [0, 0, 0, 0, 0]
         */
        $alphas = array_fill(0, $numInstances, 0);

        /**
         * This represents the bias or intercept term in SVM. It is an offset that determines
         * the position of the decision boundary. The bias term allows SVM to handle cases
         * where the data is not linearly separable.
         *
         * Values will change during the training process to find optimal solution.
         */
        $bias = 0;

        $iterations = 0;

        /**
         * The SMO algorithm is a popular and efficient method for training Support Vector Machines.
         * It optimizes the dual problem of the SVM by iteratively selecting pairs of Lagrange multipliers
         * (alpha values) to update, with the objective of maximizing the margin between the support
         * vectors. The SMO algorithm breaks down the optimization problem into a series of smaller
         * sub-problems that can be solved analytically or using numerical optimization techniques.
         *
         * Steps in algorithm:
         * 1. Initialize the Lagrange multipliers alpha and the threshold bias
         * 2. Select two Lagrange multipliers, alpha_i and alpha_j, for update.
         *    These are typically chosen using a heuristic or an optimization strategy.
         * 3. Compute the error E->i and E->j for the selected alpha values, which represent
         *    the difference between the predicted label and the actual label for the
         *    corresponding training samples.
         * 4. Compute the bounds L and H on the selected alpha values, considering the constraint 0 <= alpha <= C,
         *    where C is the regularization parameter.
         * 5. If the bounds L and H are not equal, proceed with the optimization.
         *    a. Compute the second derivative of the objective function with respect to alpha_i and alpha_j.
         *    b. Update the selected alpha_i and alpha_j values based on the optimization rule.
         *    c. Clip the updated alpha_j value to ensure it lies within the bounds L and H.
         *    d. Update the alpha_i value based on the updated alpha_j value.
         *    e. Update the threshold b using the updated alpha values and the corresponding error terms.
         * 6. Check for convergence or termination criteria, such as reaching a maximum number of iterations
         *    or a desired level of accuracy.
         * 7. Repeat steps 2-6 until convergence is achieved or the termination criteria are met.
         */
        while ($iterations < $maxIterations) {
            $numChangedAlphas = 0;

            for ($i = 0; $i < $numInstances; $i++) {
                $error_i = $this->predict($features[$i]) - $labels[$i];
                $alpha_i = $alphas[$i];

                if (($labels[$i] * $error_i < -$tolerance && $alpha_i < $regularization)
                    || ($labels[$i] * $error_i > $tolerance && $alpha_i > 0)) {

                    // Select a random index j != i
                    do {
                        $j = mt_rand(0, $numInstances - 1);
                    } while ($j === $i);

                    $error_j = $this->predict($features[$j]) - $labels[$j];
                    $alpha_j = $alphas[$j];

                    // Compute the bounds for alpha_j
                    if ($labels[$i] === $labels[$j]) {
                        $L = max(0, $alpha_i + $alpha_j - $regularization);
                        $H = min($regularization, $alpha_i + $alpha_j);
                    } else {
                        $L = max(0, $alpha_j - $alpha_i);
                        $H = min($regularization, $regularization + $alpha_j - $alpha_i);
                    }

                    if ($L === $H) {
                        continue;
                    }

                    $eta = 2 * $gramMatrix[$i][$j] - $gramMatrix[$i][$i] - $gramMatrix[$j][$j];

                    if ($eta >= 0) {
                        continue;
                    }

                    $alpha_j_old = $alpha_j;
                    $alpha_j = $alpha_j - ($labels[$j] * ($error_i - $error_j)) / $eta;

                    if ($alpha_j > $H) {
                        $alpha_j = $H;
                    } elseif ($alpha_j < $L) {
                        $alpha_j = $L;
                    }

                    if (abs($alpha_j - $alpha_j_old) < $tolerance) {
                        continue;
                    }

                    $alpha_i_old = $alpha_i;
                    $alpha_i = $alpha_i + $labels[$i] * $labels[$j] * ($alpha_j_old - $alpha_j);

                    // Update the bias term
                    $b1 = $bias - $error_i
                        - $labels[$i] * ($alpha_i - $alpha_i_old) * $gramMatrix[$i][$i]
                        - $labels[$j] * ($alpha_j - $alpha_j_old) * $gramMatrix[$i][$j];

                    $b2 = $bias - $error_j
                        - $labels[$i] * ($alpha_i - $alpha_i_old) * $gramMatrix[$i][$j]
                        - $labels[$j] * ($alpha_j - $alpha_j_old) * $gramMatrix[$j][$j];

                    if ($alpha_i > 0 && $alpha_i < $regularization) {
                        $bias = $b1;
                    } elseif ($alpha_j > 0 && $alpha_j < $regularization) {
                        $bias = $b2;
                    } else {
                        $bias = ($b1 + $b2) / 2;
                    }

                    $alphas[$i] = $alpha_i;
                    $alphas[$j] = $alpha_j;
                    $numChangedAlphas++;
                }
            }

            if ($numChangedAlphas === 0) {
                $iterations++;
            } else {
                $iterations = 0;
            }
        }

        // Store the support vectors and corresponding weights
        $this->supportVectors = array();
        $this->weights = array();

        for ($i = 0; $i < $numInstances; $i++) {
            if ($alphas[$i] > 0) {
                $this->supportVectors[] = $features[$i];
                $this->weights[] = $alphas[$i] * $labels[$i];
            }
        }

        // Compute the bias term
        if (!empty($this->supportVectors)) {
            $this->bias = $bias;
        }

        return true;
    }

    /**
     * @throws \Exception
     */
    public function predict($features): int
    {
        $prediction = $this->bias;

        foreach ($this->supportVectors as $index => $supportVector) {
            $prediction += $this->weights[$index] * $this->kernelFunction($supportVector, $features);
        }

        return $prediction >= 0 ? 1 : -1;
    }

    /**
     * @throws \Exception
     */
    private function kernelFunction(array $x, array $y): float
    {
        if ($this->kernel === 'linear') {
            return $this->linearKernel($x, $y);
        } elseif ($this->kernel === 'polynomial') {
            return $this->polynomialKernel($x, $y);
        } elseif ($this->kernel === 'rbf') {
            return $this->rbfKernel($x, $y);
        } else {
            throw new \Exception("Unsupported kernel: " . $this->kernel);
        }
    }

    /**
     * The linear kernel is used when the data is linearly separable, i.e., when the classes
     * can be separated by a straight line (in 2D) or a hyperplane (in higher dimensions).
     * It is the simplest and most computationally efficient kernel function, as it does not
     * require any complex transformation or mapping of the data into a higher-dimensional space.
     *
     * Formula: A ⋅ B = a1 * b1 + a2 * b2 + ... + an * bn
     */
    private function linearKernel(array $x, array $y): float
    {
        return $this->dotProduct($x, $y);
    }

    /**
     * The polynomial kernel allows SVMs to learn nonlinear decision boundaries by mapping the data
     * into a higher-dimensional feature space. It computes the similarity between two samples as
     * the polynomial expansion of the dot product of their feature vectors. The degree parameter
     * controls the degree of the polynomial, determining the complexity and flexibility of the
     * decision boundary.
     *
     * The polynomial kernel is capable of capturing polynomial relationships between the features
     * and the target variable. By choosing an appropriate degree, it can model different degrees
     * of non-linearity in the data. However, high degrees can lead to over-fitting, so the choice
     * of the degree should be carefully tuned based on the specific problem at hand.
     *
     * The polynomial kernel is a popular choice when dealing with data that exhibits polynomial
     * patterns or when linear separation is not sufficient to achieve good classification performance.
     *
     * Formula: K(x, y) = (gamma * <x, y> + coef0)^degree
     */
    private function polynomialKernel(array$x, array $y): float
    {
        return pow($this->dotProduct($x, $y) + 1, $this->degree);
    }

    /**
     * The RBF kernel allows the model to capture complex nonlinear relationships between the
     * features and the target variable by mapping the data into a higher-dimensional space
     * where linear separation is possible. This allows SVMs with RBF kernels to handle more
     * complex and nonlinear classification problems.
     *
     * Formula: K(x, y) = exp(-gamma * ||x - y||^2)
     */
    private function rbfKernel(array $x, array $y): float
    {
        $euclideanDistanceSquared = $this->euclideanDistanceSquared($x, $y);

        return exp(-$this->gamma * $euclideanDistanceSquared);
    }

    /**
     * The `DotProduct` is a fundamental mathematical equation used in SVM.
     * It is used to calculate the dot product between a feature vector x and
     * a weight vector w. This operation is essential for determining the
     * decision boundary and making predictions in SVM.
     *
     * Formula: A ⋅ B = a1 * b1 + a2 * b2 + ... + an * bn
     */
    private function dotProduct(array $x, array $y): float
    {
        $product = 0;
        $numFeatures = count($x);

        for ($i = 0; $i < $numFeatures; $i++) {
            $product += $x[$i] * $y[$i];
        }

        return $product;
    }

    /**
     * Calculate the euclidean distance
     *
     * Formula: d = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2)
     */
    private function euclideanDistanceSquared(array $x, array $y): float
    {
        $distanceSquared = 0;
        $numFeatures = count($x);

        for ($i = 0; $i < $numFeatures; $i++) {
            $distanceSquared += pow($x[$i] - $y[$i], 2);
        }

        return $distanceSquared;
    }
}
