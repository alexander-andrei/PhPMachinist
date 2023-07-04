<?php

namespace PhpMachinist\Models;

/**
 * The Multinomial Naive Bayes algorithm is a Bayesian learning approach
 * popular in Natural Language Processing (NLP). The program guesses the
 * tag of a text, such as an email or a newspaper story, using the Bayes
 * theorem. It calculates each tag's likelihood for a given sample and
 * outputs the tag with the greatest chance.
 *
 * The Naive Bayes method is a strong tool for analyzing text input and
 * solving problems with numerous classes. Because the Naive Bayes theorem
 * is based on the Bayes theorem, it is necessary to first comprehend the
 * Bayes theorem notion. The Bayes theorem, which was developed by Thomas Bayes
 * estimates the likelihood of occurrence based on prior knowledge of the
 * event's conditions. When predictor B itself is available, we calculate
 * the likelihood of class A.
 *
 * Formula: P(A|B) = P(A) * P(B|A)/P(B).
 */
class MultinomialNaiveBayes
{
    private array $classes = [];
    private array $classCounts = [];
    private array $featureCounts = [];

    /**
     * @param array $dataset => array of dataset representing conditions to play Tennis or not
     *  $dataset = [
     *     [1, 'Sunny', 'Hot', 'High', 'Weak'],
     *     [2, 'Sunny', 'Hot', 'High', 'Strong'],
     *     [3, 'Overcast', 'Hot', 'High', 'Weak'],
     *     [4, 'Rain', 'Mild', 'High', 'Weak'],
     *  ]
     *
     * @param array $labels => array representing actions for each condition [no => don't play tennis, yes => play tennis]
     *  $labels = ['no', 'no', 'yes', 'yes]
     */
    public function train(array $dataset, array $labels): void
    {
        foreach ($dataset as $datasetKey => $features) {
            $class = $labels[$datasetKey];

            // creates number of classes (for the case mentioned above it will only be [yes, no]
            if (!isset($this->classes[$class])) {
                $this->classes[$class] = [];
                $this->classCounts[$class] = 0;
            }

            // increase count for how many Yes/No label exists ['no' => 5, 'yes' => 9]
            $this->classCounts[$class]++;

            /**
             * Calculate how many feature are for each label
             *
             * Short example (example shows how many features [key, sunny, rain] exists for yes and no labels):
             * [
             *      'no' => [
             *          [
             *              1 => 1,
             *              6 => 1,
             *              14 => 1,
             *          ],
             *              'sunny' => 3,
             *              'rain' => 2,
             *          [
             *      ],
             *      'yes' => [
             *          [
             *              3 => 1,
             *              4 => 1,
             *          ],
             *              'sunny' => 2,
             *              'rain' => 3,
             *          [
             *      ]
             * ]
             */
            foreach ($features as $featureKey => $feature) {
                if (!isset($this->featureCounts[$class][$featureKey][$feature])) {
                    $this->featureCounts[$class][$featureKey][$feature] = 0;
                }

                $this->featureCounts[$class][$featureKey][$feature]++;
            }
        }
    }

    public function predict(array $features): string
    {
        $bestClass = null;
        $bestScore = PHP_INT_MIN;

        // Calculate the class score for each class
        foreach ($this->classes as $class => $classFeatures) {
            /**
             * Calculate logarithm of the probability of a class occurring in the dataset.
             *
             * Formula: log(P(class)) = log(count(class) / sum(counts))
             */
            $classScore = log($this->classCounts[$class] / array_sum($this->classCounts));

            /**
             * Laplace smoothing, also known as additive smoothing or pseudo-count smoothing
             * is a technique used in probability theory and statistics to handle the issue
             * of zero probabilities in probability calculations.
             *
             * In such cases, if we were to use a feature count of 0 in the denominator of
             * the conditional probability calculation, it would result in a probability of 0
             * which can be problematic for subsequent calculations.
             *
             * To avoid this issue, Laplace smoothing adds a pseudo-count of 1 to both the
             * numerator and the denominator. By adding 1 to the numerator we ensure that the
             * probability estimate is not 0, even when a feature value has not been observed
             * in a specific class.
             *
             * Formula: P(x|c) = (count(x, c) + 1) / (count(c) + |V|)
             *
             * P(x|c) => represents the probability of feature x given class c.
             * count(x, c) => is the number of times feature x appears in instances of class c.
             * count(c) => is the total count of instances in class c.
             * |V| => is the total number of unique features in the entire dataset.
             */
            foreach ($features as $key => $feature) {
                if (isset($this->featureCounts[$class][$key][$feature])) {
                    $featureCount = $this->featureCounts[$class][$key][$feature];
                    $classScore += log(
                        ($featureCount + 1) / (array_sum($this->featureCounts[$class][$key]) + count($this->featureCounts[$class][$key]))
                    );
                } else {
                    $classScore += log(1 / (array_sum($this->featureCounts[$class][$key]) + count($this->featureCounts[$class][$key]) + 1));
                }
            }

            if ($classScore > $bestScore) {
                $bestScore = $classScore;
                $bestClass = $class;
            }
        }

        return $bestClass;
    }
}