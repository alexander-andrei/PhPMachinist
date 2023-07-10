<?php

namespace PhpMachinist\Models;

/**
 * K-nearest neighbor is a supervised machine learning algorithm used for
 * classification and regression tasks. It is a non-parametric algorithm
 * meaning it does not make any assumptions about the underlying data distribution.
 *
 * In K-nearest neighbor, the "K" represents the number of nearest neighbors that
 * are considered for making predictions. The algorithm works based on the principle
 * that if a majority of the k nearest neighbors of a data point belong to a certain
 * class, then that data point is likely to belong to the same class.
 *
 * Formula (euclidean distance): d(x, y) = √((x1 - y1)² + (x2 - y2)² + ... + (xn - yn)²)
 */
class KNearestNeighbour
{
    private int $k;
    private array $trainingData;

    public function __construct(int $k)
    {
        $this->k = $k;
        $this->trainingData = [];
    }

    public function train(array $trainingData): void
    {
        $this->trainingData = $trainingData;
    }

    /**
     * @param array $sample => array of features and classes representing F1 tires.
     * Example:
     * [
     *      [
     *          'features => [6.0, 2.2, 4.0, 1.0],
     *          'class' => 'soft'
     *      ],
     *      [
     *          'features => [4.9, 3.1, 1.5, 0.1],
     *          'class' => 'medium'
     *      ],
     *
     * ]
     */
    public function predict(array $sample): string
    {
        $distances = [];

        /**
         * Calculates the Euclidean distance for each tyre
         *
         * Example above will result in:
         * [
         *      [
         *          'class' => 'soft',
         *          'distance' => 0.94868329805051,
         *      ],
         *      [
         *          'class' => 'medium',
         *          'distance' => 3.6027767069304,
         *      ],
         * ]
         */
        foreach ($this->trainingData as $data) {
            $class = $data['class'];
            $features = $data['features'];

            $distance = $this->euclideanDistance($features, $sample);

            $distances[] = ['class' => $class, 'distance' => $distance];
        }

        // Sort $distances in ascending order based on the 'distance' values of each element
        usort($distances, fn($a, $b) => $a['distance'] <=> $b['distance']);

        // Extract nearest neighbours
        $kNearest = array_slice($distances, 0, $this->k);
        $classVotes = [];

        // See which one repeats the most
        foreach ($kNearest as $neighbor) {
            $class = $neighbor['class'];
            $classVotes[$class] = ($classVotes[$class] ?? 0) + 1;
        }

        arsort($classVotes);

        // Return class with most votes
        return key($classVotes);
    }

    /**
     * The Euclidean distance formula calculates the straight-line distance between two
     * points in Euclidean space. It is commonly used as a distance metric in various algorithms.
     * The formula computes the square root of the sum of the squared differences between
     * corresponding coordinates of the two points. It measures the straight-line distance between
     * the points in the n-dimensional feature space.The Euclidean distance formula can be extended
     * to any number of dimensions and is applicable to both two-dimensional and higher-dimensional spaces.
     *
     * Formula: d(x, y) = √((x1 - y1)² + (x2 - y2)² + ... + (xn - yn)²)
     */
    private function euclideanDistance(array $features1, array $features2): float
    {
        $sum = 0;

        foreach ($features1 as $i => $feature) {
            $sum += pow($feature - $features2[$i], 2);
        }

        return sqrt($sum);
    }
}