<?php

require 'vendor/autoload.php';

// Example usage
use PhpMachinist\Models\KNearestNeighbour;

$trainingData = [
    ['features' => [6.0, 2.2, 4.0, 1.0], 'class' => 'soft'],
    ['features' => [5.7, 3.0, 4.2, 1.2], 'class' => 'soft'],
    ['features' => [5.7, 2.9, 4.2, 1.3], 'class' => 'soft'],
    ['features' => [4.9, 3.1, 1.5, 0.1], 'class' => 'medium'],
    ['features' => [5.4, 3.7, 1.5, 0.2], 'class' => 'medium'],
    ['features' => [5.0, 3.4, 1.6, 0.4], 'class' => 'medium'],
    ['features' => [6.7, 3.1, 5.6, 2.4], 'class' => 'hard'],
    ['features' => [6.3, 2.5, 5.0, 1.9], 'class' => 'hard'],
    ['features' => [6.5, 3.0, 5.2, 2.0], 'class' => 'hard'],
];

$knn = new KNearestNeighbour(3);
$knn->train($trainingData);

$sample1 = [6.1, 2.8, 4.7, 1.2];
$prediction1 = $knn->predict($sample1);

$sample2 = [5.3, 3.2, 1.3, 0.4];
$prediction2 = $knn->predict($sample2);

echo "Prediction 1: $prediction1\n";
echo "Prediction 2: $prediction2\n";